# disruption_monitoring_agent.py

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

from crewai import Agent
from tools.text_processor_tool import TextProcessorTool
from tools.disruption_analysis_tool import DisruptionAnalysisTool

class DisruptionMonitoringAgent(Agent):
    """
    Custom agent that handles:
    1) Scrape the URL (scrape_news_task) - ONLY if website_url provided AND extracted_text NOT provided
    2) Use provided text (scrape_news_task) - if extracted_text is provided (PRIORITY)
    3) Analyze disruptions (analyze_disruptions_task)
    """
    def __init__(self, company_name: str = "", **config):
        super().__init__(**config)
        # Use object.__setattr__ to bypass Pydantic validation for custom attributes
        object.__setattr__(self, 'company_name', company_name)
        # Scraping is OPTIONAL. Evaluation runs pass `extracted_text`, so scraping is not required.
        # We import the CrewAI scraping tool lazily so the framework can run even if `crewai_tools`
        # is not installed (as long as no website_url scraping is attempted).
        scrape_tool = None
        try:
            from crewai_tools import ScrapeElementFromWebsiteTool  # type: ignore
            scrape_tool = ScrapeElementFromWebsiteTool()
        except Exception as e:
            logging.warning(
                "ScrapeElementFromWebsiteTool not available (crewai_tools not installed). "
                "This is OK for evaluation runs using `extracted_text`. "
                f"Scraping via `website_url` will fail until installed. Details: {e}"
            )
        object.__setattr__(self, 'scrape_tool', scrape_tool)
        object.__setattr__(self, 'processor_tool', TextProcessorTool())
        # DisruptionAnalysisTool needs company_name for proper analysis
        # DisruptionAnalysisTool takes company_name in _run method, not __init__
        object.__setattr__(self, 'analyzer_tool', DisruptionAnalysisTool())

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute different tasks based on input keys.
        This method is called by CrewAI when the agent is assigned a task.
        
        Tasks:
        - scrape_news_task: extracted_text (PRIORITY) OR website_url -> extracted_text
        - analyze_disruptions_task: extracted_text -> disruption_analysis
        """
        logging.info(f"DisruptionMonitoringAgent.execute() called with inputs: {list(inputs.keys())}")
        
        # Log all input keys for debugging
        for key, value in inputs.items():
            if key == "extracted_text":
                logging.info(f"  {key}: {len(str(value))} characters")
            else:
                logging.info(f"  {key}: {value}")
        
        # CRITICAL: Check for disruption_analysis first to identify analyze_disruptions_task
        if "disruption_analysis" in inputs:
            # This should not happen in execute, but handle gracefully
            logging.warning("DisruptionMonitoringAgent: disruption_analysis already in inputs, skipping")
            return inputs
        
        # Task 1: Scrape news (scrape_news_task)
        # PRIORITY: Use extracted_text if provided (for synthesized scenarios)
        if "extracted_text" in inputs:
            extracted_text_value = inputs["extracted_text"]
            # Check if this is a string (the actual text) or if it's empty/None
            if extracted_text_value and isinstance(extracted_text_value, str) and len(extracted_text_value.strip()) > 0:
                # For synthesized scenarios: extracted_text provided directly
                # DO NOT SCRAPE - just pass through
                logging.info("=" * 80)
                logging.info("DisruptionMonitoringAgent: Using provided article text (synthesized scenario) - NO SCRAPING")
                logging.info(f"  Text length: {len(extracted_text_value)} characters")
                logging.info(f"  Text preview: {extracted_text_value[:200]}...")
                logging.info("=" * 80)
                return {"extracted_text": extracted_text_value}
            else:
                logging.warning(f"DisruptionMonitoringAgent: extracted_text exists but is empty or invalid: {type(extracted_text_value)}")
        
        # Task 2: Analyze disruptions (analyze_disruptions_task)
        # Check if we have extracted_text but NOT website_url, and we're in analyze phase
        # This happens when analyze_disruptions_task receives extracted_text from previous task
        if "extracted_text" in inputs and "website_url" not in inputs:
            # Check if we also have disruption_analysis context (this would be analyze task)
            # Actually, for analyze task, we should process the text
            text = inputs["extracted_text"]
            if text and isinstance(text, str) and len(text.strip()) > 0:
                logging.info("DisruptionMonitoringAgent: Analyzing disruptions from provided text.")
                # Process text first, then analyze
                cleaned = self.processor_tool.run(text=text)
                analysis = self.analyzer_tool.run(text=cleaned, company_name=self.company_name)
                # Pass through company_name if it's in inputs
                company_name = inputs.get("company_name", self.company_name)
                return {"disruption_analysis": analysis, "company_name": company_name}
        
        # Fallback: Only scrape if extracted_text is NOT provided but website_url IS provided
        if "website_url" in inputs and ("extracted_text" not in inputs or not inputs.get("extracted_text")):
            # scrape_news_task - scrape from URL (only if no extracted_text)
            url = inputs["website_url"]
            logging.warning(f"DisruptionMonitoringAgent: Scraping URL {url} (extracted_text not provided)")
            try:
                if not self.scrape_tool:
                    logging.error(
                        "Cannot scrape because ScrapeElementFromWebsiteTool is not available. "
                        "Install `crewai_tools` (or switch to providing `extracted_text`)."
                    )
                    return {"extracted_text": ""}
                # Use the scraping tool internally (not via CrewAI's tool system)
                raw_text = self.scrape_tool.run(website_url=url)
                if not raw_text:
                    logging.error(f"Failed to scrape content from {url}")
                    return {"extracted_text": ""}
                logging.info(f"Successfully scraped {len(raw_text)} characters from {url}")
                return {"extracted_text": raw_text}
            except Exception as e:
                logging.error(f"Error scraping URL {url}: {e}")
                return {"extracted_text": ""}

        # Task 2: Analyze disruptions (legacy path if cleaned_text exists)
        elif "cleaned_text" in inputs:
            # analyze_disruptions_task
            text = inputs["cleaned_text"]
            logging.info("DisruptionMonitoringAgent: Analyzing disruptions from cleaned_text.")
            analysis = self.analyzer_tool.run(text=text, company_name=self.company_name)
            return {"disruption_analysis": analysis}

        else:
            error_msg = f"Invalid input for DisruptionMonitoringAgent. Input keys: {list(inputs.keys())}"
            logging.error(error_msg)
            # Return empty extracted_text as fallback
            return {"extracted_text": ""}
