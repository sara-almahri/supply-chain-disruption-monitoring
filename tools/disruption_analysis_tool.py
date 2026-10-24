import openai
import os
import json
import logging
import re
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)

class DisruptionAnalysisInput(BaseModel):
    """Schema for DisruptionAnalysisTool."""
    text: str = Field(..., description="Clean text content of the article.")
    company_name: str = Field(..., description="Company name to tailor the analysis.")

class DisruptionAnalysisTool(BaseTool):
    name: str = "DisruptionAnalysisTool"
    description: str = "Analyzes supply chain disruptions and classifies the impact."
    args_schema: type = DisruptionAnalysisInput

    def _run(self, text: str, company_name: str) -> dict:
        """Processes text through OpenAI to analyze supply chain disruptions."""
        if not text.strip():
            logging.error("🚨 No valid text provided for analysis.")
            return {"error": "No valid text provided for analysis."}

        # Ensure OpenAI API key is set
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            logging.error("🚨 OpenAI API key is missing. Set it in environment variables.")
            return {"error": "Missing OpenAI API key."}

        try:
            # ✅ Build the structured prompt dynamically
            prompt = self.build_prompt(text, company_name)
            logging.info(f"🚀 Sending prompt to OpenAI API for {company_name}...")

            # ✅ Call OpenAI API
            response = self.call_openai_api(prompt)

            # ✅ Validate response
            if not response:
                logging.error("🚨 Empty response from OpenAI API.")
                return {"error": "OpenAI API returned an empty response."}

            # ✅ Extract structured JSON (pass company_name to replace placeholders)
            structured_output = self.extract_json(response, company_name=company_name)
            if structured_output:
                return structured_output
            else:
                logging.error("🚨 Failed to extract JSON from OpenAI response.")
                return {"error": "Failed to extract structured JSON from OpenAI output."}

        except Exception as e:
            logging.error(f"🚨 Error in DisruptionAnalysisTool: {e}")
            return {"error": f"Processing failed: {str(e)}"}

            

    def call_openai_api(self, prompt: str) -> str:
        """Calls OpenAI's GPT-4 API with a structured prompt."""
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a supply chain risk expert in the automotive industry."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=16000
            )

            output_text = response.choices[0].message.content.strip()
            logging.info(f"✅ OpenAI response received successfully.")

            # Explicitly close SSL connection
            del client  # Forces garbage collection to clean up the request
            
            return output_text
        except Exception as e:
            logging.error(f"🚨 OpenAI API request failed: {e}")
            return ""

    def extract_json(self, response: str, company_name: str = None) -> dict:
        """Extracts structured JSON safely from LLM response and ensures proper handling of empty fields."""
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                
                # Replace {company_name} placeholder with actual company name if provided
                if company_name:
                    json_str = json_str.replace("{company_name}", company_name)
                    json_str = json_str.replace("{{company_name}}", company_name)
                
                parsed_json = json.loads(json_str)

                # Ensure companies field is properly formatted
                if "companies" in parsed_json.get("involved", {}):
                    companies = parsed_json["involved"]["companies"]
                    if not companies or any("Company" in str(c) for c in companies):
                        parsed_json["involved"]["companies"] = []  # Set to empty list if invalid

                # Replace {company_name} in questions array if it still exists
                if company_name and "questions" in parsed_json:
                    questions = parsed_json["questions"]
                    if isinstance(questions, list):
                        parsed_json["questions"] = [
                            q.replace("{company_name}", company_name).replace("{{company_name}}", company_name)
                            if isinstance(q, str) else q
                            for q in questions
                        ]

                return parsed_json
            else:
                return None
        except json.JSONDecodeError as e:
            logging.error(f"🚨 JSON parsing failed: {e}")
            return None



    def build_prompt(self, article_text: str, company_name: str) -> str:

        """Constructs an advanced prompt for LLM analysis with structured JSON output."""
        examples = """
        ***** Example 1: *****
        Article: "A severe earthquake in Japan has disrupted manufacturing facilities..."
        - Expert Thought 1: As a supply chain risk expert working in an **automotive company**, I immediately recognize the **critical dependencies our company and the **automotive industry** has on Japanese suppliers**. Japan is a key **supplier of automotive semiconductors, engine components, and lithium-ion batteries**. A major earthquake will likely disrupt **our Tier-1 or Tier-2 suppliers**, potentially leading to **delays in vehicle production**.
        - Expert Thought 2: I must immediately assess **which of our direct suppliers (Tier-1) are based in Japan** and whether any of them **source subcomponents from affected regions**. A disruption at the **Tier-2 or Tier-3 level can create a cascading failure** affecting our production schedules, especially if alternative suppliers do not exist.
        - Expert Thought 3: I will first start by tracing our Tier-1 suppliers. Then systematically think of the extend supply chain. If our **Tier-1 suppliers rely on Japanese semiconductor or electronic component manufacturers**, this event could **delay the delivery of critical parts** (e.g., **ECUs, sensors, lithium-ion batteries**), forcing us to either find alternative suppliers or slow down production.
        - Expert Thought 4: Based on facts and knowledge and my expertise, I know that Toyota, Honda, Nissaan and ... are Japansese automtive manufactureres. Those companies may definetly be impacted by the disruption. Also, manufactureres supplying those companies that are based in Japan such as ...... will be impacted!
        *** Disruption Analysis ***
        {
        "type": "Natural Disaster",
        "involved": {
            "countries": ["Japan"],
            "industries": ["Automotive"],
            "companies": ["Toyota", "Honda", "Denso", "Renesas", "Nissan"]
        },
        "questions": [
            "Which of {company_name}'s Tier-1 suppliers are located in Japan? Whic industry do they operate in?",
            "Does any of {company_name}'s Tier-1 suppliers have suppliers in Japan? Who are they and what industries do they operate in?",
            "Does any of {company_name}'s Tier-2 suppliers have suppliers in Japan? Who are they and what industries do they operate in?"

        ],
        "summary": "A severe earthquake in Japan has disrupted key automotive manufacturing facilities. Given Japan’s role as a major supplier of semiconductors, sensors, and lithium-ion batteries, our company must immediately assess its Tier-1 and Tier-2 supplier dependencies to mitigate potential production delays."
        }

        ***** Example 2: *****
        Article: "US imposes trade restrictions on Chinese semiconductor companies..."
        - Expert Thought 1: As a supply chain expert in **an automotive company**, I recognize that modern vehicles depend **heavily on semiconductors**. These trade restrictions may **severely disrupt our supply chain**, especially if our suppliers **source microchips from China**.
        - Expert Thought 2: Our **Tier-1 suppliers may rely on Tier-2 or Tier-3 manufacturers in China**, particularly for **MCUs, ECUs, and other critical automotive chips**. If those manufacturers **can no longer export** due to US restrictions, we must immediately find **alternative sources** (Taiwan, South Korea, or domestic suppliers).
        - Expert Thought 3: I will first start by tracing our Tier-1 suppliers. Then systematically think of the extend supply chain. Without intervention, this disruption **could lead to a semiconductor shortage**, delaying **vehicle production** and increasing procurement costs. **We must assess which of our key suppliers are directly or indirectly dependent on China.**

        *** Disruption Analysis ***
        {
        "type": "Geopolitical",
        "involved": {
            "countries": ["USA", "China"],
            "industries": ["Automotive"],
            "companies": ["TSMC", "Nvidia", "Qualcomm", "Infineon", "Bosch"]
        },
        "questions": [
            "Which of {company_name}'s Tier-1 suppliers are in the semiconductor industry and are located in China?",
            "Does any of {company_name}'s Tier-1 suppliers have suppliers in China? Who are they and what industries do they operate in?",
            "Does any of {company_name}'s Tier-2 suppliers have suppliers in China? Who are they and what industries do they operate in?"
        ],
        "summary": "US trade restrictions on Chinese semiconductor firms could disrupt our automotive supply chain, leading to potential semiconductor shortages. Given our reliance on microchips for ECUs, sensors, and control units, we must assess supplier dependencies and identify alternative sources to maintain production stability."
        }
        """

        prompt = (
            f"**CRITICAL CONTEXT: You are working FOR {company_name} (the company we are monitoring and protecting).**\n\n"
            f"You are a **top-tier expert** in **supply chain risk management**, working FOR *******{company_name}*********** specializing in **disruptions, ripple effects, and industry interdependencies**. "
            f"Your job is to analyze disruptions and determine if they affect {company_name}'s supply chain network.\n\n"
            f"You have extensive experience analyzing news and **logically mapping disruptions to supply chains**.\n\n"
            f"**YOUR MISSION:** Analyze the news article to identify disruptions (countries, companies, industries) and determine if they could affect {company_name}'s supply chain network (up to Tier-4).\n\n"

            f"Analyze the following news article with **extreme expert-level reasoning** and extract key details in a structured format. "
            f"Use a **step-by-step chain of thought** before generating **structured JSON output**.\n\n"

            f"### **Instructions:**\n"
            f"1️⃣ **Understand the article** in extreme detail from a **supply chain risk management** perspective for {company_name}.\n"
            f"   - Identify: What disruption occurred? Where? Which companies/industries are affected?\n"
            f"   - Key question: Could this disruption affect {company_name}'s suppliers (Tier-1 to Tier-4)?\n\n"
            f"2️⃣ **Classify the disruption type** (Geopolitical, Trade Policy, Natural Disaster, Company Bankruptcy, Economic Crisis, Other).\n"
            f"3️⃣ **Generate at least 3 deep expert thoughts** about how this disruption could affect {company_name}:\n"
            f"   - Which countries are disrupted? Do {company_name}'s suppliers operate there?\n"
            f"   - Which industries are disrupted? Does {company_name} depend on those industries?\n"
            f"   - Which companies are disrupted? Are they in {company_name}'s supply chain network?\n\n"
            f"4️⃣ **Extract key impacted elements**: countries, industries, companies.\n"
            f"   - These are the disrupted entities - we will check if they exist in {company_name}'s network\n\n"
            f"5️⃣ **Formulate actionable supply chain risk questions** (but **only for queries the database can answer**):\n"
            f"   - **The database contains:** companies, supplier relationships (up to Tier-4), locations, and industries.\n"
            f"   - **DO NOT generate invalid questions beyond database capabilities.**\n"
            f"   - **Questions must reference {company_name} explicitly** - we're checking {company_name}'s network.\n"
            f"   - Example: CORRECT! *'Which of {company_name}'s Tier-1 suppliers are located in Argentina?'*\n"
            f"   - WRONG! **Avoid generic questions like 'Which suppliers are in Argentina?'**\n"
            f"   - **Focus on:** Finding suppliers in {company_name}'s network that are in disrupted countries/regions.\n"
            
            f"6️⃣ **Output findings in strict JSON format** with the following structure:\n\n"
            f"{{\n"
            f'  "type": "Disruption Type",\n'
            f'  "involved": {{\n'
            f'    "countries": ["Country1", "Country2"],\n'
            f'    "industries": ["Industry1", "Industry2"],\n'
            f'    "companies": ["Company1", "Company2"]\n'
            f"  }},\n"
            f'  "questions": [\n'
            f'    "Database-supported question 1",\n'
            f'    "Database-supported question 2",\n'
            f'    "Database-supported question 3"\n'\
            f"  ],\n"
            f'  "summary": "Expert summary of the disruption and its supply chain impact."\n'
            f"}}\n"

            f"\n### **OUTPUT RULES (STRICT)**\n"
            f"- Respond with **ONLY** the JSON object described above.\n"
            f"- **Do NOT** include markdown fences, commentary, or additional text.\n"
            f"- If a field has no values, return an empty array (e.g., \"companies\": []).\n"
            f"- Ensure arrays contain unique, factual entries mentioned in the article.\n"


            f"### **Few-Shot Examples for Reference:**\n"
            f"{examples}\n\n"

            f"### **Now, analyze this new article FOR {company_name}:**\n"
            f"**Article:**\n"
            f"\"\"\"\n{article_text}\n\"\"\"\n\n"
            f"**REMEMBER:**\n"
            f"- You are working FOR {company_name} - protecting {company_name}'s supply chain\n"
            f"- Identify disruptions (countries, companies, industries)\n"
            f"- The next step will check if these disrupted entities exist in {company_name}'s supply chain network\n"
            f"- Questions must be direct and reference {company_name} explicitly\n"
            f"- Focus on: 'Which of {company_name}'s suppliers are in disrupted regions?'\n\n"
            f"💡 **NOTE: Questions must be direct, simple, and reference {company_name}'s Tier-1, Tier-2, Tier-3, Tier-4 suppliers**\n"
            f"💡 **Think step-by-step: How could this disruption affect {company_name}?**\n"
            f"💡 **Generate the JSON output with disruption details.**"
                 )

        return prompt