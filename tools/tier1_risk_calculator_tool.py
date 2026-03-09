# tools/tier1_risk_calculator_tool.py
"""
Tier-1 Risk Calculator Tool - Returns ONLY Tier-1 supplier risk scores
to avoid LLM context overflow while maintaining agentic decision-making.

This tool:
1. Performs heavy computation internally (all 1,077 companies if needed)
2. Aggregates downstream risks to Tier-1 suppliers
3. Returns ONLY actionable Tier-1 summary to the LLM
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Tier1RiskCalculatorInput(BaseModel):
    """Input schema for Tier-1 risk calculation"""

    disruption_analysis: Dict[str, Any] = Field(
        ..., description="Disruption analysis containing affected countries/companies"
    )
    kg_results: Dict[str, Any] = Field(
        ..., description="Knowledge graph results with disrupted supply chains (Tier-1 to Tier-4)"
    )
    company_name: str = Field(..., description="Monitored company name (e.g., Tesla)")


def calculate_tier1_risk_scores(
    disruption_analysis: Dict[str, Any],
    kg_results: Dict[str, Any],
    company_name: str,
) -> Dict[str, Any]:
    """
    Calculate risk scores ONLY for Tier-1 suppliers by aggregating downstream exposure.
    
    This function:
    - Performs all heavy computation internally (graph metrics for 1000+ companies)
    - Aggregates downstream disruption impact to Tier-1
    - Returns ONLY Tier-1 summary (typically 5-20 suppliers)
    
    Returns:
        Compact JSON with ONLY Tier-1 suppliers and their aggregated risk scores.
    """
    try:
        # Import the actual agent logic
        from agents.enhanced_risk_manager_agent import EnhancedRiskManagerAgent

        # Create agent instance
        agent = EnhancedRiskManagerAgent(
            company_name=company_name,
            config={
                "role": "Risk Calculator",
                "goal": "Calculate Tier-1 risk scores",
                "backstory": "Internal computation engine",
            },
        )

        # Call execute() - this does ALL the heavy computation internally
        result = agent.execute(
            {
                "disruption_analysis": disruption_analysis,
                "kg_results": kg_results,
                "company_name": company_name,
            }
        )

        risk_assessment = result.get("risk_assessment", {})

        # Extract ONLY Tier-1 summary for LLM context
        tier1_summary = {
            "company_name": risk_assessment.get("company_name"),
            "timestamp": risk_assessment.get("timestamp"),
            "disruption_summary": risk_assessment.get("disruption_summary", {}),
            "tier1_supplier_count": len(risk_assessment.get("supplier_risk_scores", {})),
            "supplier_risk_scores": risk_assessment.get(
                "supplier_risk_scores", {}
            ),  # Only Tier-1
            "critical_suppliers": risk_assessment.get("critical_suppliers", []),
            "risk_metrics_summary": risk_assessment.get("risk_metrics_summary", {}),
            "executive_summary": risk_assessment.get("executive_summary", {}),
            "methodology": {
                "description": "Tier-1 focused risk assessment aggregating downstream (Tier-2 to Tier-4) disruption impact",
                "metrics": [
                    "Exposure Depth: Max tier of disruption in supplier's chain",
                    "Exposure Breadth: Count of unique disrupted nodes downstream",
                    "Downstream Criticality: Weighted importance of disrupted nodes",
                    "Supplier Centrality: Network importance of Tier-1 supplier",
                ],
                "thresholds": risk_assessment.get("methodology", {}).get(
                    "risk_thresholds", {}
                ),
            },
        }

        logger.info(
            f"✅ Tier-1 risk calculation complete: {tier1_summary['tier1_supplier_count']} suppliers"
        )

        return {
            "success": True,
            "tier1_risk_assessment": tier1_summary,
            "message": f"Calculated risk scores for {tier1_summary['tier1_supplier_count']} Tier-1 suppliers",
        }

    except Exception as e:
        logger.error(f"Failed to calculate Tier-1 risk scores: {e}")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "tier1_risk_assessment": {},
        }


# Create the LangChain tool
tier1_risk_calculator_tool = StructuredTool(
    name="tier1_risk_calculator",
    description="""
    Calculate comprehensive risk scores for ALL Tier-1 suppliers by analyzing their 
    downstream supply chain exposure (Tier-2, Tier-3, Tier-4).
    
    This tool:
    - Processes disrupted supply chains from kg_results
    - Computes graph metrics for all suppliers (centrality, dependency, PageRank)
    - Aggregates downstream disruption impact to Tier-1 level
    - Returns ONLY Tier-1 supplier risk scores (actionable level)
    
    Use this tool ONCE to get all Tier-1 risk scores, then interpret and summarize
    the results for the Chief Supply Chain Officer.
    
    Input:
        - disruption_analysis: Output from disruption monitoring agent
        - kg_results: Full supply chain paths from KG query agent
        - company_name: Monitored company (e.g., Tesla)
    
    Output:
        Compact JSON with ONLY Tier-1 suppliers, their risk scores (0.0-1.0),
        risk levels (CRITICAL/HIGH/MEDIUM/LOW), and executive summary.
    """,
    func=calculate_tier1_risk_scores,
    args_schema=Tier1RiskCalculatorInput,
    return_direct=False,
)


