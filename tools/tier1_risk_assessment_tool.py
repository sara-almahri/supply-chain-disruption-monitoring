# tools/tier1_risk_assessment_tool.py
"""
Tier-1 Risk Assessment Tool - Agentic Version

This tool performs heavy computation internally but returns ONLY Tier-1 supplier
risk scores (typically 5-50 companies) to keep LLM context manageable while
allowing the agent to reason about and interpret the results.

Key Design:
- Input: disruption_analysis + kg_results
- Internal: Compute metrics for ALL suppliers (Tier-1 to Tier-4)
- Output: ONLY Tier-1 aggregated risk scores + summary
- LLM Agent: Interprets results, identifies priorities, summarizes for next agent
"""

import json
import logging
from typing import Any, Dict

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tools.full_supply_chain_path_tool import build_disrupted_supply_chains

logger = logging.getLogger(__name__)


class Tier1RiskAssessmentInput(BaseModel):
    """Input schema for Tier-1 risk assessment tool"""

    disruption_analysis: Dict[str, Any] = Field(
        ..., description="Disruption analysis from monitoring agent"
    )
    kg_results: Dict[str, Any] = Field(
        ..., description="Supply chain paths from KG query agent (Tier-1 to Tier-4)"
    )
    company_name: str | None = Field(
        None, description="Monitored company name (e.g., Tesla). If omitted, the tool will attempt to infer it."
    )


def assess_tier1_risks(
    disruption_analysis: Dict[str, Any],
    kg_results: Dict[str, Any],
    company_name: str | None = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive risk scores for ALL Tier-1 suppliers.
    
    This tool:
    1. Performs heavy computation internally (graph metrics for 1000+ companies)
    2. Aggregates downstream disruption impact (Tier-2, Tier-3, Tier-4) to Tier-1
    3. Returns ONLY Tier-1 summary (typically 5-50 suppliers) for LLM reasoning
    
    The LLM agent can then:
    - Interpret the risk scores
    - Identify critical suppliers requiring immediate attention
    - Summarize findings for the Chief Supply Chain Officer
    - Adapt recommendations based on specific disruption context
    
    Returns:
        Compact JSON with ONLY Tier-1 suppliers and their risk scores.
        Size: ~5-50 suppliers (manageable for LLM context)
    """
    try:
        # Resolve company name (LLM might forget to pass it)
        resolved_company = company_name or disruption_analysis.get("company_name")
        if not resolved_company:
            resolved_company = kg_results.get("monitored_company")

        if not resolved_company:
            logger.warning(
                "tier1_risk_assessment_tool: Missing company_name; defaulting to 'Unknown Company'"
            )
            resolved_company = "Unknown Company"

        if not isinstance(kg_results, dict):
            kg_results = {}

        # CRITICAL: Check if kg_results is a metadata dict pointing to saved payload
        # If so, load the complete data from disk (fast, no LLM, complete)
        kg_results_path = kg_results.get("kg_results_path")
        if isinstance(kg_results_path, str):
            from pathlib import Path
            try:
                saved_payload = json.loads(Path(kg_results_path).read_text())
                if isinstance(saved_payload, dict):
                    logger.info(
                        f"[tier1_risk_assessment_tool] Loading complete KG data from {kg_results_path}"
                    )
                    kg_results = saved_payload
                else:
                    logger.warning(
                        f"[tier1_risk_assessment_tool] Saved payload at {kg_results_path} is not a dict"
                    )
            except FileNotFoundError:
                logger.warning(
                    f"[tier1_risk_assessment_tool] Saved kg_results file not found: {kg_results_path}"
                )
            except Exception as exc:
                logger.error(
                    f"[tier1_risk_assessment_tool] Failed to load saved kg_results: {exc}"
                )

        # Ensure kg_results structure is complete (fallback: recompute if truncated)
        def _chains_are_truncated(chains: Any) -> bool:
            if not chains:
                return False
            if not isinstance(chains, list):
                return True
            for chain in chains:
                if not isinstance(chain, list):
                    return True
                for node in chain:
                    if not isinstance(node, dict):
                        return True
            return False

        # Check if we have actual chain data (not just metadata)
        has_actual_chains = (
            isinstance(kg_results.get("tier_2"), list)
            or isinstance(kg_results.get("tier_3"), list)
            or isinstance(kg_results.get("tier_4"), list)
        )

        if not has_actual_chains or _chains_are_truncated(kg_results.get("tier_2")) or _chains_are_truncated(
            kg_results.get("tier_3")
        ) or _chains_are_truncated(kg_results.get("tier_4")):
            involved = (
                disruption_analysis.get("involved", {})
                if isinstance(disruption_analysis, dict)
                else {}
            )
            disrupted_countries = (
                involved.get("countries", []) if isinstance(involved, dict) else []
            )
            disrupted_companies = (
                involved.get("companies", []) if isinstance(involved, dict) else []
            )
            if disrupted_countries or disrupted_companies:
                logger.info(
                    "[tier1_risk_assessment_tool] Detected incomplete kg_results; recomputing full supply chain."
                )
                kg_results = build_disrupted_supply_chains(
                    monitored_company=resolved_company,
                    disrupted_countries=disrupted_countries,
                    disrupted_companies=disrupted_companies,
                )
                kg_results.setdefault("monitored_company", resolved_company)
                if disrupted_countries:
                    kg_results.setdefault("disrupted_countries", disrupted_countries)
            else:
                logger.warning(
                    "[tier1_risk_assessment_tool] kg_results incomplete but insufficient disruption context to recompute."
                )

        # Import the computation engine
        from agents.enhanced_risk_manager_agent import EnhancedRiskManagerAgent

        # Create computation engine (no LLM)
        calculator = EnhancedRiskManagerAgent(
            company_name=resolved_company,
            config={
                "role": "Risk Calculator",
                "goal": "Compute Tier-1 risk scores",
                "backstory": "Computation engine",
            },
        )

        # Perform ALL heavy computation internally
        result = calculator.execute(
            {
                "disruption_analysis": disruption_analysis,
                "kg_results": kg_results,
                "company_name": resolved_company,
            }
        )

        raw_assessment = result.get("risk_assessment")
        risk_assessment = dict(raw_assessment) if isinstance(raw_assessment, dict) else {}

        logger.info(
            "✅ Tier-1 risk assessment: %d suppliers analyzed",
            len(risk_assessment.get("supplier_risk_scores", {}) or {}),
        )

        return risk_assessment

    except Exception as e:
        logger.error(f"Failed to assess Tier-1 risks: {e}")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "tier1_risk_assessment": {},
            "message": f"Risk assessment failed: {str(e)}",
        }


# Create the LangChain tool for the agentic Risk Manager
tier1_risk_assessment_tool = StructuredTool(
    name="tier1_risk_assessment",
    description="""
    **Comprehensive Tier-1 Supplier Risk Assessment Tool**
    
    Use this tool to calculate risk scores for ALL Tier-1 suppliers by analyzing
    their downstream supply chain exposure (Tier-2, Tier-3, Tier-4).
    
    What this tool does:
    - Computes graph metrics for the entire supply chain network
    - Aggregates downstream disruption impact to each Tier-1 supplier
    - Returns risk scores (0.0-1.0) and risk levels (CRITICAL/HIGH/MEDIUM/LOW)
    - Identifies which Tier-1 suppliers require immediate attention
    
    Input Required:
        - disruption_analysis: Output from the Disruption Monitoring agent
        - kg_results: Supply chain paths from the KG Query agent
        - company_name: Monitored company (e.g., "Tesla")
    
    Output (Compact JSON):
        - supplier_risk_scores: Dict of {Tier-1 company: risk_score}
        - tier1_risk_profiles: Detailed breakdown for each Tier-1 supplier
        - critical_suppliers: List of suppliers flagged CRITICAL or HIGH
        - risk_summary: Count of suppliers per risk level
    
    After calling this tool:
    1. Review the supplier_risk_scores to understand overall exposure
    2. Focus on critical_suppliers for immediate attention
    3. Use tier1_risk_profiles for detailed risk breakdown
    4. Summarize findings and provide recommendations for the Chief Supply Chain Officer
    
    Call this tool ONCE per scenario to get all Tier-1 risk data.
    """,
    func=assess_tier1_risks,
    args_schema=Tier1RiskAssessmentInput,
    return_direct=False,  # LLM interprets the results
)

