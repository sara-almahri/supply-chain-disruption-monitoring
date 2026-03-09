"""
Risk Assessment Orchestration Tools
====================================
Two-step process to calculate Tier-1 risk scores without LLM token limits:
1. calculate_and_save_risks: Execute all risk calculations, save to disk, return summary
2. get_saved_risk_assessment: Load complete risk data for final output
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TEMP_RISK_DIR = Path(__file__).parent.parent / "output" / "temp" / "risk_payloads"


class CalculateAndSaveRisksInput(BaseModel):
    """Input for calculate_and_save_risks tool"""
    disruption_analysis: Dict[str, Any] = Field(..., description="Disruption analysis from monitoring agent")
    kg_results_path: str = Field(..., description="Path to saved KG results (from build_and_save_kg)")
    company_name: str = Field(..., description="Monitored company name (e.g., 'Tesla')")
    scenario_id: str = Field(default="default", description="Scenario identifier")


class GetSavedRiskAssessmentInput(BaseModel):
    """Input for get_saved_risk_assessment tool"""
    risk_payload_path: str = Field(..., description="Path to saved risk assessment payload")


def calculate_and_save_risks(
    disruption_analysis: Dict[str, Any],
    kg_results_path: str,
    company_name: str,
    scenario_id: str = "default",
) -> Dict[str, Any]:
    """
    Step 1: Calculate complete Tier-1 risk assessment and save to disk.
    Returns ONLY summary (small) - not the full risk data.
    
    This function:
    1. Loads complete KG data from disk (all 601 chains)
    2. Executes EnhancedRiskManagerAgent.execute() (Python-only, no LLM)
    3. Calculates risk scores for ALL Tier-1 suppliers
    4. Saves complete results to disk
    5. Returns metadata summary (critical suppliers, counts, path)
    """
    TEMP_RISK_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(
        f"[calculate_and_save_risks] Calculating Tier-1 risks for {company_name}, "
        f"KG data at: {kg_results_path}"
    )
    
    # Load complete KG data from disk
    try:
        kg_data = json.loads(Path(kg_results_path).read_text(encoding="utf-8"))
        logger.info(
            f"[calculate_and_save_risks] Loaded KG data: "
            f"{len(kg_data.get('tier_1', []))} T1, {len(kg_data.get('tier_2', []))} T2, "
            f"{len(kg_data.get('tier_3', []))} T3, {len(kg_data.get('tier_4', []))} T4"
        )
    except Exception as exc:
        logger.error(f"[calculate_and_save_risks] Failed to load KG data: {exc}")
        return {
            "success": False,
            "error": f"Failed to load KG data from {kg_results_path}: {exc}",
        }
    
    # Execute risk calculation (Python-only, using EnhancedRiskManagerAgent.execute())
    try:
        from agents.enhanced_risk_manager_agent import EnhancedRiskManagerAgent
        
        # Create a temporary risk agent instance for computation
        risk_agent = EnhancedRiskManagerAgent(
            role="Risk Calculator",
            goal="Calculate comprehensive Tier-1 risk scores",
            backstory="Internal computation agent",
            verbose=False,
        )
        
        # Execute the risk calculation (direct Python, no LLM)
        inputs = {
            "disruption_analysis": disruption_analysis,
            "kg_results": kg_data,  # Full complete data
            "company_name": company_name,
        }
        
        logger.info("[calculate_and_save_risks] Executing risk calculation...")
        risk_assessment = risk_agent.execute(inputs)
        
        logger.info(
            f"[calculate_and_save_risks] Risk calculation complete. "
            f"Tier-1 suppliers: {len(risk_assessment.get('supplier_risk_scores', {}))}"
        )
        
    except Exception as exc:
        logger.error(f"[calculate_and_save_risks] Risk calculation failed: {exc}", exc_info=True)
        return {
            "success": False,
            "error": f"Risk calculation failed: {exc}",
        }
    
    # Save complete risk assessment to disk
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_company = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in company_name)
    filename = f"{scenario_id}_{safe_company}_{timestamp}_risk.json"
    file_path = TEMP_RISK_DIR / filename
    
    try:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(risk_assessment, f, indent=2, default=str)
        
        logger.info(f"[calculate_and_save_risks] Saved risk assessment to {file_path}")
    except Exception as exc:
        logger.error(f"[calculate_and_save_risks] Failed to save risk assessment: {exc}")
        return {
            "success": False,
            "error": f"Failed to save risk assessment: {exc}",
        }
    
    # Extract summary metadata (small, LLM-friendly)
    supplier_risk_scores = risk_assessment.get("supplier_risk_scores", {})
    critical_suppliers = risk_assessment.get("critical_suppliers", [])
    tier1_risk_summary = risk_assessment.get("tier1_risk_summary", {})
    
    # Calculate risk level counts
    high_critical_count = 0
    for supplier_name, tier1_profile in tier1_risk_summary.items():
        risk_level = tier1_profile.get("risk_level", "").lower()
        if risk_level in ("high", "critical"):
            high_critical_count += 1
    
    metadata = {
        "success": True,
        "risk_payload_path": str(file_path),
        "company_name": company_name,
        "tier1_supplier_count": len(supplier_risk_scores),
        "critical_suppliers": critical_suppliers[:5],  # Top 5 for summary
        "high_critical_count": high_critical_count,
        "top_3_risks": dict(sorted(supplier_risk_scores.items(), key=lambda x: x[1], reverse=True)[:3]),
        "message": (
            f"Calculated risk scores for {len(supplier_risk_scores)} Tier-1 suppliers. "
            f"{high_critical_count} require immediate attention. "
            f"Saved to {file_path}"
        ),
    }
    
    logger.info(
        f"[calculate_and_save_risks] Summary: {len(supplier_risk_scores)} Tier-1 suppliers, "
        f"{high_critical_count} high/critical"
    )
    
    return metadata


def get_saved_risk_assessment(risk_payload_path: str) -> Dict[str, Any]:
    """
    Step 2: Load the saved risk assessment from disk for final output.
    This returns the FULL risk data structure without streaming through LLM.
    """
    try:
        file_path = Path(risk_payload_path)
        if not file_path.exists():
            logger.error(f"[get_saved_risk_assessment] File not found: {risk_payload_path}")
            return {
                "error": f"Risk assessment file not found: {risk_payload_path}",
                "supplier_risk_scores": {},
                "tier1_risk_profiles": {},
                "critical_suppliers": [],
            }
        
        risk_data = json.loads(file_path.read_text(encoding="utf-8"))
        
        logger.info(
            f"[get_saved_risk_assessment] Loaded risk assessment from {risk_payload_path}: "
            f"{len(risk_data.get('supplier_risk_scores', {}))} Tier-1 suppliers"
        )
        
        return risk_data
        
    except Exception as exc:
        logger.error(f"[get_saved_risk_assessment] Failed to load risk data: {exc}")
        return {
            "error": str(exc),
            "supplier_risk_scores": {},
            "tier1_risk_profiles": {},
            "critical_suppliers": [],
        }


# Create the tools
calculate_and_save_risks_tool = StructuredTool(
    name="calculate_and_save_risks",
    description=(
        "Step 1: Calculate comprehensive Tier-1 risk scores for all suppliers affected by disruption. "
        "This tool loads the complete KG data, executes risk calculations (exposure depth, breadth, "
        "criticality, centrality), and saves results to disk. "
        "Returns ONLY summary metadata (path, counts, top risks) - not the full risk data. "
        "Use this first, then call get_saved_risk_assessment to retrieve the full data."
    ),
    func=calculate_and_save_risks,
    args_schema=CalculateAndSaveRisksInput,
    return_direct=False,
)

get_saved_risk_assessment_tool = StructuredTool(
    name="get_saved_risk_assessment",
    description=(
        "Step 2: Load the complete risk assessment from disk (saved by calculate_and_save_risks). "
        "This returns the FULL risk structure with all Tier-1 supplier scores, profiles, and critical suppliers. "
        "Use the risk_payload_path from calculate_and_save_risks output. "
        "This bypasses LLM token limits by loading directly from disk."
    ),
    func=get_saved_risk_assessment,
    args_schema=GetSavedRiskAssessmentInput,
    return_direct=True,  # Return directly without LLM processing
)

