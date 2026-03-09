"""
Tier-1 Comprehensive Risk Calculation Tool
===========================================
This tool encapsulates the EXACT same logic used in ground truth generation
to calculate Tier-1 risk scores. It performs heavy computation in Python
and returns the complete risk assessment.

The Risk Manager Agent orchestrates this tool — it decides WHEN to call
it and returns the output. The tool handles all data loading and computation.

Architecture:
    1. The KG agent saves KG data + metadata to disk at a predictable path
       keyed by scenario_id (e.g., {scenario_id}_meta.json).
    2. This tool accepts just scenario_id (a simple string), loads all
       required data from disk, and computes risk scores deterministically.
    3. This file-based handoff avoids unreliable LLM serialization of
       complex structured data through tool call arguments.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TEMP_KG_DIR = Path(__file__).parent.parent / "output" / "temp" / "kg_payloads"


class Tier1RiskCalculationInput(BaseModel):
    """Input schema for Tier-1 risk calculation.
    
    Only scenario_id is required. The tool loads all data from disk.
    """
    scenario_id: str = Field(
        ...,
        description="Scenario identifier (e.g., 'BMW_SC001'). "
                    "The tool loads KG data and disruption metadata from disk using this ID."
    )
    company_name: str = Field(
        default="",
        description="Monitored company name (e.g., 'Tesla'). "
                    "If not provided, will be derived from saved metadata."
    )


def calculate_tier1_comprehensive_risks(
    scenario_id: str,
    company_name: str = "",
) -> Dict[str, Any]:
    """
    Calculate comprehensive Tier-1 risk assessment using the SAME logic as ground truth.
    
    Architecture:
        The KG agent (upstream) saves KG data and metadata to a predictable path.
        This tool loads everything from disk using scenario_id — no LLM serialization needed.
    
    Steps:
        1. Load metadata from {scenario_id}_meta.json (contains kg_payload_path, countries, etc.)
        2. Load full KG data from the kg_payload_path referenced in metadata
        3. Compute risk scores using GroundTruthGenerator (100% deterministic Python)
        4. Return complete risk assessment for ALL Tier-1 suppliers
    
    Returns:
        Complete risk assessment dict with supplier_risk_scores, tier1_risk_profiles,
        critical_suppliers, methodology, etc.
    """
    try:
        # ---- Step 1: Load metadata saved by the KG agent ----
        meta_path = TEMP_KG_DIR / f"{scenario_id}_meta.json"
        
        if not meta_path.exists():
            logger.error(
                f"[calculate_tier1_comprehensive_risks] Meta file not found: {meta_path}. "
                f"The KG agent must run first and save metadata."
            )
            return {
                "error": f"Meta file not found for scenario {scenario_id}. "
                         f"Expected at {meta_path}",
                "supplier_risk_scores": {},
                "tier1_risk_profiles": [],
                "critical_suppliers": [],
            }
        
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info(
            f"[calculate_tier1_comprehensive_risks] Loaded metadata for {scenario_id}: "
            f"company={meta.get('monitored_company')}, "
            f"chains={meta.get('total_chains')}"
        )
        
        # ---- Step 2: Load full KG data from the path in metadata ----
        kg_payload_path = meta.get("kg_payload_path", "")
        if not kg_payload_path or not Path(kg_payload_path).exists():
            logger.error(
                f"[calculate_tier1_comprehensive_risks] KG payload file not found: "
                f"{kg_payload_path}"
            )
            return {
                "error": f"KG payload file not found: {kg_payload_path}",
                "supplier_risk_scores": {},
                "tier1_risk_profiles": [],
                "critical_suppliers": [],
            }
        
        actual_kg_results = json.loads(
            Path(kg_payload_path).read_text(encoding="utf-8")
        )
        
        chain_counts = {
            f"tier_{t}": len(actual_kg_results.get(f"tier_{t}", []))
            for t in range(1, 5)
        }
        logger.info(
            f"[calculate_tier1_comprehensive_risks] Loaded KG data: "
            f"T1={chain_counts['tier_1']}, T2={chain_counts['tier_2']}, "
            f"T3={chain_counts['tier_3']}, T4={chain_counts['tier_4']}"
        )
        
        # ---- Step 3: Resolve company name ----
        if not company_name or company_name.strip() == "":
            company_name = meta.get("monitored_company", "")
        if not company_name:
            company_name = actual_kg_results.get("monitored_company", "Unknown Company")
        
        # ---- Step 4: Build minimal disruption_analysis from metadata ----
        # The risk calculation only needs disrupted_countries from the DA.
        disruption_analysis = {
            "involved": {
                "countries": meta.get("disrupted_countries", []),
                "companies": meta.get("disrupted_companies", []),
            }
        }
        
        # ---- Step 5: Compute risk scores (100% deterministic Python) ----
        from evaluation.ground_truth_generator import GroundTruthGenerator
        
        logger.info(
            f"[calculate_tier1_comprehensive_risks] Computing risk for {company_name} "
            f"({sum(chain_counts.values())} total chains)"
        )
        
        gt_generator = GroundTruthGenerator(monitored_company=company_name)
        risk_assessment = gt_generator._calculate_risk_assessment(
            kg_results=actual_kg_results,
            disruption_analysis=disruption_analysis,
            scenario={"company_name": company_name},
        )
        
        supplier_count = len(risk_assessment.get("supplier_risk_scores", {}))
        logger.info(
            f"[calculate_tier1_comprehensive_risks] Complete. "
            f"Risk scores for {supplier_count} Tier-1 suppliers."
        )
        
        # ---- Step 6: Save risk output to disk for reliable downstream retrieval ----
        # CrewAI converts the returned dict to a Python repr string which may
        # contain unterminated string literals that break downstream parsing.
        # Saving to disk as valid JSON guarantees the worker can load it.
        risk_output_path = TEMP_KG_DIR / f"{scenario_id}_risk.json"
        with risk_output_path.open("w", encoding="utf-8") as f:
            json.dump(risk_assessment, f, indent=2, default=str)
        logger.info(
            f"[calculate_tier1_comprehensive_risks] Saved risk output to {risk_output_path}"
        )
        
        return risk_assessment
        
    except Exception as exc:
        logger.error(
            f"[calculate_tier1_comprehensive_risks] Risk calculation failed: {exc}",
            exc_info=True
        )
        return {
            "error": str(exc),
            "supplier_risk_scores": {},
            "tier1_risk_profiles": [],
            "critical_suppliers": [],
            "summary": {
                "total_disrupted_chains": 0,
                "tier_breakdown": {"tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 0},
            },
        }


# Create the tool
tier1_comprehensive_risk_tool = StructuredTool(
    name="calculate_tier1_comprehensive_risks",
    description=(
        "Calculate comprehensive Tier-1 risk scores for all suppliers affected by a disruption. "
        "This tool loads all required data (KG results, disruption metadata) from disk using "
        "the scenario_id — you only need to pass the scenario_id string. "
        "The tool uses the SAME deterministic Python logic as ground truth generation: "
        "exposure depth, breadth, criticality, and centrality metrics are computed and "
        "aggregated to produce risk scores for ALL Tier-1 suppliers. "
        "Returns a complete risk assessment dict with supplier_risk_scores, "
        "tier1_risk_profiles, critical_suppliers, methodology, etc."
    ),
    func=calculate_tier1_comprehensive_risks,
    args_schema=Tier1RiskCalculationInput,
    return_direct=True,  # Return complete result directly without LLM processing
)
