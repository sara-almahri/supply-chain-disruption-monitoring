"""
KG Orchestration Tools
======================
Two-step process to handle large KG outputs without LLM token limits:
1. build_and_save_kg: Execute query, save to disk, return metadata (small)
2. get_saved_kg_for_output: Load from disk for final output (bypasses LLM streaming)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.full_supply_chain_path_tool import build_disrupted_supply_chains

logger = logging.getLogger(__name__)

TEMP_KG_DIR = Path(__file__).parent.parent / "output" / "temp" / "kg_payloads"


class BuildAndSaveKGInput(BaseModel):
    """Input for build_and_save_kg tool"""
    monitored_company: str = Field(..., description="Exact knowledge-graph name of the monitored company")
    ongoing_disruption: bool = Field(..., description="The Disruption Monitoring Agent's ongoing_disruption value")
    disrupted_countries: List[str] = Field(default_factory=list, description="involved.countries from the disruption analysis")
    disrupted_industries: List[str] = Field(default_factory=list, description="involved.industries from the disruption analysis")
    disrupted_companies: List[str] = Field(default_factory=list, description="involved.companies from the disruption analysis")
    scope: str = Field(default="country", description='scope from the disruption analysis: "country" or "industries"')
    scenario_id: str = Field(default="default", description="Scenario identifier")


class GetSavedKGInput(BaseModel):
    """Input for get_saved_kg_for_output tool"""
    kg_payload_path: str = Field(..., description="Path to saved KG payload")


def build_and_save_kg(
    monitored_company: str,
    ongoing_disruption: bool = True,
    disrupted_countries: List[str] = None,
    disrupted_industries: List[str] = None,
    disrupted_companies: List[str] = None,
    scope: str = "country",
    scenario_id: str = "default",
) -> Dict[str, Any]:
    """
    Step 1: Build the disrupted supply chains and save them to disk.
    Returns ONLY metadata (small) - not the full chains. No query is run when there is no ongoing disruption.
    """
    TEMP_KG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[build_and_save_kg] {monitored_company}: ongoing={ongoing_disruption}, scope={scope}, "
        f"countries={disrupted_countries}, industries={disrupted_industries}, companies={disrupted_companies}"
    )

    if ongoing_disruption:
        kg_results = build_disrupted_supply_chains(
            monitored_company=monitored_company,
            disrupted_countries=disrupted_countries or [],
            disrupted_companies=disrupted_companies or [],
            disrupted_industries=disrupted_industries or [],
            scope=scope,
        )
    else:
        kg_results = {f"tier_{t}": [] for t in range(1, 5)}
        kg_results.update({
            "monitored_company": monitored_company,
            "disrupted_countries": [],
            "resolution": None,
            "summary": {"total_disrupted_chains": 0, **{f"tier_{t}_count": 0 for t in range(1, 5)}},
        })
    resolution = kg_results.get("resolution") or {}

    # Save to disk
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    safe_company = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in monitored_company)
    filename = f"{scenario_id}_{safe_company}_{timestamp}_kg.json"
    file_path = TEMP_KG_DIR / filename
    
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(kg_results, f, indent=2, default=str)
    
    # Extract metadata (small, LLM-friendly)
    metadata = {
        "success": True,
        "kg_payload_path": str(file_path),
        "monitored_company": kg_results.get("monitored_company"),
        "ongoing_disruption": bool(ongoing_disruption),
        "scope": resolution.get("scope", scope),
        "disrupted_countries": resolution.get("countries", []),
        "disrupted_industries": resolution.get("industries", []),
        "disrupted_companies": resolution.get("companies", []),
        "unresolved": resolution.get("unresolved", {}),
        "requested": {"countries": disrupted_countries or [], "industries": disrupted_industries or [],
                      "companies": disrupted_companies or [], "scope": scope},
        "scenario_id": scenario_id,
        "chain_counts": {
            "tier_1": len(kg_results.get("tier_1", [])),
            "tier_2": len(kg_results.get("tier_2", [])),
            "tier_3": len(kg_results.get("tier_3", [])),
            "tier_4": len(kg_results.get("tier_4", [])),
        },
        "total_chains": kg_results.get("summary", {}).get("total_disrupted_chains", 0),
        "message": f"Built and saved {kg_results.get('summary', {}).get('total_disrupted_chains', 0)} disrupted supply chains to {file_path}",
    }
    
    # --- Save companion meta file at PREDICTABLE path for downstream tools ---
    # The Risk Manager tool can look up this file by scenario_id alone,
    # avoiding unreliable LLM serialization of complex data.
    meta_path = TEMP_KG_DIR / f"{scenario_id}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(
        f"[build_and_save_kg] Saved {metadata['total_chains']} chains to {file_path}, "
        f"meta to {meta_path}"
    )
    
    return metadata


def get_saved_kg_for_output(kg_payload_path: str) -> Dict[str, Any]:
    """
    Step 2: Load the saved KG payload from disk for final output.
    This returns the FULL data structure without streaming through LLM.
    """
    try:
        file_path = Path(kg_payload_path)
        if not file_path.exists():
            logger.error(f"[get_saved_kg_for_output] File not found: {kg_payload_path}")
            return {
                "error": f"KG payload file not found: {kg_payload_path}",
                "tier_1": [],
                "tier_2": [],
                "tier_3": [],
                "tier_4": [],
            }
        
        kg_data = json.loads(file_path.read_text(encoding="utf-8"))
        
        logger.info(
            f"[get_saved_kg_for_output] Loaded KG data from {kg_payload_path}: "
            f"{len(kg_data.get('tier_1', []))} T1, {len(kg_data.get('tier_2', []))} T2, "
            f"{len(kg_data.get('tier_3', []))} T3, {len(kg_data.get('tier_4', []))} T4"
        )
        
        return kg_data
        
    except Exception as exc:
        logger.error(f"[get_saved_kg_for_output] Failed to load KG data: {exc}")
        return {
            "error": str(exc),
            "tier_1": [],
            "tier_2": [],
            "tier_3": [],
            "tier_4": [],
        }


# Create the tools
build_and_save_kg_tool = StructuredTool(
    name="build_and_save_kg",
    description=(
        "Build the disrupted supply chains (Tier-1 to Tier-4) of the monitored company from the knowledge graph "
        "and save them to disk. Pass the disruption analysis fields unchanged. "
        "Returns ONLY metadata (path, resolved names, chain counts) - not the full chains."
    ),
    func=build_and_save_kg,
    args_schema=BuildAndSaveKGInput,
    return_direct=True,
)

get_saved_kg_tool = StructuredTool(
    name="get_saved_kg_for_output",
    description=(
        "Step 2: Load the complete KG data from disk (saved by build_and_save_kg). "
        "This returns the FULL supply chain structure with all tiers. "
        "Use the kg_payload_path from build_and_save_kg output. "
        "This bypasses LLM token limits by loading directly from disk."
    ),
    func=get_saved_kg_for_output,
    args_schema=GetSavedKGInput,
    return_direct=True,  # Return directly without LLM processing
)

