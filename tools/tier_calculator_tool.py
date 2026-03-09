# tools/tier_calculator_tool.py
import logging
from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from tools.neo4j_setup import graph

logger = logging.getLogger(__name__)

class TierInput(BaseModel):
    start_company: str = Field(..., description="Anchor company name")
    target_company: str = Field(..., description="Target company name")

def calculate_tier(start_company: str, target_company: str) -> Dict[str, int]:
    cypher = """
    MATCH (start:Company {name:$startName}), (target:Company {name:$targetName}),
          path = shortestPath((start)-[:suppliesTo*..10]->(target))
    RETURN length(path) AS distance
    """
    result = graph.query(cypher, {"startName": start_company, "targetName": target_company})
    if not result or len(result) == 0 or result[0]["distance"] is None:
        return {"tier": -1}
    return {"tier": result[0]["distance"]}

calculate_tier_struct = StructuredTool(
    name="calculate-tier",
    description="Calculate BFS distance (tier) from start_company to target_company via :suppliesTo edges.",
    func=calculate_tier,
    args_schema=TierInput
)
