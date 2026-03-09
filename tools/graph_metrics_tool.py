# graph_metrics_tool.py

import logging
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError, validate_arguments  # FIXED IMPORT
from langchain_core.tools import StructuredTool
from .neo4j_setup import graph

logger = logging.getLogger(__name__)
MAX_COMPANIES = 500  # Strict safety limit

class GraphMetricsInput(BaseModel):
    metric_type: str = Field(..., pattern="^(degree|dependency_ratio)$")
    target_company: Optional[str] = Field(None, max_length=100)
    disrupted_companies: Optional[List[str]] = Field(None, max_items=500)

@validate_arguments
def get_degree_centrality() -> Dict[str, int]:
    """Batch-optimized degree calculation"""
    try:
        result = graph.query("""
            MATCH (c:Company)
            OPTIONAL MATCH (c)<-[:suppliesTo]-(s)
            WITH c, count(s) AS degree
            RETURN c.name AS company, degree
            ORDER BY degree DESC
            LIMIT $limit
        """, {"limit": MAX_COMPANIES})
        
        return {r["company"]: r["degree"] for r in result}
    
    except Exception as e:
        logger.error(f"Degree calculation failed: {str(e)}")
        return {}

@validate_arguments
def calculate_dependency_ratio(
    target_company: str,
    disrupted_companies: List[str]
) -> float:
    """Enterprise-grade dependency calculation"""
    try:
        if len(disrupted_companies) > MAX_COMPANIES:
            raise ValueError(f"Max {MAX_COMPANIES} companies allowed")
        
        query = """
        MATCH (c:Company {name: $target})
        WITH c, $disrupted AS disrupted
        UNWIND disrupted AS company
        MATCH (c)-[:suppliesTo*1..3]->(sub:Company {name: company})
        RETURN count(DISTINCT sub) AS affected_count, 
               size($disrupted) AS total_companies
        """
        
        result = graph.query(query, {
            "target": target_company,
            "disrupted": disrupted_companies[:MAX_COMPANIES]
        })
        
        if result and result[0]["total_companies"] > 0:
            return round(result[0]["affected_count"] / result[0]["total_companies"], 3)
        return 0.0
    
    except Exception as e:
        logger.error(f"Dependency ratio failed: {str(e)}")
        return 0.0

def graph_metrics_tool_entrypoint(**kwargs) -> Dict:
    """Full implementation with validation"""
    try:
        params = GraphMetricsInput(**kwargs).dict()
        
        if params["metric_type"] == "degree":
            return {"degrees": get_degree_centrality()}
            
        if params["metric_type"] == "dependency_ratio":
            if not params.get("target_company") or not params.get("disrupted_companies"):
                raise ValueError("Missing required parameters")
                
            return {
                "dependency_ratio": calculate_dependency_ratio(
                    params["target_company"],
                    params["disrupted_companies"]
                )
            }
            
        raise ValueError("Invalid metric type")
        
    except ValidationError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return {"error": "Invalid input parameters"}
    except Exception as e:
        logger.error(f"Metrics failed: {str(e)}")
        return {"error": "Metric calculation failed"}

graph_metrics_tool = StructuredTool(
    name="supply_chain_metrics",
    description="Computes node metrics with enterprise safety controls",
    func=graph_metrics_tool_entrypoint,
    args_schema=GraphMetricsInput
)