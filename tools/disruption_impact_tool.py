import logging
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from tools.neo4j_setup import graph
from tools.graph_metrics_tool import graph_metrics_tool

logger = logging.getLogger(__name__)

class ImpactAnalysisInput(BaseModel):
    tiers: Dict[int, List[str]] = Field(...,
        description="Supplier tiers {1: [companies], 2: [...]}")
    disrupted_companies: List[str] = Field(...,
        description="List of disrupted company names")
    risk_threshold: float = Field(default=0.7,
        description="Risk score cutoff for critical suppliers")

class DisruptionImpactTool:
    """Deterministic risk calculator with tier processing"""
    
    def calculate_risk_profile(self, tiers: Dict[int, List[str]], disrupted: List[str]) -> Dict:
        """Batch process risk scores using parameterized Cypher"""
        try:
            # Get all degrees first
            degree_data = graph_metrics_tool._run(metric_type="degree")["degrees"]
            
            # Batch calculate dependency ratios
            dependency_query = """
                UNWIND $companies AS company
                MATCH (c:Company {name: company})-[:suppliesTo*1..3]->(sub)
                WITH company, 
                     count(DISTINCT sub) AS total,
                     count(DISTINCT CASE WHEN sub.name IN $disrupted THEN sub END) AS risky
                RETURN company, 
                       CASE WHEN total > 0 THEN risky*1.0/total ELSE 0.0 END AS ratio
            """
            companies = [c for tier in tiers.values() for c in tier]
            dep_results = graph.query(dependency_query, {
                "companies": companies,
                "disrupted": disrupted
            })
            dep_ratios = {r["company"]: r["ratio"] for r in dep_results}
            
            # Calculate risk scores
            risk_assessment = {}
            for tier, companies in tiers.items():
                for company in companies:
                    dep_ratio = dep_ratios.get(company, 0)
                    in_degree = degree_data.get(company, 0)
                    tier_weight = 1 / max(tier, 1)
                    
                    risk_score = min(
                        (dep_ratio * 0.6) + 
                        (in_degree * 0.3) + 
                        (tier_weight * 0.1),
                        1.0
                    )
                    risk_assessment[company] = {
                        "risk_score": round(risk_score, 3),
                        "tier": tier,
                        "dependency": round(dep_ratio, 3),
                        "degree": in_degree
                    }
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk calculation failed: {str(e)}")
            return {}

    def _run(self, **kwargs) -> Dict:
        try:
            params = ImpactAnalysisInput(**kwargs).dict()
            risk_data = self.calculate_risk_profile(
                params["tiers"],
                params["disrupted_companies"]
            )
            
            critical = [
                company for company, data in risk_data.items()
                if data["risk_score"] >= params["risk_threshold"]
            ]
            
            return {
                "risk_assessment": risk_data,
                "critical_suppliers": sorted(critical, 
                    key=lambda x: risk_data[x]["risk_score"], reverse=True)
            }
            
        except Exception as e:
            logger.error(f"Impact analysis failed: {str(e)}")
            return {"error": str(e)}

disruption_impact_tool = StructuredTool(
    name="supply_chain_risk_assessor",
    description="Computes supply chain risk scores from tiered data",
    func=DisruptionImpactTool()._run,
    args_schema=ImpactAnalysisInput
)