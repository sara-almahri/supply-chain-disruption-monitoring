# tools/company_lister_tool.py
import logging
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from tools.neo4j_setup import graph, neo4j_vector

logger = logging.getLogger(__name__)

class CompanyListInput(BaseModel):
    country: Optional[str] = Field(None, description="Filter by country name")
    industry: Optional[str] = Field(None, description="Filter by industry name")
    query_text: Optional[str] = Field(None, description="Semantic search if no filters used")
    limit: Optional[int] = Field(20000, description="Max number of results")

def company_list(country: Optional[str] = None,
                 industry: Optional[str] = None,
                 query_text: Optional[str] = None,
                 limit: Optional[int] = 20000) -> List[Dict]:
    logger.info(f"Listing companies with filters: country={country}, industry={industry}, query_text={query_text}")

    if query_text and not country and not industry:
        try:
            results = neo4j_vector.similarity_search(query_text, k=limit)
            return [{"name": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    where_clauses = []
    params = {"limit": limit}
    if country:
        where_clauses.append("co.name = $country")
        params["country"] = country
    if industry:
        where_clauses.append("c.industry = $industry")
        params["industry"] = industry

    cypher = "MATCH (c:Company)-[:locatedIn]->(co:Country)"
    if where_clauses:
        cypher += " WHERE " + " AND ".join(where_clauses)
    cypher += " RETURN c.name AS name, c.industry AS industry, co.name AS country LIMIT $limit"

    logger.debug(f"Running Cypher query: {cypher}")
    logger.debug(f"With parameters: {params}")

    try:
        results = graph.query(cypher, params)
        return results
    except Exception as e:
        logger.error(f"Cypher query failed: {e}")
        return []

company_list_struct = StructuredTool(
    name="company-list",
    description="Lists companies by optional country/industry filters or by semantic search.",
    func=company_list,
    args_schema=CompanyListInput
)
