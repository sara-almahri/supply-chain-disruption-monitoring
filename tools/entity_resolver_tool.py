# tools/entity_resolver_tool.py

import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

# Import our shared graph and vector from the setup
from .neo4j_setup import graph, neo4j_vector

logger = logging.getLogger(__name__)

##############################################################################
# Tool #1: Entity Resolution
##############################################################################

class EntityResolutionInput(BaseModel):
    entity_name: str = Field(..., description="Name to resolve (company or country)")

def fuzzy_match_name(
    name: str,
    label: str = "Company",
    limit: int = 3
) -> List[str]:
    """Naive fuzzy match using CONTAINS in Neo4j. Returns deterministic sorted results."""
    cypher = f"""
    MATCH (n:{label})
    WHERE toLower(n.name) CONTAINS toLower($name)
    RETURN n.name AS found_name
    ORDER BY 
      CASE WHEN toLower(n.name) = toLower($name) THEN 0 ELSE 1 END,
      size(n.name),
      n.name
    LIMIT $limit
    """
    results = graph.query(cypher, {"name": name, "limit": limit})
    return [r["found_name"] for r in results]

def resolve_entity(entity_name: str) -> Dict[str, Any]:
    """
    Resolve an entity name to canonical matches (companies or countries).
    """
    name = entity_name.strip()
    logger.info(f"Resolving entity: {name}")

    # 1) Vector search among Company nodes
    try:
        company_matches = neo4j_vector.similarity_search(name, k=3)
        candidates = [doc.page_content for doc in company_matches]
        logger.debug(f"Vector search results: {candidates}")
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        candidates = []

    # 2) Additional naive fuzzy matching for Company
    try:
        fuzzy_candidates = fuzzy_match_name(name, label="Company", limit=3)
        logger.debug(f"Fuzzy match results: {fuzzy_candidates}")
    except Exception as e:
        logger.error(f"Fuzzy match failed: {e}")
        fuzzy_candidates = []

    # Combine and deduplicate while preserving deterministic order
    combined = sorted(list(set(candidates + fuzzy_candidates)))

    # 3) Check if user might have meant a Country
    try:
        country_candidates = fuzzy_match_name(name, label="Country", limit=3)
        logger.debug(f"Country candidates: {country_candidates}")
    except Exception as e:
        logger.error(f"Country match failed: {e}")
        country_candidates = []

    return {
        "company_candidates": combined,
        "country_candidates": country_candidates
    }

# Wrap in a StructuredTool
resolve_entity_struct = StructuredTool(
    name="resolve-entity",
    description="Resolves an entity name (company or country) to canonical match(es) in Neo4j.",
    func=resolve_entity,
    args_schema=EntityResolutionInput
)
