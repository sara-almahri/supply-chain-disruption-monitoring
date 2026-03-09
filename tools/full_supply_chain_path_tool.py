"""
tools/full_supply_chain_path_tool.py
------------------------------------

Shared utility that retrieves the complete disrupted supply-chain paths
for a monitored company, mirroring the logic used by the ground-truth
generator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from tools.neo4j_setup import graph

logger = logging.getLogger(__name__)


def build_disrupted_supply_chains(
    monitored_company: str,
    disrupted_countries: List[str] | None = None,
    disrupted_companies: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Return full Tier-0 → Tier-4 chains for all suppliers located in disrupted
    regions (or explicitly disrupted companies).
    """
    # CRITICAL: Resolve monitored company to canonical KG name
    monitored_company = _resolve_company_name(monitored_company)
    
    disrupted_countries = _normalise_list(disrupted_countries)
    disrupted_companies = _normalise_list(disrupted_companies)

    if not disrupted_countries and not disrupted_companies:
        logger.info(
            "[SupplyChainPathTool] No disrupted countries/companies supplied; returning empty chains."
        )
        return _empty_result(monitored_company, disrupted_countries)

    records = _run_path_query(
        monitored_company=monitored_company,
        disrupted_countries=disrupted_countries,
        disrupted_companies=disrupted_companies,
    )

    tier_map = _format_records_by_tier(
        monitored_company,
        records,
        disrupted_countries,
    )

    summary = {
        "total_disrupted_chains": sum(len(tier_map[tier]) for tier in range(1, 5)),
        "tier_1_count": len(tier_map[1]),
        "tier_2_count": len(tier_map[2]),
        "tier_3_count": len(tier_map[3]),
        "tier_4_count": len(tier_map[4]),
    }

    logger.info(
        "[SupplyChainPathTool] Retrieved disrupted supply chains "
        "(Tier-1: %s, Tier-2: %s, Tier-3: %s, Tier-4: %s)",
        summary["tier_1_count"],
        summary["tier_2_count"],
        summary["tier_3_count"],
        summary["tier_4_count"],
    )

    return {
        "tier_1": tier_map[1],
        "tier_2": tier_map[2],
        "tier_3": tier_map[3],
        "tier_4": tier_map[4],
        "monitored_company": monitored_company,
        "disrupted_countries": disrupted_countries,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_company_name(name: str) -> str:
    """
    Resolve company name to its canonical form in the KG using semantic similarity.
    Uses OpenAI embeddings for the most accurate matching.
    Falls back to string matching if embeddings are not available.
    
    E.g., 'Tesla' → 'Tesla Inc', 'Tesla Motors' → 'Tesla Inc'
    """
    if not name:
        return name
    
    # Try embedding-based resolution first (most accurate)
    resolved = _resolve_with_embeddings(name)
    if resolved:
        return resolved
    
    # Fallback to string matching
    logger.info(f"[SupplyChainPathTool] Embeddings not available, using string matching for '{name}'")
    resolve_query = """
    MATCH (c:Company)
    WHERE toLower(c.name) CONTAINS toLower($name)
       OR toLower($name) CONTAINS toLower(c.name)
    RETURN c.name AS exact_name
    ORDER BY
      CASE WHEN toLower(c.name) = toLower($name) THEN 0 ELSE 1 END,
      size(c.name),
      c.name
    LIMIT 1
    """
    try:
        result = graph.query(resolve_query, {"name": name})
        if result and result[0].get("exact_name"):
            resolved = result[0]["exact_name"]
            if resolved != name:
                logger.info(f"[SupplyChainPathTool] Resolved '{name}' → '{resolved}' (string matching)")
            return resolved
    except Exception as exc:
        logger.error(f"[SupplyChainPathTool] Failed to resolve company name '{name}': {exc}")
    
    logger.warning(f"[SupplyChainPathTool] Using company name as-is: '{name}'")
    return name


def _resolve_with_embeddings(name: str) -> str | None:
    """
    Use semantic similarity search to resolve company name.
    Returns the most similar company name from the KG.
    """
    import os
    from openai import OpenAI
    
    try:
        # Check if companies have embeddings
        check_query = """
        MATCH (c:Company)
        WHERE c.embedding IS NOT NULL
        RETURN count(c) as count
        LIMIT 1
        """
        result = graph.query(check_query)
        if not result or result[0].get("count", 0) == 0:
            return None
        
        # Generate embedding for the query
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=[f"Company: {name}"],
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        # Find most similar company using vector similarity
        # Try vector index first (Neo4j 5.11+)
        similarity_query = """
        CALL db.index.vector.queryNodes('company_embedding_index', 1, $embedding)
        YIELD node, score
        RETURN node.name AS name, score
        """
        
        try:
            result = graph.query(similarity_query, {"embedding": query_embedding})
            if result and result[0].get("name"):
                resolved = result[0]["name"]
                similarity = result[0].get("score", 0)
                
                if resolved != name:
                    logger.info(
                        f"[SupplyChainPathTool] Resolved '{name}' → '{resolved}' "
                        f"(semantic similarity: {similarity:.3f})"
                    )
                return resolved
        except Exception as index_error:
            # Fallback to manual cosine similarity if index doesn't exist
            logger.debug(f"Vector index not available, using manual similarity: {index_error}")
            
            manual_query = """
            MATCH (c:Company)
            WHERE c.embedding IS NOT NULL
            WITH c, 
                 gds.similarity.cosine(c.embedding, $embedding) AS similarity
            RETURN c.name AS name, similarity
            ORDER BY similarity DESC
            LIMIT 1
            """
            
            result = graph.query(manual_query, {"embedding": query_embedding})
            if result and result[0].get("name"):
                resolved = result[0]["name"]
                similarity = result[0].get("similarity", 0)
                
                if resolved != name:
                    logger.info(
                        f"[SupplyChainPathTool] Resolved '{name}' → '{resolved}' "
                        f"(semantic similarity: {similarity:.3f})"
                    )
                return resolved
        
        return None
        
    except Exception as exc:
        logger.warning(f"[SupplyChainPathTool] Embedding-based resolution failed for '{name}': {exc}")
        return None


def _normalise_list(values: List[str] | None) -> List[str]:
    if not values:
        return []
    return sorted({str(v).strip() for v in values if str(v).strip()})


def _run_path_query(
    monitored_company: str,
    disrupted_countries: List[str],
    disrupted_companies: List[str],
) -> List[Dict[str, Any]]:
    """
    Execute Neo4j query returning full chains and metadata for disrupted suppliers.
    
    BATCHED BY TIER to avoid memory limits:
    - Query each tier (1-4) separately
    - This prevents Neo4j from trying to return all paths in one transaction
    - If one tier fails, others can still succeed
    """
    all_results = []
    
    # Query each tier separately to avoid memory limits
    for target_tier in range(1, 5):
        query = f"""
        MATCH (start:Company {{name: $monitored_company}})
        CALL apoc.path.expandConfig(start, {{
          relationshipFilter: "suppliesTo<",
          bfs: true,
          maxLevel: {target_tier},
          uniqueNodes: "GLOBAL"
        }}) YIELD path
        WITH nodes(path) AS chainNodes, length(path) AS depth
        WHERE depth = {target_tier}
        WITH chainNodes, depth, chainNodes[-1] AS finalNode
        OPTIONAL MATCH (finalNode)-[:locatedIn]->(finalCountry:Country)
        WITH chainNodes, depth,
             finalNode,
             collect(DISTINCT finalCountry.name) AS finalCountries
        WITH chainNodes, depth, finalNode,
             [c IN finalCountries WHERE c IS NOT NULL] AS cleanedCountries
        WHERE (
              size(cleanedCountries) > 0
          AND any(c IN cleanedCountries WHERE c IN $disrupted_countries)
        )
        OR finalNode.name IN $disrupted_companies
        WITH depth,
             [node IN chainNodes | {{
                company: node.name,
                industry: node.industry,
                countries: [(node)-[:locatedIn]->(co:Country) | co.name]
             }}] AS processedChain
        RETURN depth AS tier, processedChain
        ORDER BY processedChain[-1].company, processedChain[-1].countries
        """

        params = {
            "monitored_company": monitored_company,
            "disrupted_countries": disrupted_countries,
            "disrupted_companies": disrupted_companies,
        }

        try:
            tier_results = graph.query(query, params) or []
            all_results.extend(tier_results)
            logger.info(
                f"[SupplyChainPathTool] Tier {target_tier}: Retrieved {len(tier_results)} chains"
            )
        except Exception as exc:
            logger.warning(
                f"[SupplyChainPathTool] Tier {target_tier} query failed (memory limit?): {exc}"
            )
            # Continue with other tiers even if one fails
            continue
    
    return all_results


def _format_records_by_tier(
    monitored_company: str,
    records: List[Dict[str, Any]],
    disrupted_countries: List[str],
) -> Dict[int, List[List[Dict[str, str]]]]:
    """
    Convert Neo4j records to tier buckets with deduplicated, sorted chains.
    """
    tier_map: Dict[int, List[List[Dict[str, str]]]] = {1: [], 2: [], 3: [], 4: []}
    # CRITICAL: Deduplication key includes (company, country, industry) for each node
    seen_per_tier: Dict[int, Set[Tuple[Tuple[str, str, str], ...]]] = {1: set(), 2: set(), 3: set(), 4: set()}
    disrupted_set = set(disrupted_countries or [])

    for record in records:
        tier = int(record.get("tier", 0))
        processed_chain = record.get("processedChain") or []

        if tier not in tier_map or len(processed_chain) < 2:
            continue

        formatted_chain = _format_chain(
            monitored_company,
            processed_chain,
            disrupted_set,
        )
        if not formatted_chain:
            continue

        # CRITICAL: Deduplication key must include company, country, AND industry
        # to properly distinguish chains that differ only in industry
        key = tuple(
            (node["company"], node["country"], node.get("industry", "Unknown"))
            for node in formatted_chain
        )
        if key in seen_per_tier[tier]:
            continue

        tier_map[tier].append(formatted_chain)
        seen_per_tier[tier].add(key)

    for tier, chains in tier_map.items():
        def chain_key(chain: List[Dict[str, str]]):
            final = chain[-1]
            signature = tuple(
                (node["company"], node["country"], node["industry"])
                for node in chain
            )
            return (final["company"], final["country"], len(chain), signature)

        chains.sort(key=chain_key)

    return tier_map


def _format_chain(
    monitored_company: str,
    raw_chain: List[Dict[str, Any]],
    disrupted_set: Set[str],
) -> List[Dict[str, str]]:
    """
    Normalise country selection and ensure the chain starts with the monitored company.
    """
    formatted: List[Dict[str, str]] = []

    for node in raw_chain:
        company = node.get("company")
        if not company:
            return []

        countries = node.get("countries") or []
        industry = node.get("industry") or "Unknown"
        country = _select_country(countries, disrupted_set)

        formatted.append(
            {
                "company": company,
                "country": country,
                "industry": industry if industry else "Unknown",
            }
        )

    if not formatted:
        return []

    if formatted[0]["company"] != monitored_company:
        formatted.insert(0, _fetch_company_profile(monitored_company))

    return formatted


def _select_country(countries: List[str], disrupted_set: Set[str]) -> str:
    if not countries:
        return "Unknown"
    for country in countries:
        if country in disrupted_set:
            return country
    return sorted(countries)[0]


def _fetch_company_profile(company_name: str) -> Dict[str, str]:
    query = """
    MATCH (c:Company {name: $name})
    OPTIONAL MATCH (c)-[:locatedIn]->(co:Country)
    RETURN c.industry AS industry, collect(DISTINCT co.name) AS countries
    """
    try:
        rows = graph.query(query, {"name": company_name}) or []
        if rows:
            row = rows[0]
            countries = row.get("countries") or []
            country = _select_country(countries, set())
            industry = row.get("industry") or "Unknown"
            return {"company": company_name, "country": country, "industry": industry}
    except Exception as exc:
        logger.error("[SupplyChainPathTool] Failed to fetch company profile for '%s': %s", company_name, exc)

    return {"company": company_name, "country": "Unknown", "industry": "Unknown"}


def _empty_result(monitored_company: str, disrupted_countries: List[str]) -> Dict[str, Any]:
    return {
        "tier_1": [],
        "tier_2": [],
        "tier_3": [],
        "tier_4": [],
        "monitored_company": monitored_company,
        "disrupted_countries": disrupted_countries or [],
        "summary": {
            "total_disrupted_chains": 0,
            "tier_1_count": 0,
            "tier_2_count": 0,
            "tier_3_count": 0,
            "tier_4_count": 0,
        },
    }

# ------------------------------------------------------------------------------
# Structured tool exposure
# ------------------------------------------------------------------------------


class SupplyChainPathArgs(BaseModel):
    """Arguments for the build_disrupted_supply_chains tool."""

    monitored_company: str = Field(
        ...,
        description="Exact company name in the knowledge graph that should be treated as Tier-0.",
    )
    disrupted_countries: List[str] | None = Field(
        default=None,
        description="List of disrupted countries. Provide [] if there are none.",
    )
    disrupted_companies: List[str] | None = Field(
        default=None,
        description=(
            "Explicit disrupted supplier names to ensure inclusion even if they are "
            "outside the disrupted countries list."
        ),
    )


def _build_disrupted_supply_chains_tool_func(
    monitored_company: str,
    disrupted_countries: List[str] | None = None,
    disrupted_companies: List[str] | None = None,
) -> Dict[str, Any]:
    return build_disrupted_supply_chains(
        monitored_company=monitored_company,
        disrupted_countries=disrupted_countries,
        disrupted_companies=disrupted_companies,
    )


build_disrupted_supply_chains_tool = StructuredTool(
    name="build_disrupted_supply_chains",
    description=(
        "Retrieve the complete Tier-0 → Tier-4 disrupted supply chain paths for the "
        "monitored company, ordered deterministically and capped at Tier-4. "
        "Returns a JSON dict with keys: tier_1, tier_2, tier_3, tier_4, monitored_company, "
        "disrupted_countries, summary. Use whenever you need the exhaustive list of disrupted chains."
    ),
    func=_build_disrupted_supply_chains_tool_func,
    args_schema=SupplyChainPathArgs,
    return_direct=False,  # Let agent handle output properly
)

