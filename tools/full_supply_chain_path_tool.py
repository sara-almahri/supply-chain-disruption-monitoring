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

from offline.names import country_label, default_resolver
from tools.neo4j_setup import graph

logger = logging.getLogger(__name__)


def build_disrupted_supply_chains(
    monitored_company: str,
    disrupted_countries: List[str] | None = None,
    disrupted_companies: List[str] | None = None,
    disrupted_industries: List[str] | None = None,
    scope: str = "country",
) -> Dict[str, Any]:
    """
    Return the Tier-0 → Tier-4 chains that end at a disrupted company: a named disrupted company, or a company
    located in a disrupted country and, when scope is "industries", operating in a disrupted industry.
    Names are resolved to graph names by offline.names; names that cannot be resolved are reported, not guessed.
    """
    monitored_company = _resolve_company_name(monitored_company)
    resolution = default_resolver().disruption(
        disrupted_countries or [], disrupted_industries or [], disrupted_companies or [], scope=scope)
    match = resolution.pop("match")
    resolution["unresolved_count"] = sum(len(v) for v in resolution["unresolved"].values())
    if resolution["unresolved_count"]:
        logger.warning("[SupplyChainPathTool] Unresolved names: %s", resolution["unresolved"])

    if not match["countries"] and not match["companies"]:
        logger.info("[SupplyChainPathTool] No resolved disrupted country or company; returning empty chains.")
        result = _empty_result(monitored_company, resolution["countries"])
        result["resolution"] = resolution
        return result

    records = _run_path_query(
        monitored_company=monitored_company,
        disrupted_countries=match["countries"],
        disrupted_companies=match["companies"],
        disrupted_industries=match["industries"],
    )

    tier_map = _format_records_by_tier(
        monitored_company,
        records,
        match["countries"],
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
        "disrupted_countries": resolution["countries"],
        "resolution": resolution,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_company_name(name: str) -> str:
    """The graph name of the monitored company: the name itself when it is a graph name, otherwise the single
    company that offline.names resolves it to. Anything else is an error rather than a guess."""
    resolver = default_resolver()
    if name in resolver.company_names:
        return name
    names = resolver.company(name)
    if names and len(names) == 1:
        resolved = next(iter(names))
        logger.info("[SupplyChainPathTool] Resolved monitored company '%s' to '%s'", name, resolved)
        return resolved
    raise ValueError(f"Monitored company '{name}' does not resolve to exactly one company in the knowledge graph")


def _run_path_query(
    monitored_company: str,
    disrupted_countries: List[str],
    disrupted_companies: List[str],
    disrupted_industries: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Execute Neo4j query returning full chains and metadata for disrupted suppliers.
    Arguments are graph spellings; disrupted_industries is None unless the disruption is confined to named industries.
    
    BATCHED BY TIER to avoid memory limits:
    - Query each tier (1-4) separately
    - This prevents Neo4j from trying to return all paths in one transaction
    - uniqueness NODE_PATH: a path contains no company twice.
      The previous key uniqueNodes is not an APOC option and was ignored.
    - A failed tier query raises; it is not skipped.
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
          uniqueness: "NODE_PATH"
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
          AND ($disrupted_industries IS NULL OR finalNode.industry IN $disrupted_industries)
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
            "disrupted_industries": disrupted_industries,
        }

        try:
            tier_results = graph.query(query, params) or []
            all_results.extend(tier_results)
            logger.info(
                f"[SupplyChainPathTool] Tier {target_tier}: Retrieved {len(tier_results)} chains"
            )
        except Exception as exc:
            raise RuntimeError(
                f"[SupplyChainPathTool] Tier {target_tier} query failed for "
                f"{monitored_company}: {exc}"
            ) from exc
    
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
    return country_label(countries, disrupted_set)


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
        description="Companies named as disrupted. Provide [] if there are none.",
    )
    disrupted_industries: List[str] | None = Field(
        default=None,
        description="Industries named as disrupted. Provide [] if there are none.",
    )
    scope: str = Field(
        default="country",
        description='"industries" when the disruption is confined to the named industries, otherwise "country".',
    )


def _build_disrupted_supply_chains_tool_func(
    monitored_company: str,
    disrupted_countries: List[str] | None = None,
    disrupted_companies: List[str] | None = None,
    disrupted_industries: List[str] | None = None,
    scope: str = "country",
) -> Dict[str, Any]:
    return build_disrupted_supply_chains(
        monitored_company=monitored_company,
        disrupted_countries=disrupted_countries,
        disrupted_companies=disrupted_companies,
        disrupted_industries=disrupted_industries,
        scope=scope,
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

