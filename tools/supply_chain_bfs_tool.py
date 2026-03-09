# supply_chain_bfs_tool.py

import logging
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from .neo4j_setup import graph  # Ensure this import points to your actual Neo4j connection

logger = logging.getLogger(__name__)

##############################################################################
# Tool: Supply Chain BFS with Tier Chunking and Full Path Retrieval
##############################################################################

class SupplyChainBFSInput(BaseModel):
    start_company: str = Field(..., description="Root company canonical name")
    max_tier: Optional[int] = Field(
        default=4,
        description="Max BFS depth (default: 4 for Tier-4)."
    )
    location: Optional[str] = Field(
        default=None,
        description="Optional filter by country (applied to final node)."
    )
    industry: Optional[str] = Field(
        default=None,
        description="Optional filter by industry (applied to final node)."
    )
    limit: Optional[int] = Field(
        default=100,
        description="Max number of chains to retrieve per tier, before chunking."
    )
    direction: Optional[str] = Field(
        default="DOWNSTREAM",
        description="BFS direction: 'DOWNSTREAM', 'UPSTREAM', or 'BOTH'."
    )

def supply_chain_bfs(
    start_company: str,
    max_tier: int = 4,
    location: Optional[str] = None,
    industry: Optional[str] = None,
    limit: Optional[int] = 100,
    direction: str = "DOWNSTREAM",
    chunk_size: int = 50  # define chunk size for final results
) -> Dict[int, List[List[List[Dict]]]]:
    """
    Perform a BFS from 'start_company' up to 'max_tier' in the specified 'direction'. 
    Return a dictionary keyed by tier, each value is a list of chunked sub-lists 
    (to avoid massive single outputs).
    
    Each node dict:
      - 'company': Supplier’s canonical name
      - 'industry': Standardized industry classification
      - 'country': The 'locatedIn' country or 'Unknown'
      - 'supplies_to': The immediate upstream supplier (None for the 1st node)

    If 'location' or 'industry' is provided, we ONLY retain chains where the 
    final node matches that location/industry.

    BFS results are grouped by tier = (depth - 1), min=1.
    We then limit the total number of chains per tier to 'limit', 
    and finally chunk each tier's results into sub-lists of up to 'chunk_size'.
    """
    logger.info(
        f"Running supply_chain_bfs for start_company={start_company}, "
        f"max_tier={max_tier}, location={location}, industry={industry}, "
        f"limit={limit}, direction={direction}, chunk_size={chunk_size}"
    )

    direction = direction.upper()
    if direction == "DOWNSTREAM":
        rel_filter = "suppliesTo>"
    elif direction == "UPSTREAM":
        rel_filter = "<suppliesTo"
    else:
        rel_filter = "suppliesTo"

    cypher = f"""
    MATCH (start:Company {{name: $start_company}})
    CALL apoc.path.expandConfig(start, {{
      relationshipFilter: "{rel_filter}",
      bfs: true,
      uniqueNodes: "GLOBAL",
      maxLevel: $max_tier
    }}) YIELD path
    WITH nodes(path) AS chain, length(path) AS depth
    WITH [n IN chain | {{
        name: n.name,
        industry: n.industry,
        country: head([c IN [(n)-[:locatedIn]->(co:Country) | co.name] WHERE c IS NOT NULL])
    }}] AS processedChain, depth
    """

    if location or industry:
        cypher += """
    WITH processedChain, depth, last(processedChain) AS finalNode
    WHERE ($location IS NULL OR toLower(finalNode.country) = toLower($location))
      AND ($industry IS NULL OR toLower(finalNode.industry) = toLower($industry))
    RETURN processedChain AS chain, depth
    ORDER BY depth, last(processedChain).name
    """
    else:
        cypher += """
    RETURN processedChain AS chain, depth
    ORDER BY depth, last(processedChain).name
    """

    params = {
        "start_company": start_company,
        "max_tier": max_tier,
        "location": location,
        "industry": industry
    }

    logger.info(f"BFS Cypher Query:\n{cypher}\nParams: {params}")
    raw_results = graph.query(cypher, params)
    logger.info(f"Raw BFS query results count: {len(raw_results)}")

    tier_groups: Dict[int, List[List[Dict]]] = {}
    for r in raw_results:
        chain_nodes = r["chain"]
        depth = r["depth"]
        tier = depth  # Corrected: depth=1 is Tier-1, depth=2 is Tier-2, etc.
        processed_chain = []
        for node in chain_nodes:
            processed_chain.append({
                "company": node.get("name", "Unknown"),
                "industry": node.get("industry", "Unknown"),
                "country": node["country"] if node.get("country") else "Unknown",
                "supplies_to": None
            })
        # Link 'supplies_to'
        for i in range(1, len(processed_chain)):
            processed_chain[i]["supplies_to"] = processed_chain[i-1]["company"]
        tier_groups.setdefault(tier, []).append(processed_chain)

    # Apply 'limit' per tier
    for t in tier_groups:
        tier_groups[t] = tier_groups[t][:limit]

    # Chunk the results
    chunked_results: Dict[int, List[List[List[Dict]]]] = {}
    for t, chains in tier_groups.items():
        tier_chunks = []
        for i in range(0, len(chains), chunk_size):
            tier_chunks.append(chains[i : i + chunk_size])
        chunked_results[t] = tier_chunks

    return chunked_results


def recompile_full_answer(
    chunked_results: Dict[int, List[List[List[Dict]]]]
) -> Dict[int, List[List[Dict]]]:
    """
    Convert the BFS output from chunked form -> a single dict[tier] = list of chains.
    Each chain is a list of nodes.
    """
    full_answer: Dict[int, List[List[Dict]]] = {}
    for tier, chunks in chunked_results.items():
        all_chains_for_tier = []
        for chunk in chunks:
            all_chains_for_tier.extend(chunk)
        full_answer[tier] = all_chains_for_tier
    return full_answer


supply_chain_bfs_struct = StructuredTool(
    name="supply-chain-bfs",
    description=(
        "Perform BFS from a start_company (UPSTREAM/DOWNSTREAM/BOTH, up to max_tier), "
        "returning tiered chains. Each chain is an ordered list of node dicts with "
        "fields: company, industry, country, supplies_to. We chunk large results per tier "
        "to prevent massive single outputs. Use 'recompile_full_answer' to merge chunked "
        "results if needed."
    ),
    func=supply_chain_bfs,
    args_schema=SupplyChainBFSInput
)
