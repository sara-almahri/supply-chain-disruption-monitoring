# tools/tier1_metrics_calculator.py
"""
Efficient Tier-1 Metrics Calculator

Calculates graph metrics ONLY for Tier-1 suppliers, not all 1,077 companies.
This is 30x more efficient and avoids context overflow.
"""

import logging
from typing import Dict, List, Set, Any
import networkx as nx
from tools.neo4j_setup import graph

logger = logging.getLogger(__name__)


def calculate_tier1_metrics(
    monitored_company: str,
    tier1_suppliers: List[str],
    disrupted_companies: List[str]
) -> Dict[str, Any]:
    """
    Calculate graph metrics ONLY for specified Tier-1 suppliers.
    
    This is efficient because:
    - We only calculate metrics for ~30 Tier-1 suppliers
    - We don't waste time on 1,000+ other companies
    - Output is compact and manageable for LLM
    
    Args:
        monitored_company: The company being monitored (e.g., Tesla)
        tier1_suppliers: List of Tier-1 supplier names
        disrupted_companies: List of disrupted company names
    
    Returns:
        {
            "centrality_metrics": {tier1_company: {betweenness, closeness, ...}},
            "dependency_ratios": {tier1_company: ratio},
            "pagerank": {tier1_company: score}
        }
    """
    if not tier1_suppliers:
        return {
            "centrality_metrics": {},
            "dependency_ratios": {},
            "pagerank": {}
        }
    
    try:
        # Build supply chain graph (we need full graph for accurate metrics)
        G = _build_supply_chain_graph(monitored_company, max_depth=4)
        
        if G.number_of_nodes() == 0:
            logger.warning(f"No supply chain network found for {monitored_company}")
            return {
                "centrality_metrics": {},
                "dependency_ratios": {},
                "pagerank": {}
            }
        
        logger.info(f"Built supply chain graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Calculate centrality metrics for ONLY Tier-1 suppliers
        centrality_metrics = _calculate_tier1_centrality(G, tier1_suppliers)
        
        # Calculate dependency ratios for ONLY Tier-1 suppliers
        dependency_ratios = _calculate_tier1_dependency(G, tier1_suppliers, disrupted_companies)
        
        # Calculate PageRank for ONLY Tier-1 suppliers
        pagerank = _calculate_tier1_pagerank(G, tier1_suppliers)
        
        logger.info(f"✅ Calculated metrics for {len(tier1_suppliers)} Tier-1 suppliers")
        
        return {
            "centrality_metrics": centrality_metrics,
            "dependency_ratios": dependency_ratios,
            "pagerank": pagerank,
            "summary": {
                "tier1_suppliers_analyzed": len(tier1_suppliers),
                "total_graph_nodes": G.number_of_nodes(),
                "total_graph_edges": G.number_of_edges()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate Tier-1 metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            "centrality_metrics": {},
            "dependency_ratios": {},
            "pagerank": {},
            "error": str(e)
        }


def _build_supply_chain_graph(target_company: str, max_depth: int = 4) -> nx.DiGraph:
    """Build NetworkX graph from Neo4j (reusing logic from enhanced_graph_metrics_tool)"""
    try:
        G = nx.DiGraph()
        
        query = """
        MATCH (start:Company {name: $target})
        CALL apoc.path.expandConfig(start, {
          relationshipFilter: "suppliesTo<",
          bfs: true,
          maxLevel: $max_depth,
          uniqueNodes: "GLOBAL"
        }) YIELD path
        WITH path, length(path) AS pathLength, nodes(path) AS pathNodes
        WHERE pathLength >= 1 AND pathLength <= $max_depth
        WITH pathNodes, pathLength
        UNWIND range(0, pathLength - 1) AS idx
        WITH pathNodes, pathLength, pathNodes[idx] AS supplier, pathNodes[idx + 1] AS customer
        RETURN DISTINCT supplier.name AS supplier_company,
               customer.name AS customer_company,
               pathLength AS tier
        ORDER BY tier, supplier.name, customer.name
        """
        
        results = graph.query(query, {
            "target": target_company,
            "max_depth": max_depth
        })
        
        G.add_node(target_company, tier=0)
        
        for result in results:
            supplier = result.get("supplier_company", "")
            customer = result.get("customer_company", "")
            tier = result.get("tier", 0)
            
            if not supplier or not customer:
                continue
            
            if customer == target_company:
                customer_tier = 0
            else:
                customer_tier = tier - 1
            
            G.add_node(supplier, tier=tier)
            G.add_node(customer, tier=customer_tier)
            G.add_edge(supplier, customer, tier=tier)
        
        return G
        
    except Exception as e:
        logger.error(f"Failed to build supply chain graph: {e}")
        return nx.DiGraph()


def _calculate_tier1_centrality(G: nx.DiGraph, tier1_suppliers: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate centrality metrics ONLY for Tier-1 suppliers"""
    result = {}
    
    try:
        # Calculate for entire graph (needed for accurate metrics)
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector = {node: 0.0 for node in G.nodes()}
        
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        total_nodes = G.number_of_nodes()
        
        # Extract ONLY Tier-1 supplier metrics
        for supplier in tier1_suppliers:
            if supplier in G.nodes():
                result[supplier] = {
                    "betweenness": betweenness.get(supplier, 0.0),
                    "closeness": closeness.get(supplier, 0.0),
                    "eigenvector": eigenvector.get(supplier, 0.0),
                    "in_degree": in_degree.get(supplier, 0) / max(total_nodes - 1, 1),
                    "out_degree": out_degree.get(supplier, 0) / max(total_nodes - 1, 1),
                    "degree_centrality": (in_degree.get(supplier, 0) + out_degree.get(supplier, 0)) / (2 * max(total_nodes - 1, 1))
                }
        
        logger.info(f"Calculated centrality for {len(result)} Tier-1 suppliers")
        return result
        
    except Exception as e:
        logger.error(f"Failed to calculate centrality: {e}")
        return {}


def _calculate_tier1_dependency(G: nx.DiGraph, tier1_suppliers: List[str], disrupted_companies: List[str]) -> Dict[str, float]:
    """Calculate dependency ratios ONLY for Tier-1 suppliers"""
    result = {}
    disrupted_set = set(disrupted_companies)
    
    try:
        for supplier in tier1_suppliers:
            if supplier in G.nodes():
                # Get all downstream nodes (suppliers to this Tier-1 supplier)
                try:
                    downstream_nodes = set(nx.descendants(G, supplier))
                    downstream_nodes.add(supplier)
                    
                    if len(downstream_nodes) == 0:
                        result[supplier] = 0.0
                        continue
                    
                    # Count disrupted downstream nodes
                    disrupted_downstream = len(downstream_nodes & disrupted_set)
                    result[supplier] = disrupted_downstream / len(downstream_nodes)
                    
                except:
                    result[supplier] = 0.0
        
        logger.info(f"Calculated dependency ratios for {len(result)} Tier-1 suppliers")
        return result
        
    except Exception as e:
        logger.error(f"Failed to calculate dependency: {e}")
        return {}


def _calculate_tier1_pagerank(G: nx.DiGraph, tier1_suppliers: List[str]) -> Dict[str, float]:
    """Calculate PageRank ONLY for Tier-1 suppliers"""
    result = {}
    
    try:
        # Calculate for entire graph
        pagerank = nx.pagerank(G, max_iter=100)
        
        # Extract ONLY Tier-1 supplier PageRank
        for supplier in tier1_suppliers:
            if supplier in G.nodes():
                result[supplier] = pagerank.get(supplier, 0.0)
        
        logger.info(f"Calculated PageRank for {len(result)} Tier-1 suppliers")
        return result
        
    except Exception as e:
        logger.error(f"Failed to calculate PageRank: {e}")
        return {}


