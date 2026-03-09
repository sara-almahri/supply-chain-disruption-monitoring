# enhanced_graph_metrics_tool.py
# Comprehensive graph metrics for robust risk assessment

import logging
import networkx as nx
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from .neo4j_setup import graph

logger = logging.getLogger(__name__)
MAX_COMPANIES = 500

class EnhancedGraphMetricsInput(BaseModel):
    """Input schema for enhanced graph metrics calculation"""
    target_company: str = Field(..., description="Target company name for metrics calculation")
    disrupted_companies: Optional[List[str]] = Field(None, description="List of disrupted companies")
    include_centrality: bool = Field(True, description="Calculate centrality metrics")
    include_pagerank: bool = Field(True, description="Calculate PageRank")
    include_dependency: bool = Field(True, description="Calculate dependency ratio")

def build_networkx_graph(target_company: str, max_depth: int = 4) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from Neo4j for the target company's supply chain.
    Traces up to max_depth tiers using APOC path expansion.
    """
    try:
        G = nx.DiGraph()
        
        # Query to get the supply chain network using APOC path expansion
        # Direction: target <-[suppliesTo]- supplier (upstream suppliers)
        # CRITICAL: Use APOC path.expandConfig to avoid parameter issues in variable-length relationships
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
        
        # Add nodes and edges - build complete paths
        # Store all nodes first to ensure Tier-0 (target) is included
        G.add_node(target_company, tier=0)
        
        for result in results:
            supplier = result.get("supplier_company", "")
            customer = result.get("customer_company", "")
            tier = result.get("tier", 0)
            
            if not supplier or not customer:
                continue
            
            # Add nodes with tier information
            # Customer is closer to target (lower tier), supplier is further (higher tier)
            if customer == target_company:
                customer_tier = 0
            else:
                customer_tier = tier - 1
            
            G.add_node(supplier, tier=tier)
            G.add_node(customer, tier=customer_tier)
            # Edge: supplier -> customer (supplier supplies to customer)
            G.add_edge(supplier, customer, tier=tier)
        
        logger.info(f"Built NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges for {target_company}")
        return G
    
    except Exception as e:
        logger.error(f"Failed to build NetworkX graph: {e}")
        import traceback
        traceback.print_exc()
        return nx.DiGraph()

def calculate_centrality_metrics(G: nx.DiGraph, target_company: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate comprehensive centrality metrics for all nodes in the graph.
    Returns: Dict[node_name, {betweenness, closeness, eigenvector, degree}]
    """
    metrics = {}
    
    try:
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph, returning empty metrics")
            return metrics
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(G, normalized=True)
        
        # Calculate closeness centrality (for strongly connected components)
        # Use reverse graph for closeness (how close suppliers are to target)
        G_reverse = G.reverse()
        closeness = {}
        for node in G.nodes():
            try:
                # Calculate closeness from node to target
                if nx.has_path(G_reverse, node, target_company):
                    path_length = nx.shortest_path_length(G_reverse, node, target_company)
                    closeness[node] = 1.0 / (path_length + 1) if path_length > 0 else 1.0
                else:
                    closeness[node] = 0.0
            except:
                closeness[node] = 0.0
        
        # Calculate eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
        except:
            # Fallback to degree centrality if eigenvector fails
            eigenvector = dict(G.degree())
            max_degree = max(eigenvector.values()) if eigenvector.values() else 1.0
            eigenvector = {k: v / max_degree for k, v in eigenvector.items()}
        
        # Calculate degree centrality (in-degree for suppliers)
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        max_in_degree = max(in_degree.values()) if in_degree.values() else 1.0
        max_out_degree = max(out_degree.values()) if out_degree.values() else 1.0
        
        # Normalize degrees
        in_degree_norm = {k: v / max_in_degree if max_in_degree > 0 else 0.0 
                          for k, v in in_degree.items()}
        out_degree_norm = {k: v / max_out_degree if max_out_degree > 0 else 0.0 
                           for k, v in out_degree.items()}
        
        # Combine all metrics
        for node in G.nodes():
            metrics[node] = {
                "betweenness": betweenness.get(node, 0.0),
                "closeness": closeness.get(node, 0.0),
                "eigenvector": eigenvector.get(node, 0.0),
                "in_degree": in_degree_norm.get(node, 0.0),
                "out_degree": out_degree_norm.get(node, 0.0),
                "degree_centrality": (in_degree_norm.get(node, 0.0) + out_degree_norm.get(node, 0.0)) / 2.0
            }
        
        logger.info(f"Calculated centrality metrics for {len(metrics)} nodes")
        return metrics
    
    except Exception as e:
        logger.error(f"Failed to calculate centrality metrics: {e}")
        return metrics

def calculate_pagerank(G: nx.DiGraph, alpha: float = 0.85) -> Dict[str, float]:
    """
    Calculate PageRank for all nodes in the graph.
    PageRank indicates the importance/influence of a node in the network.
    """
    try:
        if G.number_of_nodes() == 0:
            return {}
        
        pagerank = nx.pagerank(G, alpha=alpha, max_iter=100)
        logger.info(f"Calculated PageRank for {len(pagerank)} nodes")
        return pagerank
    
    except Exception as e:
        logger.error(f"Failed to calculate PageRank: {e}")
        return {}

def calculate_dependency_ratio(
    target_company: str,
    disrupted_companies: List[str],
    G: nx.DiGraph
) -> Dict[str, float]:
    """
    Calculate dependency ratio for each supplier:
    Ratio of disrupted downstream companies to total downstream companies.
    """
    dependency_ratios = {}
    
    try:
        if not disrupted_companies:
            return dependency_ratios
        
        disrupted_set = set(disrupted_companies)
        
        # For each node, calculate how many of its downstream nodes are disrupted
        for node in G.nodes():
            if node == target_company:
                continue
            
            # Get all downstream nodes (nodes reachable from this node)
            try:
                downstream_nodes = set(nx.descendants(G, node))
                downstream_nodes.add(node)  # Include self
                
                if len(downstream_nodes) == 0:
                    dependency_ratios[node] = 0.0
                    continue
                
                # Count disrupted downstream nodes
                disrupted_downstream = len(downstream_nodes & disrupted_set)
                dependency_ratios[node] = disrupted_downstream / len(downstream_nodes)
            
            except:
                dependency_ratios[node] = 0.0
        
        logger.info(f"Calculated dependency ratios for {len(dependency_ratios)} nodes")
        return dependency_ratios
    
    except Exception as e:
        logger.error(f"Failed to calculate dependency ratios: {e}")
        return {}

def calculate_comprehensive_metrics(
    target_company: str,
    disrupted_companies: Optional[List[str]] = None,
    include_centrality: bool = True,
    include_pagerank: bool = True,
    include_dependency: bool = True
) -> Dict[str, Any]:
    """
    Calculate comprehensive graph metrics for robust risk assessment.
    
    Returns:
        {
            "centrality_metrics": {company: {betweenness, closeness, eigenvector, ...}},
            "pagerank": {company: score},
            "dependency_ratios": {company: ratio},
            "summary": {...}
        }
    """
    disrupted_companies = disrupted_companies or []
    
    try:
        # Build the supply chain graph
        G = build_networkx_graph(target_company, max_depth=4)
        
        if G.number_of_nodes() == 0:
            logger.warning(f"No supply chain network found for {target_company}")
            return {
                "error": f"No supply chain network found for {target_company}",
                "centrality_metrics": {},
                "pagerank": {},
                "dependency_ratios": {}
            }
        
        results = {
            "centrality_metrics": {},
            "pagerank": {},
            "dependency_ratios": {},
            "summary": {
                "total_nodes": G.number_of_nodes(),
                "total_edges": G.number_of_edges(),
                "target_company": target_company,
                "disrupted_companies_count": len(disrupted_companies)
            }
        }
        
        # Calculate centrality metrics
        if include_centrality:
            results["centrality_metrics"] = calculate_centrality_metrics(G, target_company)
        
        # Calculate PageRank
        if include_pagerank:
            results["pagerank"] = calculate_pagerank(G)
        
        # Calculate dependency ratios
        if include_dependency and disrupted_companies:
            results["dependency_ratios"] = calculate_dependency_ratio(
                target_company, disrupted_companies, G
            )
        
        logger.info(f"✅ Comprehensive metrics calculated for {target_company}")
        return results
    
    except Exception as e:
        logger.error(f"Failed to calculate comprehensive metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "centrality_metrics": {},
            "pagerank": {},
            "dependency_ratios": {}
        }

def enhanced_graph_metrics_tool_entrypoint(
    target_company: str,
    disrupted_companies: Optional[List[str]] = None,
    include_centrality: bool = True,
    include_pagerank: bool = True,
    include_dependency: bool = True
) -> Dict[str, Any]:
    """
    Entry point for enhanced graph metrics tool.
    Calculates comprehensive metrics for robust risk assessment.
    """
    return calculate_comprehensive_metrics(
        target_company=target_company,
        disrupted_companies=disrupted_companies or [],
        include_centrality=include_centrality,
        include_pagerank=include_pagerank,
        include_dependency=include_dependency
    )

enhanced_graph_metrics_tool = StructuredTool(
    name="enhanced_supply_chain_metrics",
    description="""
    Calculates comprehensive graph metrics for supply chain risk assessment:
    - Centrality metrics: betweenness, closeness, eigenvector, degree
    - PageRank: importance/influence score
    - Dependency ratios: exposure to disrupted suppliers
    
    Use this for robust, data-driven risk assessment.
    """,
    func=enhanced_graph_metrics_tool_entrypoint,
    args_schema=EnhancedGraphMetricsInput
)

