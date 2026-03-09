#agents/product_search_agent.py

import logging
from typing import Dict, Any, List
from crewai import Agent

# Import the structured networkx plotting tool.
from tools.networkx_plot_tool import networkx_plot_struct

class ProductSearchAgent(Agent):
    """
    Product Search Agent:
    For each disrupted supplier chain (as provided by kg_results), this agent:
      - Enriches each node with product info (currently using industry as product placeholder).
      - Constructs a complete product map linking every adjacent pair of companies along each full chain.
      - Calls the networkx plotting tool to generate an interactive network visualization.
      - Ensures all chains are complete from Tier-0 (monitored company) to disrupted suppliers.
    """
    def __init__(self, **config):
        super().__init__(**config)
        logging.info("🚀 ProductSearchAgent initialized")
        # Note: Product search via external API is currently disabled for speed
        # Products are set to industry names as placeholders
        # Can be enabled later if needed

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        kg_results = inputs.get("kg_results", {})
        if not kg_results:
            return {"error": "No kg_results provided."}

        # Get monitored company and disrupted countries from kg_results
        monitored_company = kg_results.get("monitored_company", "Tesla Inc")
        disrupted_countries = kg_results.get("disrupted_countries", [])
        
        logging.info(f"[ProductSearchAgent] Processing supply chain for {monitored_company}")
        logging.info(f"[ProductSearchAgent] Disrupted countries: {disrupted_countries}")
        
        # CRITICAL: kg_results already contains COMPLETE chains from Tier-0 (monitored_company) to disrupted suppliers
        # Each chain in tier_1, tier_2, tier_3, tier_4 is a complete path:
        # [Tier-0 (monitored_company), Tier-1, Tier-2, Tier-3, Tier-4 (disrupted supplier)]
        # We need to ensure all chains start with Tier-0 and create product map for each adjacent pair

        # Enrich each node in all chains with product info (skip product search for now to speed up)
        # For now, use industry as product placeholder - can enhance later with actual product search
        for tier in ["tier_1", "tier_2", "tier_3", "tier_4"]:
            if tier in kg_results:
                for chain in kg_results[tier]:
                    for node in chain:
                        company_name = node.get("company", "")
                        industry = node.get("industry", "Unknown")
                        # Use industry as product for now (can enhance with product search later)
                        if not node.get("products"):
                            node["products"] = industry or "Unknown Product"
                        # Optionally search for products (commented out for speed)
                        # if company_name:
                        #     product_info = self._search_products(company_name)
                        #     node["products"] = product_info

        # Build a complete product map from all chains in kg_results.
        # Each chain is already complete from Tier-0 to disrupted supplier
        product_map = self.build_full_product_map(kg_results, monitored_company)

        logging.info(f"[ProductSearchAgent] Built product map with {len(product_map)} supply chain links")
        logging.info(f"[ProductSearchAgent] Product map ready for visualization agent")

        # Return product details - visualization will be handled by dedicated VisualizationAgent
        product_details = {
            "chain_map": product_map
        }
        
        return {
            "product_details": product_details
        }
    
    # Note: Product search method is currently disabled
    # Products are set to industry names as placeholders for speed
    # Can be enabled later with OpenAI search or SerperDevTool if needed

    def build_full_product_map(self, kg_results: Dict[str, Any], monitored_company: str) -> List[Dict[str, Any]]:
        """
        Constructs a complete product map from kg_results.
        
        CRITICAL: Each chain in kg_results is already a COMPLETE path from Tier-0 (monitored_company) 
        to the disrupted supplier. The chains are structured as:
        - tier_1: [[Tier-0, Tier-1-disrupted], ...]
        - tier_2: [[Tier-0, Tier-1, Tier-2-disrupted], ...]
        - tier_3: [[Tier-0, Tier-1, Tier-2, Tier-3-disrupted], ...]
        - tier_4: [[Tier-0, Tier-1, Tier-2, Tier-3, Tier-4-disrupted], ...]
        
        We create an edge for every adjacent pair in each chain to show the complete supply chain path.
        
        Args:
            kg_results: Dictionary with 'tier_1', 'tier_2', 'tier_3', 'tier_4' keys, each containing
                       lists of complete chains from Tier-0 to disrupted supplier
            monitored_company: The main company (Tier-0) being monitored (e.g., "Tesla Inc")
        
        Returns:
            List of edge dictionaries representing the full supply chain paths.
        """
        product_map = []
        all_chains = []
        
        # Combine all chains from different tiers into one list
        for tier in ["tier_1", "tier_2", "tier_3", "tier_4"]:
            if tier in kg_results:
                all_chains.extend(kg_results[tier])
        
        logging.info(f"[ProductSearchAgent] Processing {len(all_chains)} complete supply chain paths")
        
        # Process each chain - they should already be complete from Tier-0 to disrupted supplier
        for chain_idx, chain in enumerate(all_chains):
            if not chain or len(chain) < 2:
                logging.warning(f"[ProductSearchAgent] Skipping invalid chain {chain_idx}: {chain}")
                continue
            
            # Verify chain starts with monitored_company (Tier-0)
            first_company = chain[0].get("company", "")
            if first_company != monitored_company:
                logging.warning(f"[ProductSearchAgent] Chain {chain_idx} does not start with {monitored_company}, found: {first_company}")
                # Prepend monitored_company if missing (shouldn't happen, but safety check)
                monitored_node = {
                    "company": monitored_company,
                    "country": chain[0].get("country", "Unknown"),  # Use country from first node if available
                    "industry": "Automotive",  # Default - could be enhanced
                    "products": "Automotive Manufacturing"  # Default product
                }
                chain.insert(0, monitored_node)
            
            # Create an edge for every adjacent pair in the chain
            # This ensures ALL nodes are connected in a path from Tier-0 to disrupted supplier
            for i in range(len(chain) - 1):
                customer = chain[i]      # Company at tier i (closer to monitored_company)
                supplier = chain[i+1]    # Supplier at tier i+1 (supplies to customer)
                
                # Get product information
                supplier_product = supplier.get("products") or supplier.get("industry") or "Unknown Product"
                customer_product = customer.get("products") or customer.get("industry") or "Unknown Product"
                
                product_map.append({
                    "supplier_name": supplier.get("company", "Unknown Supplier"),
                    "supplier_country": supplier.get("country", "Unknown Country"),
                    "supplier_industry": supplier.get("industry", "Unknown"),
                    "product": supplier_product,
                    "customer_name": customer.get("company", "Unknown Customer"),
                    "customer_country": customer.get("country", "Unknown Country"),
                    "customer_industry": customer.get("industry", "Unknown"),
                    "customer_product": customer_product
                })
                
                logging.debug(f"  Added link: {supplier.get('company')} -> {customer.get('company')} (product: {supplier_product})")
        
        logging.info(f"[ProductSearchAgent] Built product map with {len(product_map)} supply chain links")
        logging.info(f"[ProductSearchAgent] All chains are complete from {monitored_company} (Tier-0) to disrupted suppliers")
        
        return product_map
