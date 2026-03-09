# agents/visualization_agent.py

import logging
import os
from typing import Dict, Any, List
from crewai import Agent

# Import the PROFESSIONAL NETWORKX visualization tool
from tools.networkx_professional_visualization_tool import networkx_professional_visualization_tool_func

class VisualizationAgent(Agent):
    """
    Professional Visualization Agent:
    Creates CEO-ready, top-tier visualizations of disrupted supply chains.
    
    Features:
    - Full disrupted chain visualization from Tier-0 (monitored company) to disrupted suppliers
    - Products displayed on each supply chain link
    - Countries clearly shown for each company
    - Professional, clear design suitable for C-suite presentations
    - Interactive exploration capabilities
    """
    
    def __init__(self, **config):
        super().__init__(**config)
        logging.info("🚀 VisualizationAgent initialized - Creating CEO-ready visualizations")
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate professional, CEO-ready visualization of disrupted supply chains.
        
        Inputs:
            - product_details: Contains chain_map (product map) from ProductSearchAgent
            - kg_results: Supply chain data with full chains
            - disruption_analysis: Disruption details
        
        Returns:
            - visualization_file_path: Path to generated HTML visualization
            - visualization_html: HTML content
        """
        # Get product map from product_details
        product_details = inputs.get("product_details", {})
        if isinstance(product_details, str):
            try:
                import json
                product_details = json.loads(product_details)
            except:
                product_details = {}
        
        chain_map = product_details.get("chain_map", [])
        if not chain_map:
            # Try to get from kg_results and build product map
            kg_results = inputs.get("kg_results", {})
            if kg_results:
                logging.info("[VisualizationAgent] Building product map from kg_results")
                chain_map = self._build_product_map_from_kg(kg_results)
        
        if not chain_map:
            logging.error("[VisualizationAgent] No product map or kg_results provided")
            return {
                "error": "No supply chain data provided for visualization",
                "visualization_file_path": None
            }
        
        # Get disruption information
        disruption_analysis = inputs.get("disruption_analysis", {})
        if isinstance(disruption_analysis, str):
            try:
                import json
                disruption_analysis = json.loads(disruption_analysis)
            except:
                disruption_analysis = {}
        
        # Get monitored company
        kg_results = inputs.get("kg_results", {})
        if isinstance(kg_results, str):
            try:
                import json
                kg_results = json.loads(kg_results)
            except:
                kg_results = {}
        
        monitored_company = kg_results.get("monitored_company", "Tesla Inc")
        disrupted_countries = kg_results.get("disrupted_countries", [])
        
        if not disrupted_countries and disruption_analysis:
            involved = disruption_analysis.get("involved", {})
            disrupted_countries = involved.get("countries", [])
        
        logging.info(f"[VisualizationAgent] Generating CEO-ready visualization for {monitored_company}")
        logging.info(f"[VisualizationAgent] Disrupted countries: {disrupted_countries}")
        logging.info(f"[VisualizationAgent] Supply chain links: {len(chain_map)}")
        
        # Generate PROFESSIONAL NETWORKX visualization for CEO/executive presentations
        # Uses ACTUAL product_map from Product Intelligence Agent
        try:
            logging.info(f"[VisualizationAgent] Calling PROFESSIONAL NETWORKX visualization tool")
            logging.info(f"[VisualizationAgent] Chain map size: {len(chain_map)}")
            logging.info(f"[VisualizationAgent] Chain map sample: {chain_map[:2] if len(chain_map) >= 2 else chain_map}")
            
            # Call the NetworkX visualization function with ACTUAL product_map data
            visualization_result = networkx_professional_visualization_tool_func(
                product_map=chain_map,
                disrupted_countries=disrupted_countries,
                main_company=monitored_company,
                disruption_analysis=disruption_analysis
            )
            
            visualization_file_path = visualization_result.get("file_path", "")
            visualization_html = visualization_result.get("html", "")
            
            if visualization_file_path:
                # Verify file exists
                if os.path.exists(visualization_file_path):
                    file_size = os.path.getsize(visualization_file_path)
                    logging.info(f"✅ [VisualizationAgent] CEO-ready visualization saved to: {visualization_file_path}")
                    logging.info(f"   ✅ File verified: {file_size:,} bytes")
                    logging.info(f"   📊 Ready for C-suite presentation")
                else:
                    logging.error(f"❌ [VisualizationAgent] File not found at: {visualization_file_path}")
                    visualization_file_path = ""
            else:
                logging.error("❌ [VisualizationAgent] No file path returned from visualization tool")
                visualization_file_path = ""
            
            # Pass through company_name for downstream tasks
            company_name = inputs.get("company_name", monitored_company)
            
            return {
                "visualization_file_path": visualization_file_path,
                "visualization_html": visualization_html,
                "visualization_generated": bool(visualization_file_path),
                "company_name": company_name
            }
            
        except Exception as e:
            logging.error(f"❌ [VisualizationAgent] Failed to generate visualization: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "visualization_file_path": None,
                "visualization_generated": False
            }
    
    def _build_product_map_from_kg(self, kg_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build product map from kg_results if product_details is not available.
        This is a fallback method.
        """
        product_map = []
        monitored_company = kg_results.get("monitored_company", "Tesla Inc")
        
        # Process all tiers
        for tier_key in ["tier_1", "tier_2", "tier_3", "tier_4"]:
            tier_chains = kg_results.get(tier_key, [])
            for chain in tier_chains:
                if not chain or len(chain) < 2:
                    continue
                
                # Create edges for each adjacent pair
                for i in range(len(chain) - 1):
                    customer = chain[i]
                    supplier = chain[i+1]
                    
                    product_map.append({
                        "supplier_name": supplier.get("company", "Unknown"),
                        "supplier_country": supplier.get("country", "Unknown"),
                        "supplier_industry": supplier.get("industry", "Unknown"),
                        "product": supplier.get("industry", "Unknown Product"),
                        "customer_name": customer.get("company", "Unknown"),
                        "customer_country": customer.get("country", "Unknown"),
                        "customer_industry": customer.get("industry", "Unknown"),
                        "customer_product": customer.get("industry", "Unknown Product")
                    })
        
        return product_map

