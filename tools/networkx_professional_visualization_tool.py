# tools/networkx_professional_visualization_tool.py
# PROFESSIONAL NETWORKX VISUALIZATION FOR CEO/EXECUTIVE PRESENTATIONS

import os
import logging
import json
import uuid
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

class NetworkXVisualizationInput(BaseModel):
    """Input schema for NetworkX visualization"""
    product_map: List[Dict[str, Any]] = Field(..., description="Product map from Product Intelligence Agent")
    disrupted_countries: Optional[List[str]] = Field(None, description="List of disrupted countries")
    main_company: Optional[str] = Field(None, description="Main company (Tier-0)")
    disruption_analysis: Optional[Dict[str, Any]] = Field(None, description="Disruption analysis")

def build_networkx_executive_visualization(
    product_map: List[Dict[str, Any]],
    disrupted_countries: Optional[List[str]] = None,
    main_company: Optional[str] = None,
    disruption_analysis: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Build a PROFESSIONAL NetworkX visualization for CEO/executive presentations.
    
    This creates a CLEAR, HIERARCHICAL network diagram showing:
    - Full disruption chain from Tier-0 (main company) to Tier-4
    - Products labeled on every edge
    - Countries shown for each company
    - Color-coded disruption propagation
    - Professional executive-level design
    
    Returns:
        tuple: (html_content, file_path)
    """
    disrupted_countries = disrupted_countries or []
    main_company = main_company or "Tesla Inc"
    
    logger.info(f"[NetworkX Viz] Building PROFESSIONAL executive visualization")
    logger.info(f"[NetworkX Viz] Product map entries: {len(product_map)}")
    logger.info(f"[NetworkX Viz] Disrupted countries: {disrupted_countries}")
    logger.info(f"[NetworkX Viz] Main company: {main_company}")
    
    if not product_map or len(product_map) == 0:
        logger.error("❌ [NetworkX Viz] EMPTY PRODUCT MAP - Cannot visualize!")
        error_html = """
        <html>
        <body style="font-family: Arial; padding: 40px; background: #fff3cd;">
            <h1 style="color: #856404;">⚠️ Visualization Error</h1>
            <h2>Empty Product Map</h2>
            <p><strong>The Product Intelligence Agent returned an EMPTY product map.</strong></p>
            <p>This means no supply chain links were identified. Possible causes:</p>
            <ul>
                <li>No disrupted suppliers found in the knowledge graph</li>
                <li>Data not passed correctly from KG Query Agent</li>
                <li>Product Search Agent failed to build the chain map</li>
            </ul>
            <p><strong>Check the logs for ProductSearchAgent output.</strong></p>
        </body>
        </html>
        """
        return error_html, ""
    
    logger.info(f"[NetworkX Viz] Processing {len(product_map)} supply chain links")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Track node information
    node_info = {}
    edge_info = {}
    
    # Process product map to build graph
    for idx, link in enumerate(product_map):
        supplier = link.get("supplier_name", "").strip()
        customer = link.get("customer_name", "").strip()
        supplier_country = link.get("supplier_country", "Unknown").strip()
        customer_country = link.get("customer_country", "Unknown").strip()
        product = link.get("product", "Unknown Product").strip()
        supplier_industry = link.get("supplier_industry", "Unknown").strip()
        
        if not supplier or not customer:
            logger.warning(f"[NetworkX Viz] Skipping invalid link {idx}: supplier={supplier}, customer={customer}")
            continue
        
        # Add nodes
        if supplier not in node_info:
            node_info[supplier] = {
                "country": supplier_country,
                "industry": supplier_industry,
                "is_disrupted": supplier_country in disrupted_countries,
                "is_main": supplier == main_company
            }
        
        if customer not in node_info:
            node_info[customer] = {
                "country": customer_country,
                "industry": link.get("customer_industry", "Unknown"),
                "is_disrupted": customer_country in disrupted_countries,
                "is_main": customer == main_company
            }
        
        # Add edge with product information
        edge_key = (supplier, customer)
        if edge_key not in edge_info:
            edge_info[edge_key] = {
                "product": product,
                "is_disrupted": supplier_country in disrupted_countries or customer_country in disrupted_countries
            }
            G.add_edge(supplier, customer, product=product)
    
    logger.info(f"[NetworkX Viz] Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    if G.number_of_nodes() == 0:
        logger.error("❌ [NetworkX Viz] Graph has NO NODES!")
        return "<html><body><h1>Error: No nodes in graph</h1></body></html>", ""
    
    # Calculate tiers using BFS from main company
    tiers = {}
    if main_company in G:
        from collections import deque
        
        # Build reverse graph (from customers to suppliers)
        reverse_edges = defaultdict(list)
        for supplier, customer in G.edges():
            reverse_edges[customer].append(supplier)
        
        # BFS from main company
        queue = deque([(main_company, 0)])
        visited = {main_company}
        tiers[main_company] = 0
        
        while queue:
            node, tier = queue.popleft()
            
            # Get suppliers of this node
            if node in reverse_edges and tier < 4:
                for supplier in reverse_edges[node]:
                    if supplier not in visited:
                        visited.add(supplier)
                        tiers[supplier] = tier + 1
                        queue.append((supplier, tier + 1))
    
    logger.info(f"[NetworkX Viz] Tiers calculated: {len(tiers)} nodes with tiers")
    
    # Create figure with high DPI for professional quality
    fig = plt.figure(figsize=(24, 16), dpi=150, facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    
    # Hierarchical layout by tier (left to right: Tier-0 to Tier-4)
    pos = {}
    tier_nodes = defaultdict(list)
    
    for node in G.nodes():
        tier = tiers.get(node, 5)
        if tier <= 4:
            tier_nodes[tier].append(node)
    
    # Position nodes hierarchically
    for tier in range(5):
        nodes_in_tier = tier_nodes[tier]
        if not nodes_in_tier:
            continue
        
        # Sort nodes in tier by name for consistency
        nodes_in_tier = sorted(nodes_in_tier)
        
        x = tier * 4  # Horizontal spacing by tier
        num_nodes = len(nodes_in_tier)
        
        # Vertical spacing
        for i, node in enumerate(nodes_in_tier):
            y = (i - num_nodes / 2) * 2  # Center vertically, space by 2
            pos[node] = (x, y)
    
    logger.info(f"[NetworkX Viz] Positioned {len(pos)} nodes")
    
    # Determine node colors and sizes
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    for node in G.nodes():
        info = node_info.get(node, {})
        
        # Color by status
        if info.get("is_main"):
            node_colors.append('#FFD700')  # Gold for main company
            node_sizes.append(3000)
        elif info.get("is_disrupted"):
            node_colors.append('#DC143C')  # Red for disrupted
            node_sizes.append(2000)
        else:
            node_colors.append('#4A90E2')  # Blue for normal
            node_sizes.append(1500)
        
        # Create label with company name, tier, and country
        tier = tiers.get(node, 5)
        country = info.get("country", "Unknown")
        tier_label = f"T{tier}" if tier <= 4 else "T?"
        
        # Truncate long names
        node_name = node if len(node) <= 25 else node[:22] + "..."
        node_labels[node] = f"{node_name}\n[{tier_label}] {country}"
    
    # Draw the network
    logger.info(f"[NetworkX Viz] Drawing network...")
    
    # Draw edges first (behind nodes)
    for (supplier, customer), info in edge_info.items():
        if supplier in pos and customer in pos:
            edge_color = '#DC143C' if info["is_disrupted"] else '#7f8c8d'
            edge_width = 4 if info["is_disrupted"] else 2
            edge_style = 'solid'
            
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(supplier, customer)],
                edge_color=edge_color,
                width=edge_width,
                style=edge_style,
                arrows=True,
                arrowsize=30,
                arrowstyle='->',
                node_size=node_sizes,
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors='black',
        linewidths=3,
        ax=ax
    )
    
    # Draw node labels
    nx.draw_networkx_labels(
        G, pos,
        labels=node_labels,
        font_size=10,
        font_weight='bold',
        font_family='sans-serif',
        ax=ax
    )
    
    # Draw edge labels (products)
    edge_labels = {}
    for (supplier, customer), info in edge_info.items():
        if supplier in pos and customer in pos:
            product = info["product"]
            # Truncate long product names
            product_label = product if len(product) <= 30 else product[:27] + "..."
            edge_labels[(supplier, customer)] = product_label
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=9,
        font_color='#2c3e50',
        font_weight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9),
        ax=ax
    )
    
    # Title and info
    disruption_type = disruption_analysis.get("type", "Unknown") if disruption_analysis else "Unknown"
    disrupted_count = sum(1 for info in node_info.values() if info.get("is_disrupted"))
    
    plt.title(
        f"Supply Chain Disruption Analysis - {main_company}\n"
        f"Disruption: {disruption_type} | Affected Region: {', '.join(disrupted_countries) if disrupted_countries else 'Multiple'} | "
        f"{disrupted_count} Disrupted Suppliers | {G.number_of_nodes()} Total Companies",
        fontsize=20,
        fontweight='bold',
        pad=20
    )
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#FFD700', edgecolor='black', label=f'{main_company} (Tier-0)', linewidth=2),
        mpatches.Patch(facecolor='#DC143C', edgecolor='black', label='Disrupted Supplier', linewidth=2),
        mpatches.Patch(facecolor='#4A90E2', edgecolor='black', label='Normal Supplier', linewidth=2),
        mpatches.Patch(facecolor='#DC143C', label='Disrupted Supply Link', linewidth=4),
        mpatches.Patch(facecolor='#7f8c8d', label='Normal Supply Link', linewidth=2),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=14,
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Legend",
        title_fontsize=16
    )
    
    # Add tier labels
    for tier in range(5):
        if tier in tier_nodes and tier_nodes[tier]:
            x = tier * 4
            ax.text(
                x, max(pos.values(), key=lambda p: p[1])[1] + 3,
                f"Tier {tier}",
                fontsize=16,
                fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#667eea', edgecolor='black', linewidth=2, alpha=0.8),
                color='white'
    )
    
    # Add footer
    ax.text(
        0.5, -0.02,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Supply Chain Disruption Monitoring Framework | CONFIDENTIAL",
        transform=ax.transAxes,
        fontsize=10,
        ha='center',
        style='italic',
        color='gray'
    )
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save to file
    try:
        base_dir = os.path.join(os.getcwd(), "visualizations", "network_plots")
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        company_suffix = main_company.replace(" ", "_").replace("/", "_")
        
        # Save as PNG (high quality)
        png_file = f"NetworkX_Executive_{company_suffix}_{timestamp}_{unique_id}.png"
        png_path = os.path.join(base_dir, png_file)
        
        plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"✅ [NetworkX Viz] Saved PNG: {png_path}")
        
        # Create HTML wrapper to display the image
        html_file = f"NetworkX_Executive_{company_suffix}_{timestamp}_{unique_id}.html"
        html_path = os.path.join(base_dir, html_file)
        
        # Calculate relative path from HTML to PNG
        png_rel_path = os.path.basename(png_path)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supply Chain Disruption Visualization - {main_company}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 95%;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .info {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .viz-container {{
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
        }}
        
        .viz-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        
        .instructions {{
            padding: 30px;
            background: white;
            text-align: center;
        }}
        
        .instructions h2 {{
            color: #1e3c72;
            margin-bottom: 15px;
        }}
        
        .instructions p {{
            font-size: 1.1em;
            color: #7f8c8d;
            line-height: 1.6;
        }}
        
        .download-btn {{
            display: inline-block;
            margin-top: 20px;
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1.2em;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .download-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Supply Chain Disruption Visualization</h1>
            <div class="info">
                <strong>{main_company}</strong> | Executive Report
            </div>
        </div>
        
        <div class="viz-container">
            <img src="{png_rel_path}" alt="Supply Chain Network Diagram" />
        </div>
        
        <div class="instructions">
            <h2>📋 Executive Summary</h2>
            <p>
                This network diagram shows the complete disrupted supply chain from <strong>{main_company}</strong> (Tier-0) 
                to all affected suppliers through Tier-4.
            </p>
            <p style="margin-top: 15px;">
                <strong>Red nodes</strong> indicate disrupted suppliers in affected regions.<br>
                <strong>Gold node</strong> represents {main_company}.<br>
                <strong>Blue nodes</strong> are normal suppliers.<br>
                <strong>Products are labeled on each supply chain link.</strong>
            </p>
            <a href="{png_rel_path}" download class="download-btn">
                ⬇️ Download High-Resolution Image
            </a>
        </div>
        
        <div class="footer">
            <p><strong>CONFIDENTIAL</strong> | {main_company} Supply Chain Risk Assessment</p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ [NetworkX Viz] Saved HTML: {html_path}")
        logger.info(f"✅ [NetworkX Viz] Visualization complete!")
        
        plt.close()
        
        return html_content, html_path
        
    except Exception as e:
        logger.error(f"❌ [NetworkX Viz] Failed to save: {e}")
        import traceback
        traceback.print_exc()
        plt.close()
        return f"<html><body><h1>Error: {e}</h1></body></html>", ""

def networkx_professional_visualization_tool_func(
    product_map: List[Dict[str, Any]],
    disrupted_countries: Optional[List[str]] = None,
    main_company: Optional[str] = None,
    disruption_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates a PROFESSIONAL NetworkX visualization for CEO presentations.
    
    Uses the ACTUAL product_map from Product Intelligence Agent.
    Creates a clear, hierarchical network diagram with:
    - Full disrupted chain (Tier-0 to Tier-4)
    - Products labeled on edges
    - Countries shown for companies
    - Professional executive-level design
    """
    html, file_path = build_networkx_executive_visualization(
        product_map=product_map,
        disrupted_countries=disrupted_countries,
        main_company=main_company,
        disruption_analysis=disruption_analysis
    )
    
    return {"html": html, "file_path": file_path}

networkx_professional_visualization_tool = StructuredTool(
    name="networkx_professional_supply_chain_visualizer",
    description="""
    Generates a PROFESSIONAL NetworkX visualization for CEO/executive presentations.
    
    Uses the ACTUAL product_map from Product Intelligence Agent (NOT random data).
    
    Features:
    - Full disrupted chain from Tier-0 (main company) to Tier-4
    - Products labeled on every supply chain link
    - Countries shown for each company
    - Hierarchical layout (left to right by tier)
    - Professional color-coding and styling
    - High-resolution PNG output
    - Executive-level quality suitable for board presentations
    """,
    func=networkx_professional_visualization_tool_func,
    args_schema=NetworkXVisualizationInput
)



