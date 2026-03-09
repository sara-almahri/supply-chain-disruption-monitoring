# tools/professional_visualization_tool.py
# CEO-Ready Professional Supply Chain Disruption Visualization

import logging
import os
import uuid
import networkx as nx
from pyvis.network import Network
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from datetime import datetime

logging.basicConfig(level=logging.INFO)

class ProfessionalVisualizationInput(BaseModel):
    """Input schema for professional CEO-ready visualization"""
    product_map: List[Dict[str, Any]] = Field(
        ...,
        description="List of supply chain links with supplier_name, supplier_country, product, customer_name, customer_country"
    )
    disrupted_countries: Optional[List[str]] = Field(
        default=None,
        description="Countries experiencing disruption (highlighted in red)"
    )
    main_company: Optional[str] = Field(
        default=None,
        description="The monitored company (Tier-0, highlighted in gold)"
    )
    disruption_analysis: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Disruption analysis details for context"
    )

def build_professional_visualization(
    product_map: List[Dict[str, Any]],
    disrupted_countries: Optional[List[str]] = None,
    main_company: Optional[str] = None,
    disruption_analysis: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Build a CEO-ready, professional supply chain disruption visualization.
    
    Features:
    - Clear, hierarchical layout (Tier-0 to Tier-4)
    - Products displayed on every supply chain link
    - Countries clearly shown for each company
    - Disruption propagation highlighted in red
    - Professional color scheme and styling
    - Interactive exploration
    - Executive summary panel
    """
    disrupted_countries = disrupted_countries or []
    main_company = main_company or "Tesla Inc"
    
    # Initialize PyVis network with professional settings
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#2d3436",
        directed=True,
        select_menu=True,
        filter_menu=True
    )
    
    # Build NetworkX graph for tier calculation
    G = nx.DiGraph()
    companies_data = {}
    all_countries = set()
    disruption_sources = set()
    
    # CRITICAL: Log product_map to debug empty visualization
    logging.info(f"[build_professional_visualization] Processing product_map with {len(product_map)} links")
    if not product_map:
        logging.error("❌ [build_professional_visualization] product_map is EMPTY! Cannot generate visualization.")
        # Return error HTML
        error_html = """
        <html>
            <body style="padding: 40px; font-family: Arial; background: #fff3cd; color: #856404; border: 2px solid #ffc107;">
                <h1>⚠️ Visualization Error</h1>
                <h2>No Supply Chain Data Available</h2>
                <p>The product map is empty. This means no supply chain links were found or passed to the visualization tool.</p>
                <p><strong>Possible causes:</strong></p>
                <ul>
                    <li>ProductSearchAgent did not build the product map correctly</li>
                    <li>Data was not passed through the task chain properly</li>
                    <li>No supply chain relationships exist in the knowledge graph</li>
                </ul>
                <p>Please check the logs for more details.</p>
            </body>
        </html>
        """
        return error_html, ""
    
    # Process product map to build graph
    for link in product_map:
        supplier = link.get("supplier_name", "").strip()
        customer = link.get("customer_name", "").strip()
        supplier_country = link.get("supplier_country", "Unknown").strip()
        customer_country = link.get("customer_country", "Unknown").strip()
        product = link.get("product", "Unknown Product").strip()
        
        if not supplier or not customer:
            continue
        
        # Track countries
        if supplier_country != "Unknown":
            all_countries.add(supplier_country)
        if customer_country != "Unknown":
            all_countries.add(customer_country)
        
        # Track companies
        if supplier not in companies_data:
            companies_data[supplier] = {
                "country": supplier_country,
                "industry": link.get("supplier_industry", "Unknown"),
                "is_disrupted": supplier_country in disrupted_countries,
                "products": [product]
            }
            if supplier_country in disrupted_countries:
                disruption_sources.add(supplier)
        else:
            if product not in companies_data[supplier]["products"]:
                companies_data[supplier]["products"].append(product)
        
        if customer not in companies_data:
            companies_data[customer] = {
                "country": customer_country,
                "industry": link.get("customer_industry", "Unknown"),
                "is_disrupted": customer_country in disrupted_countries,
                "products": []
            }
        
        # Add edge to graph
        G.add_edge(supplier, customer, product=product)
    
    # Calculate tiers using BFS from main_company
    tiers = {}
    if main_company in G:
        # BFS from main_company (Tier-0)
        queue = [(main_company, 0)]
        visited = {main_company}
        
        while queue:
            node, tier = queue.pop(0)
            tiers[node] = tier
            
            # Traverse upstream (suppliers)
            for supplier in G.predecessors(node):
                if supplier not in visited and tier < 4:  # Limit to Tier-4
                    visited.add(supplier)
                    queue.append((supplier, tier + 1))
    
    # Set tier for main_company
    tiers[main_company] = 0
    
    # Professional color scheme
    COLORS = {
        "main_company": {"bg": "#FFD700", "border": "#DAA520", "text": "#000000"},  # Gold
        "disruption_source": {"bg": "#DC143C", "border": "#8B0000", "text": "#FFFFFF"},  # Crimson
        "disrupted": {"bg": "#FF6347", "border": "#CD5C5C", "text": "#FFFFFF"},  # Tomato
        "normal": {"bg": "#4A90E2", "border": "#2E5C8A", "text": "#FFFFFF"},  # Blue
        "country_disrupted": {"bg": "#DC143C", "border": "#8B0000", "text": "#FFFFFF"},
        "country_normal": {"bg": "#2ECC71", "border": "#27AE60", "text": "#FFFFFF"}  # Green
    }
    
    # Configure network with professional hierarchical layout
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 14,
          "face": "Arial",
          "bold": true,
          "color": "#2d3436"
        },
        "borderWidth": 3,
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.2)",
          "size": 10,
          "x": 3,
          "y": 3
        },
        "shapeProperties": {
          "useBorderWithImage": true
        }
      },
      "edges": {
        "smooth": {
          "type": "dynamic",
          "roundness": 0.5
        },
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1.2,
            "type": "arrow"
          }
        },
        "color": {
          "inherit": "from"
        },
        "width": 3,
        "hoverWidth": 5,
        "font": {
          "size": 11,
          "align": "middle",
          "color": "#34495e"
        }
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "levelSeparation": 300,
          "nodeSpacing": 200,
          "treeSpacing": 250,
          "direction": "LR",
          "sortMethod": "directed",
          "shakeTowards": "none"
        }
      },
      "physics": {
        "enabled": false,
        "stabilization": {
          "enabled": true,
          "iterations": 200
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "keyboard": {
          "enabled": true,
          "speed": {
            "x": 10,
            "y": 10,
            "zoom": 0.05
          }
        },
        "multiselect": true,
        "zoomView": true,
        "dragView": true
      }
    }
    """)
    
    # Add company nodes with professional styling
    # CRITICAL: Sort companies for deterministic, consistent visualization
    nodes_added = set()
    sorted_companies = sorted(companies_data.items(), key=lambda x: (tiers.get(x[0], 5), x[0]))  # Sort by tier, then name
    for company, data in sorted_companies:
        tier = tiers.get(company, 5)
        if tier > 4:
            continue  # Skip beyond Tier-4
        
        country = data["country"]
        is_disrupted = data["is_disrupted"]
        is_main = (company == main_company)
        is_source = (company in disruption_sources)
        industry = data.get("industry", "Unknown")
        products = ", ".join(data.get("products", [])[:3])  # Show first 3 products
        
        # Determine node color
        if is_main:
            color = COLORS["main_company"]
            size = 80
            shape = "star"
        elif is_source:
            color = COLORS["disruption_source"]
            size = 60
            shape = "diamond"
        elif is_disrupted:
            color = COLORS["disrupted"]
            size = 50
            shape = "dot"
        else:
            color = COLORS["normal"]
            size = 45
            shape = "dot"
        
        # Create professional label
        tier_label = f"Tier-{tier}" if tier > 0 else "Tier-0 ({main_company.split()[0]})"
        node_label = f"{company}\n{tier_label}\n📍 {country}"
        
        # Professional tooltip
        tooltip = f"""
        <div style="font-family: Arial, sans-serif; padding: 10px;">
          <h3 style="margin: 0 0 10px 0; color: {color['text']};">{company}</h3>
          <p style="margin: 5px 0;"><b>Tier:</b> {tier_label}</p>
          <p style="margin: 5px 0;"><b>Country:</b> {country}</p>
          <p style="margin: 5px 0;"><b>Industry:</b> {industry}</p>
          <p style="margin: 5px 0;"><b>Products:</b> {products if products else 'N/A'}</p>
          <p style="margin: 5px 0;"><b>Status:</b> {'🔴 DISRUPTION SOURCE' if is_source else '🟠 DISRUPTED' if is_disrupted else '🟢 Normal'}</p>
        </div>
        """
        
        net.add_node(
            company,
            label=node_label,
            color=color["bg"],
            borderWidth=4,
            borderColor=color["border"],
            size=size,
            shape=shape,
            font={"size": 12, "color": color["text"], "face": "Arial Bold"},
            title=tooltip,
            level=tier,
            mass=6 if is_main else (5 if is_source else 4)
        )
        nodes_added.add(company)
    
    # Add country nodes
    # CRITICAL: Sort countries for deterministic, consistent visualization
    sorted_countries = sorted(all_countries)  # Alphabetical sort
    for country in sorted_countries:
        is_disrupted = (country in disrupted_countries)
        color = COLORS["country_disrupted"] if is_disrupted else COLORS["country_normal"]
        
        net.add_node(
            country,
            label=f"🌍 {country}",
            color=color["bg"],
            borderWidth=3,
            borderColor=color["border"],
            size=55,
            shape="box",
            font={"size": 14, "color": color["text"], "face": "Arial Bold"},
            title=f"Country: {country}\nStatus: {'🔴 DISRUPTED' if is_disrupted else '🟢 Normal'}",
            level=5,
            margin=20
        )
        nodes_added.add(country)
    
    # Add supply chain edges with product labels
    # CRITICAL: Sort product_map for deterministic, consistent visualization
    sorted_product_map = sorted(product_map, key=lambda x: (
        x.get("supplier_name", ""),
        x.get("customer_name", ""),
        x.get("product", "")
    ))
    for link in sorted_product_map:
        supplier = link.get("supplier_name", "").strip()
        customer = link.get("customer_name", "").strip()
        product = link.get("product", "Unknown Product").strip()
        
        if supplier not in nodes_added or customer not in nodes_added:
            continue
        
        supplier_disrupted = companies_data.get(supplier, {}).get("is_disrupted", False)
        customer_disrupted = companies_data.get(customer, {}).get("is_disrupted", False)
        
        # Edge color based on disruption
        if supplier_disrupted or customer_disrupted:
            edge_color = "#DC143C"  # Crimson for disrupted
            edge_width = 5
        else:
            edge_color = "#34495e"  # Dark gray for normal
            edge_width = 3
        
        # Product label (truncate if too long)
        product_label = product if len(product) <= 25 else product[:22] + "..."
        
        net.add_edge(
            supplier,
            customer,
            label=product_label,
            color=edge_color,
            width=edge_width,
            arrows="to",
            title=f"Product: {product}\nFrom: {supplier}\nTo: {customer}",
            font={"size": 10, "align": "middle", "color": "#2d3436"}
        )
    
    # Add location edges (company -> country)
    # CRITICAL: Sort for deterministic, consistent visualization
    sorted_companies_for_location = sorted(companies_data.items(), key=lambda x: x[0])  # Sort by company name
    for company, data in sorted_companies_for_location:
        country = data["country"]
        if country != "Unknown" and country in nodes_added and company in nodes_added:
            is_disrupted = data["is_disrupted"]
            net.add_edge(
                company,
                country,
                color="#95a5a6" if not is_disrupted else "#DC143C",
                width=2,
                dashes=True,
                arrows="to",
                title=f"Location: {company} is located in {country}",
                smooth={"type": "straightCross", "roundness": 0}
            )
    
    # Build executive summary
    disruption_type = disruption_analysis.get("type", "Unknown") if disruption_analysis else "Unknown"
    affected_industries = []
    if disruption_analysis:
        involved = disruption_analysis.get("involved", {})
        affected_industries = involved.get("industries", [])
    
    # Professional legend and executive summary
    executive_summary = f"""
    <div id="executive-summary" style="
        position: absolute;
        top: 20px;
        left: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        z-index: 1000;
        font-family: 'Arial', sans-serif;
        max-width: 350px;
        border: 3px solid rgba(255,255,255,0.3);
    ">
        <h2 style="margin: 0 0 15px 0; font-size: 20px; border-bottom: 2px solid rgba(255,255,255,0.5); padding-bottom: 10px;">
            📊 Executive Summary
        </h2>
        <div style="margin-bottom: 12px;">
            <p style="margin: 5px 0; font-size: 13px;"><b>Monitored Company:</b><br>{main_company}</p>
            <p style="margin: 5px 0; font-size: 13px;"><b>Disruption Type:</b><br>{disruption_type}</p>
            <p style="margin: 5px 0; font-size: 13px;"><b>Disrupted Countries:</b><br>{', '.join(disrupted_countries) if disrupted_countries else 'None'}</p>
            <p style="margin: 5px 0; font-size: 13px;"><b>Affected Industries:</b><br>{', '.join(affected_industries[:3]) if affected_industries else 'N/A'}</p>
            <p style="margin: 5px 0; font-size: 13px;"><b>Disrupted Suppliers:</b><br>{len(disruption_sources)} companies</p>
            <p style="margin: 5px 0; font-size: 13px;"><b>Total Companies:</b><br>{len(companies_data)} companies</p>
        </div>
    </div>
    """
    
    legend_html = f"""
    <div id="legend" style="
        position: absolute;
        top: 20px;
        right: 20px;
        background: rgba(255,255,255,0.98);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        z-index: 1000;
        font-family: 'Arial', sans-serif;
        max-width: 320px;
        border: 3px solid #34495e;
    ">
        <h3 style="margin: 0 0 20px 0; color: #2d3436; border-bottom: 3px solid #34495e; padding-bottom: 12px; font-size: 18px;">
            🎨 Legend
        </h3>
        
        <div style="margin-bottom: 20px;">
            <h4 style="margin: 0 0 12px 0; color: #e74c3c; font-size: 14px; font-weight: bold;">Node Types</h4>
            <div style="display: grid; grid-template-columns: auto 1fr; gap: 12px 18px; align-items: center;">
                <div style="width: 30px; height: 30px; background: #FFD700; border: 4px solid #DAA520; border-radius: 50%; clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);"></div>
                <div><b>Main Company (Tier-0)</b></div>
                
                <div style="width: 30px; height: 30px; background: #DC143C; border: 4px solid #8B0000; transform: rotate(45deg);"></div>
                <div><b>Disruption Source</b></div>
                
                <div style="width: 30px; height: 30px; background: #FF6347; border: 4px solid #CD5C5C; border-radius: 50%;"></div>
                <div>Disrupted Company</div>
                
                <div style="width: 30px; height: 30px; background: #4A90E2; border: 4px solid #2E5C8A; border-radius: 50%;"></div>
                <div>Normal Supplier</div>
                
                <div style="width: 30px; height: 30px; background: #DC143C; border: 4px solid #8B0000;"></div>
                <div>Country (Disrupted)</div>
                
                <div style="width: 30px; height: 30px; background: #2ECC71; border: 4px solid #27AE60;"></div>
                <div>Country (Normal)</div>
            </div>
        </div>
        
        <div style="border-top: 3px solid #ddd; padding-top: 15px; margin-top: 15px;">
            <h4 style="margin: 0 0 12px 0; color: #34495e; font-size: 14px; font-weight: bold;">Link Types</h4>
            <div style="display: grid; grid-template-columns: auto 1fr; gap: 12px 18px; align-items: center;">
                <div style="border-bottom: 4px solid #DC143C; width: 50px;"></div>
                <div>Disrupted Supply Chain</div>
                
                <div style="border-bottom: 3px solid #34495e; width: 50px;"></div>
                <div>Normal Supply Chain</div>
                
                <div style="border-bottom: 2px dashed #95a5a6; width: 50px;"></div>
                <div>Location Link</div>
            </div>
        </div>
        
        <div style="margin-top: 15px; padding-top: 15px; border-top: 3px solid #ddd; font-size: 11px; color: #777;">
            <p style="margin: 5px 0;"><b>💡 Tips:</b></p>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>Hover over nodes for details</li>
                <li>Products shown on supply chain links</li>
                <li>Red paths indicate disruption propagation</li>
                <li>Drag to explore, zoom to focus</li>
            </ul>
        </div>
    </div>
    """
    
    # Save visualization
    try:
        base_dir = os.path.join(os.getcwd(), "visualizations", "network_plots")
        os.makedirs(base_dir, exist_ok=True)
        
        # Unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_id = str(uuid.uuid4())[:8]
        company_suffix = main_company.replace(" ", "_").replace("/", "_") if main_company else "supply_chain"
        file_name = f"CEO_SupplyChain_{company_suffix}_{timestamp}_{unique_id}.html"
        file_path = os.path.join(base_dir, file_name)
        
        logging.info(f"🔄 Generating CEO-ready visualization: {file_name}")
        logging.info(f"   📊 Total nodes to visualize: {len(nodes_added)}")
        logging.info(f"   📊 Total companies: {len(companies_data)}")
        logging.info(f"   📊 Total countries: {len(all_countries)}")
        logging.info(f"   📊 Total product map links: {len(product_map)}")
        
        # CRITICAL: Force write HTML to ensure nodes/edges are included
        # PyVis sometimes has issues with save_graph not including data
        net.write_html(file_path)
        
        # Inject executive summary and legend
        with open(file_path, "r+", encoding="utf-8") as f:
            content = f.read()
            content = content.replace("</body>", f"{executive_summary}{legend_html}</body>")
            # Add professional CSS
            custom_css = """
            <style>
                body {
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                }
                #mynetworkid {
                    border: 4px solid #34495e;
                    border-radius: 12px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    background: white;
                }
            </style>
            """
            content = content.replace("</head>", f"{custom_css}</head>")
            f.seek(0)
            f.write(content)
            f.truncate()
        
        # Verify file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Visualization file not created at {file_path}")
        
        file_size = os.path.getsize(file_path)
        logging.info(f"✅ CEO-ready visualization saved: {file_path}")
        logging.info(f"   ✅ File size: {file_size:,} bytes")
        logging.info(f"   📊 Ready for C-suite presentation")
        
        # Read HTML content
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        return html_content, file_path
        
    except Exception as e:
        logging.error(f"Visualization generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        error_html = f"""
        <html>
            <body style="padding: 20px; color: #dc3545;">
                <h2>Visualization Error</h2>
                <p>Failed to generate network diagram: {str(e)}</p>
                <pre>{traceback.format_exc()}</pre>
            </body>
        </html>
        """
        return error_html, ""

def professional_visualization_tool_func(
    product_map: List[Dict[str, Any]],
    disrupted_countries: Optional[List[str]] = None,
    main_company: Optional[str] = None,
    disruption_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates CEO-ready, professional supply chain disruption visualization.
    
    Features:
    - Full disrupted chain from Tier-0 to disrupted suppliers
    - Products on every link
    - Countries for each company
    - Professional design for C-suite presentations
    """
    html, file_path = build_professional_visualization(
        product_map=product_map,
        disrupted_countries=disrupted_countries,
        main_company=main_company,
        disruption_analysis=disruption_analysis
    )
    
    return {"html": html, "file_path": file_path}

professional_visualization_tool = StructuredTool(
    name="professional_supply_chain_visualizer",
    description=(
        "Generates CEO-ready, professional supply chain disruption visualizations. "
        "Features: Full disrupted chains (Tier-0 to Tier-4), products on links, "
        "countries for companies, professional design suitable for C-suite presentations."
    ),
    func=professional_visualization_tool_func,
    args_schema=ProfessionalVisualizationInput
)

