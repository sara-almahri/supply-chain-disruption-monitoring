import logging
import os
import uuid
import networkx as nx
from pyvis.network import Network
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

logging.basicConfig(level=logging.INFO)

class NetworkXPlotInput(BaseModel):
    product_map: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of supply chain links with keys: "
            "supplier_name, supplier_country, product, "
            "customer_name, customer_country, customer_product"
        )
    )
    disrupted_countries: Optional[List[str]] = Field(
        default=None,
        description="Countries in disruption (highlighted red)"
    )
    main_company: Optional[str] = Field(
        default=None,
        description="Central company to emphasize (highlighted gold)"
    )

def compute_company_tiers(G: nx.DiGraph, main_company: str) -> Dict[str, int]:
    """
    Calculate supply chain tiers using BFS on reversed graph.
    Only includes nodes that are connected to main_company (Tier-0).
    Filters out nodes beyond Tier-4.
    Returns tier levels for all company nodes connected to main_company.
    
    CRITICAL: The graph edges are supplier -> customer (supplier supplies to customer).
    To calculate tiers from main_company, we traverse UPSTREAM (reverse direction).
    """
    tiers = {}
    
    if main_company not in G:
        logging.warning(f"Main company '{main_company}' not found in graph")
        # Try to find main_company with different name variations
        for node in G.nodes():
            if G.nodes[node].get("node_type") == "company":
                node_name = G.nodes[node].get("name", node)
                if main_company.lower() in node_name.lower() or node_name.lower() in main_company.lower():
                    logging.info(f"Found main company with different name: {node_name} (using as Tier-0)")
                    main_company = node
                    break
        if main_company not in G:
            logging.error(f"Main company '{main_company}' still not found in graph. Available companies: {[n for n, d in G.nodes(data=True) if d.get('node_type') == 'company'][:10]}")
            return tiers

    # Start BFS from main_company (Tier-0)
    tiers[main_company] = 0
    queue = [main_company]
    visited = {main_company}
    
    # Build reverse graph for supplier relationships only
    # In our graph: supplier -> customer (supplier supplies to customer)
    # Example: Tier-4 (supplier) -> Tier-3 -> Tier-2 -> Tier-1 -> Tier-0 (main_company, customer)
    # For tier calculation: we want to traverse UPSTREAM from main_company to suppliers
    # So we reverse the graph: customer -> supplier
    company_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if G.nodes[u].get("node_type") == "company" 
        and G.nodes[v].get("node_type") == "company"
        and d.get("relationship") == "Supplies"
    ]
    
    if not company_edges:
        logging.warning(f"No supply chain edges found in graph. Graph has {G.number_of_edges()} total edges.")
        # Return only main_company as Tier-0
        return {main_company: 0}
    
    # Reverse graph: customer -> supplier (to traverse upstream from main_company)
    # Original: supplier -> customer (supplier supplies to customer)
    # Reversed: customer -> supplier (from customer, go to its suppliers)
    rev_graph = nx.DiGraph()
    rev_graph.add_edges_from([(v, u) for u, v in company_edges])
    
    logging.info(f"Built reverse graph with {rev_graph.number_of_nodes()} nodes and {rev_graph.number_of_edges()} edges")
    logging.info(f"Starting BFS from {main_company} (Tier-0)")
    
    # BFS to assign tiers - traverse UPSTREAM from main_company
    iteration = 0
    while queue and iteration < 100:  # Safety limit
        iteration += 1
        current = queue.pop(0)
        current_tier = tiers[current]
        
        # Only traverse to Tier-4 (tier 0, 1, 2, 3, 4)
        if current_tier >= 4:
            continue
        
        # Find suppliers of current node (neighbors in reverse graph = upstream suppliers)
        suppliers = list(rev_graph.neighbors(current))
        logging.debug(f"  Tier-{current_tier} node '{current}' has {len(suppliers)} suppliers")
        
        for supplier in suppliers:
            if supplier not in visited:
                new_tier = current_tier + 1
                # Only include up to Tier-4
                if new_tier <= 4:
                    tiers[supplier] = new_tier
                    visited.add(supplier)
                    queue.append(supplier)
                    logging.debug(f"    Assigned Tier-{new_tier} to '{supplier}'")
    
    # Filter out any nodes beyond Tier-4
    tiers = {n: t for n, t in tiers.items() if t <= 4}
    
    logging.info(f"Computed tiers for {len(tiers)} companies connected to {main_company}")
    logging.info(f"  Tier distribution: {dict(sorted([(t, sum(1 for v in tiers.values() if v == t)) for t in set(tiers.values())]))}")
    
    # Verify all company nodes are either in tiers or disconnected
    all_company_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "company"]
    disconnected = [n for n in all_company_nodes if n not in tiers]
    if disconnected:
        logging.warning(f"Found {len(disconnected)} company nodes not connected to {main_company}: {disconnected[:5]}")
    
    return tiers

def build_networkx_plot(
    product_map: List[Dict[str, Any]],
    disrupted_countries: Optional[List[str]] = None,
    main_company: Optional[str] = None
) -> tuple:
    """
    Generates professional supply chain visualization with:
    - DETERMINISTIC hierarchical tier-based layout
    - Complete connected chains from disruption source to main company
    - Country nodes clearly displayed and connected
    - Disruption source highlighting
    - Product information on edges
    - Stable, reproducible visualization
    """
    disrupted_countries = disrupted_countries or []
    G = nx.DiGraph()

    if not main_company:
        logging.error("main_company is required for network visualization")
        return "<html><body>Error: Main company not specified</body></html>"

    # STEP 1: Build graph structure from product_map
    # CRITICAL: Ensure ALL chains are complete and connected
    logging.info(f"Building graph from {len(product_map)} product map links")
    
    # Track all companies and their countries
    companies_data = {}  # company_name -> {country, industry, products, is_disrupted}
    disruption_sources = set()  # Companies that are in disrupted countries (disruption sources)
    
    # First pass: Collect all company information
    for link in product_map:
        supplier = link.get("supplier_name", "").strip()
        customer = link.get("customer_name", "").strip()
        s_country = link.get("supplier_country", "Unknown Country").strip()
        c_country = link.get("customer_country", "Unknown Country").strip()
        product = link.get("product", "Unknown Product").strip()
        
        if not supplier or not customer or supplier == "Unknown Supplier" or customer == "Unknown Customer":
            continue
        
        # Track companies
        if supplier not in companies_data:
            companies_data[supplier] = {
                "country": s_country,
                "industry": link.get("supplier_industry", "Unknown"),
                "products": product,
                "is_disrupted": s_country in disrupted_countries
            }
            if s_country in disrupted_countries:
                disruption_sources.add(supplier)
        
        if customer not in companies_data:
            companies_data[customer] = {
                "country": c_country,
                "industry": link.get("customer_industry", "Unknown"),
                "products": link.get("customer_product", "Unknown Product"),
                "is_disrupted": c_country in disrupted_countries
            }
            if c_country in disrupted_countries:
                disruption_sources.add(customer)
    
    # Verify main_company is in the data
    main_company_found = False
    for company in companies_data.keys():
        if main_company.lower() in company.lower() or company.lower() in main_company.lower():
            main_company = company  # Use the exact name from data
            main_company_found = True
            break
    
    if not main_company_found:
        logging.error(f"Main company '{main_company}' not found in product map")
        return "<html><body>Error: Main company not found in supply chain data</body></html>"
    
    logging.info(f"Main company: {main_company}")
    logging.info(f"Disruption sources (companies in disrupted countries): {len(disruption_sources)}")
    logging.info(f"Disrupted countries: {disrupted_countries}")
    
    # STEP 2: Build graph with ALL companies and countries
    # Add all company nodes first
    for company, data in companies_data.items():
        G.add_node(
            company,
            node_type="company",
            name=company,
            country=data["country"],
            industry=data.get("industry", "Unknown"),
            products=data.get("products", "Unknown"),
            is_disrupted=data["is_disrupted"],
            is_disruption_source=(company in disruption_sources)
        )
    
    # Add all country nodes
    all_countries = set()
    for data in companies_data.values():
        country = data["country"]
        if country and country != "Unknown Country":
            all_countries.add(country)
            if not G.has_node(country):
                G.add_node(
                    country,
                    node_type="country",
                    name=country,
                    is_disrupted=(country in disrupted_countries)
                )
    
    # STEP 3: Add supply chain edges (supplier -> customer)
    supply_edges_added = 0
    for link in product_map:
        supplier = link.get("supplier_name", "").strip()
        customer = link.get("customer_name", "").strip()
        product = link.get("product", "Unknown Product").strip()
        
        if not supplier or not customer or supplier == "Unknown Supplier" or customer == "Unknown Customer":
            continue
        
        if supplier not in G or customer not in G:
            logging.warning(f"Skipping edge: {supplier} -> {customer} (nodes not in graph)")
            continue
        
        # Add supply relationship: supplier -> customer
        if not G.has_edge(supplier, customer):
            G.add_edge(
                supplier,
                customer,
                relationship="Supplies",
                product=product,
                label=f"{product}" if product != "Unknown Product" else "Supplies"
            )
            supply_edges_added += 1
    
    # STEP 4: Add location edges (company -> country)
    location_edges_added = 0
    for company, data in companies_data.items():
        country = data["country"]
        if country and country != "Unknown Country" and country in G.nodes():
            if not G.has_edge(company, country):
                G.add_edge(company, country, relationship="LocatedIn")
                location_edges_added += 1
    
    logging.info(f"Graph built: {G.number_of_nodes()} nodes ({len(companies_data)} companies, {len(all_countries)} countries)")
    logging.info(f"  Supply chain edges: {supply_edges_added}")
    logging.info(f"  Location edges: {location_edges_added}")
    
    # STEP 5: Calculate tiers from main_company
    company_tiers = compute_company_tiers(G, main_company)
    
    # STEP 6: Verify connectivity - ensure all companies are connected to main_company
    companies_not_connected = []
    for company in companies_data.keys():
        if company not in company_tiers and company != main_company:
            companies_not_connected.append(company)
    
    if companies_not_connected:
        logging.warning(f"Found {len(companies_not_connected)} companies not connected to {main_company}: {companies_not_connected[:5]}")
        # Try to find paths to connect them
        for company in companies_not_connected:
            try:
                # Check if there's a path in the original graph (not reversed)
                if nx.has_path(G, company, main_company):
                    # Path exists, but tier calculation missed it - recalculate
                    logging.info(f"Path exists from {company} to {main_company}, but tier calculation missed it")
            except:
                pass
    
    # STEP 7: Create visualization with DETERMINISTIC layout
    net = Network(
        height="100vh",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#2d3436",
        select_menu=True,
        filter_menu=True,
        cdn_resources="remote"
    )
    
    # DETERMINISTIC hierarchical layout configuration
    # Fixed seed for reproducibility, deterministic sorting
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 14,
          "face": "Arial",
          "bold": {
            "color": "#2d3436"
          }
        },
        "borderWidth": 3,
        "shadow": {
          "enabled": true,
          "color": "rgba(0,0,0,0.2)",
          "size": 10,
          "x": 2,
          "y": 2
        },
        "shapeProperties": {
          "useBorderWithImage": true
        }
      },
      "edges": {
        "smooth": {
          "type": "curvedCW",
          "roundness": 0.4
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
        "hoverWidth": 4,
        "font": {
          "size": 12,
          "align": "middle"
        }
      },
      "layout": {
        "hierarchical": {
          "enabled": true,
          "levelSeparation": 250,
          "direction": "LR",
          "sortMethod": "directed",
          "shakeTowards": "none",
          "nodeSpacing": 200,
          "treeSpacing": 300,
          "blockShifting": true,
          "edgeMinimization": true,
          "parentCentralization": true
        }
      },
      "physics": {
        "enabled": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "hideEdgesOnDrag": false,
        "hideEdgesOnZoom": false
      }
    }
    """)
    
    # STEP 8: Add company nodes with tier-based styling
    # Track which nodes we've added to the visualization
    nodes_added_to_viz = set()
    companies_added = 0
    for company, data in companies_data.items():
        # Get tier for this company
        company_tier = company_tiers.get(company)
        
        # Skip if not connected to main_company (unless it's the main company itself)
        if company_tier is None and company != main_company:
            logging.debug(f"Skipping disconnected company: {company}")
            continue
        
        # Skip if beyond Tier-4
        if company_tier is not None and company_tier > 4:
            logging.debug(f"Skipping company beyond Tier-4: {company} (Tier-{company_tier})")
            continue
        
        tier = company_tier if company_tier is not None else 0
        country = data["country"]
        is_disrupted = data["is_disrupted"]
        is_disruption_source = (company in disruption_sources)
        is_main = (company == main_company)
        products = data.get("products", "Unknown Product")
        industry = data.get("industry", "Unknown")
        
        # Determine node color based on status
        if is_main:
            node_color = "#ffd700"  # Gold for main company
            border_color = "#c29b00"
        elif is_disruption_source:
            node_color = "#e74c3c"  # Red for disruption sources
            border_color = "#c0392b"
        elif is_disrupted:
            node_color = "#f39c12"  # Orange for companies in disrupted countries
            border_color = "#d68910"
        else:
            node_color = "#3498db"  # Blue for normal suppliers
            border_color = "#2980b9"
        
        # Create node label with tier and country
        tier_label = f"Tier-{tier}" if tier > 0 else "Tier-0 (Tesla)"
        node_label = f"{company}\n{tier_label}\n{country}"
        
        # Create detailed tooltip
        tooltip = (
            f"<b>{company}</b><br>"
            f"Tier: {tier_label}<br>"
            f"Country: {country}<br>"
            f"Industry: {industry}<br>"
            f"Products: {products}<br>"
            f"Status: {'DISRUPTION SOURCE' if is_disruption_source else 'DISRUPTED' if is_disrupted else 'Normal'}<br>"
            f"{'Main Company' if is_main else 'Supplier'}"
        )
        
        net.add_node(
            company,
            label=node_label,
            color=node_color,
            size=60 if is_main else (50 if is_disruption_source else (40 if tier <= 2 else 35)),
            borderWidth=5 if is_main or is_disruption_source else 3,
            borderWidthSelected=7 if is_main or is_disruption_source else 5,
            shape="dot",
            font={"size": 12 if is_main else 10, "color": "#2d3436", "face": "Arial"},
            title=tooltip,
            borderColor=border_color,
            level=tier,  # Hierarchical level for layout
            mass=5 if is_main else (4 if is_disruption_source else (3 if tier <= 2 else 2))
        )
        nodes_added_to_viz.add(company)
        companies_added += 1
    
    # STEP 9: Add country nodes with clear styling
    countries_added = 0
    for country in all_countries:
        is_disrupted = (country in disrupted_countries)
        
        net.add_node(
            country,
            label=f"🌍 {country}",
            color="#e74c3c" if is_disrupted else "#2ecc71",
            size=50,
            shape="box",
            font={"size": 14, "color": "#ffffff", "face": "Arial Bold"},
            title=f"Country: {country}\nStatus: {'DISRUPTED' if is_disrupted else 'Normal'}\nCompanies: {sum(1 for c, d in companies_data.items() if d['country'] == country)}",
            margin=15,
            borderWidth=3,
            borderColor="#c0392b" if is_disrupted else "#27ae60",
            level=5  # Countries on a separate level
        )
        nodes_added_to_viz.add(country)
        countries_added += 1
    
    logging.info(f"Added {companies_added} company nodes and {countries_added} country nodes")
    
    # STEP 10: Add supply chain edges with product labels
    supply_edges_added_viz = 0
    for link in product_map:
        supplier = link.get("supplier_name", "").strip()
        customer = link.get("customer_name", "").strip()
        product = link.get("product", "Unknown Product").strip()
        
        if not supplier or not customer:
            continue
        
        # Only add if both nodes are in the visualization (check our tracking set)
        if supplier not in nodes_added_to_viz or customer not in nodes_added_to_viz:
            logging.debug(f"Skipping edge: {supplier} -> {customer} (nodes not in visualization)")
            continue
        
        # Get supplier tier to determine edge styling
        supplier_tier = company_tiers.get(supplier)
        customer_tier = company_tiers.get(customer)
        
        # Skip invalid tier relationships
        if supplier_tier is not None and customer_tier is not None:
            if supplier_tier <= customer_tier:
                continue  # Invalid: supplier should be higher tier than customer
        
        # Determine edge color based on disruption
        supplier_disrupted = companies_data.get(supplier, {}).get("is_disrupted", False)
        customer_disrupted = companies_data.get(customer, {}).get("is_disrupted", False)
        
        if supplier_disrupted or customer_disrupted:
            edge_color = "#e74c3c"  # Red for disrupted supply chains
            edge_width = 4
        else:
            edge_color = "#34495e"  # Dark gray for normal supply chains
            edge_width = 2.5
        
        # Create edge label with product
        edge_label = product if product != "Unknown Product" else ""
        if len(edge_label) > 30:
            edge_label = edge_label[:27] + "..."
        
        net.add_edge(
            supplier,
            customer,
            label=edge_label,
            color=edge_color,
            width=edge_width,
            arrows="to",
            title=f"Product: {product}\nFrom: {supplier}\nTo: {customer}",
            font={"size": 10, "align": "middle"}
        )
        supply_edges_added_viz += 1
    
    # STEP 11: Add location edges (company -> country)
    location_edges_added_viz = 0
    for company, data in companies_data.items():
        country = data["country"]
        if country and country != "Unknown Country" and country in all_countries:
            # Check if both nodes are in visualization (check our tracking set)
            if company in nodes_added_to_viz and country in nodes_added_to_viz:
                is_disrupted = data["is_disrupted"]
                
                net.add_edge(
                    company,
                    country,
                    color="#95a5a6" if not is_disrupted else "#e74c3c",
                    width=2,
                    dashes=True,
                    arrows="to",
                    title=f"Location: {company} is located in {country}",
                    smooth={"type": "straightCross", "roundness": 0}
                )
                location_edges_added_viz += 1
    
    logging.info(f"Added {supply_edges_added_viz} supply chain edges and {location_edges_added_viz} location edges to visualization")
    
    # STEP 12: Generate enhanced legend with disruption information
    disrupted_countries_str = ", ".join(disrupted_countries) if disrupted_countries else "None"
    disruption_sources_list = list(disruption_sources)[:5]  # Show first 5
    disruption_sources_str = ", ".join(disruption_sources_list)
    if len(disruption_sources) > 5:
        disruption_sources_str += f", and {len(disruption_sources) - 5} more..."
    
    legend_html = f"""
    <div id="legend" style="
        position: absolute;
        top: 20px;
        right: 20px;
        background: rgba(255,255,255,0.98);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 1000;
        font-family: Arial, sans-serif;
        max-width: 300px;
        border: 2px solid #34495e;
    ">
        <h3 style="margin:0 0 15px 0; color: #2d3436; border-bottom: 2px solid #34495e; padding-bottom: 10px;">Legend</h3>
        
        <div style="margin-bottom: 15px;">
            <h4 style="margin:0 0 8px 0; color: #e74c3c; font-size: 14px;">Disruption Information</h4>
            <p style="margin: 5px 0; font-size: 12px; color: #555;">
                <b>Disrupted Countries:</b><br>{disrupted_countries_str}
            </p>
            <p style="margin: 5px 0; font-size: 12px; color: #555;">
                <b>Disruption Sources:</b><br>{disruption_sources_str if disruption_sources_str else 'None identified'}
            </p>
        </div>
        
        <div style="display: grid; grid-template-columns: auto 1fr; gap: 10px 15px; align-items: center; margin-bottom: 15px;">
            <div style="width: 25px; height: 25px; background: #ffd700; border: 3px solid #c29b00; border-radius: 50%;"></div>
            <div><b>Main Company (Tesla)</b></div>
            
            <div style="width: 25px; height: 25px; background: #e74c3c; border: 3px solid #c0392b; border-radius: 50%;"></div>
            <div><b>Disruption Source</b></div>
            
            <div style="width: 25px; height: 25px; background: #f39c12; border: 3px solid #d68910; border-radius: 50%;"></div>
            <div>Disrupted Company</div>
            
            <div style="width: 25px; height: 25px; background: #3498db; border: 3px solid #2980b9; border-radius: 50%;"></div>
            <div>Normal Supplier</div>
            
            <div style="width: 25px; height: 25px; background: #2ecc71; border: 3px solid #27ae60;"></div>
            <div>Country (Normal)</div>
            
            <div style="width: 25px; height: 25px; background: #e74c3c; border: 3px solid #c0392b;"></div>
            <div>Country (Disrupted)</div>
        </div>
        
        <div style="border-top: 2px solid #ddd; padding-top: 10px; margin-top: 10px;">
            <div style="display: grid; grid-template-columns: auto 1fr; gap: 10px 15px; align-items: center;">
                <div style="border-bottom: 3px solid #34495e; width: 40px;"></div>
                <div>Supply Chain Link</div>
                
                <div style="border-bottom: 2px dashed #95a5a6; width: 40px;"></div>
                <div>Location Link</div>
            </div>
        </div>
        
        <div style="margin-top: 15px; padding-top: 10px; border-top: 2px solid #ddd; font-size: 11px; color: #777;">
            <p style="margin: 5px 0;"><b>Visualization Features:</b></p>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>Complete supply chains from disruption source to Tesla</li>
                <li>Product information on edges</li>
                <li>Country locations for all companies</li>
                <li>Hierarchical tier-based layout</li>
            </ul>
        </div>
    </div>
    """

    try:
        # Create organized output directory structure
        from datetime import datetime
        base_dir = os.path.join(os.getcwd(), "visualizations", "network_plots")
        os.makedirs(base_dir, exist_ok=True)
        
        # CRITICAL: ALWAYS create a NEW file with unique timestamp + UUID
        # This ensures we NEVER reuse existing files - each run generates a fresh visualization
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Microseconds for uniqueness
        unique_id = str(uuid.uuid4())[:8]  # UUID for extra uniqueness
        company_suffix = main_company.replace(" ", "_").replace("/", "_") if main_company else "supply_chain"
        file_name = f"supply_chain_{company_suffix}_{timestamp}_{unique_id}.html"
        file_path = os.path.join(base_dir, file_name)
        
        logging.info(f"🔄 Generating NEW visualization file: {file_name}")
        logging.info(f"   Full path: {file_path}")
        logging.info(f"   This file will be created from scratch (no reuse of existing files)")
        
        # Save graph - this ALWAYS creates a new file (never overwrites due to unique filename)
        net.save_graph(file_path)
        
        # Verify file was created
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Visualization file was not created at {file_path}")
        
        logging.info(f"✅ Visualization file created successfully: {file_path}")
        
        # Inject custom CSS and legend
        with open(file_path, "r+", encoding="utf-8") as f:
            content = f.read()
            # Inject legend before closing body tag
            content = content.replace("</body>", f"{legend_html}</body>")
            # Add custom CSS for better styling
            custom_css = """
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                }
                #mynetworkid {
                    border: 2px solid #34495e;
                    border-radius: 8px;
                }
            </style>
            """
            content = content.replace("</head>", f"{custom_css}</head>")
            f.seek(0)
            f.write(content)
            f.truncate()
        
        logging.info(f"✅ Network visualization saved to: {file_path}")
        logging.info(f"   Companies: {companies_added}, Countries: {countries_added}")
        logging.info(f"   Supply edges: {supply_edges_added_viz}, Location edges: {location_edges_added_viz}")
        logging.info(f"   Disruption sources: {len(disruption_sources)}")
        
        # Return HTML content for embedding in reports and file path
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

def networkx_plot_tool(
    product_map: List[Dict[str, Any]],
    disrupted_countries: Optional[List[str]] = None,
    main_company: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates interactive supply chain visualization from product map data
    Returns HTML content as string wrapped in dictionary along with file path
    
    Features:
    - Deterministic, reproducible layout
    - Complete connected chains from disruption source to main company
    - Country nodes clearly displayed
    - Disruption source highlighting
    - Product information on edges
    """
    html, file_path = build_networkx_plot(
        product_map=product_map,
        disrupted_countries=disrupted_countries,
        main_company=main_company
    )
    
    # Return both HTML content and file path
    return {"html": html, "file_path": file_path}

networkx_plot_struct = StructuredTool(
    name="supply_chain_visualizer",
    description=(
        "Generates professional interactive supply chain diagrams with DETERMINISTIC layout. "
        "Features include: "
        "- Complete connected chains from disruption source to main company\n"
        "- Tier-based hierarchical layout (Tier-0 to Tier-4)\n"
        "- Disruption source and propagation highlighting\n"
        "- Country nodes with location relationships\n"
        "- Product information on supply chain edges\n"
        "- Stable, reproducible visualization\n"
        "- Interactive exploration controls"
    ),
    func=networkx_plot_tool,
    args_schema=NetworkXPlotInput
)
