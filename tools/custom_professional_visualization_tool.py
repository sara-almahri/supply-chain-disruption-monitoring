# tools/custom_professional_visualization_tool.py
# CUSTOM PROFESSIONAL SUPPLY CHAIN VISUALIZATION
# Built from scratch without PyVis - Pure HTML/JavaScript/D3.js

import os
import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

class CustomVisualizationInput(BaseModel):
    """Input schema for custom professional visualization"""
    product_map: List[Dict[str, Any]] = Field(..., description="Product map with supply chain links")
    disrupted_countries: Optional[List[str]] = Field(None, description="List of disrupted countries")
    main_company: Optional[str] = Field(None, description="Main company (Tier-0)")
    disruption_analysis: Optional[Dict[str, Any]] = Field(None, description="Disruption analysis data")

def build_custom_professional_visualization(
    product_map: List[Dict[str, Any]],
    disrupted_countries: Optional[List[str]] = None,
    main_company: Optional[str] = None,
    disruption_analysis: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    Build a custom, professional supply chain visualization from scratch.
    
    NO PyVis, NO filters - just pure professional visualization showing:
    - Full disrupted chain from Tier-0 (main company) to Tier-4
    - Products on every link
    - Countries for each company
    - Clear disruption propagation
    - Professional design suitable for CEO presentation
    """
    disrupted_countries = disrupted_countries or []
    main_company = main_company or "Tesla Inc"
    
    logger.info(f"[Custom Viz] Building professional visualization for {main_company}")
    logger.info(f"[Custom Viz] Product map links: {len(product_map)}")
    logger.info(f"[Custom Viz] Disrupted countries: {disrupted_countries}")
    
    if not product_map:
        logger.error("❌ [Custom Viz] Empty product map - cannot generate visualization")
        return "<html><body><h1>Error: No supply chain data</h1></body></html>", ""
    
    # Build network data structure
    nodes = {}
    edges = []
    
    # Process product map
    for link in product_map:
        supplier = link.get("supplier_name", "").strip()
        customer = link.get("customer_name", "").strip()
        supplier_country = link.get("supplier_country", "Unknown").strip()
        customer_country = link.get("customer_country", "Unknown").strip()
        product = link.get("product", "Unknown Product").strip()
        supplier_industry = link.get("supplier_industry", "Unknown").strip()
        customer_industry = link.get("customer_industry", "Unknown").strip()
        
        if not supplier or not customer:
            continue
        
        # Add nodes
        if supplier not in nodes:
            nodes[supplier] = {
                "id": supplier,
                "name": supplier,
                "country": supplier_country,
                "industry": supplier_industry,
                "is_disrupted": supplier_country in disrupted_countries,
                "is_main": supplier == main_company
            }
        
        if customer not in nodes:
            nodes[customer] = {
                "id": customer,
                "name": customer,
                "country": customer_country,
                "industry": customer_industry,
                "is_disrupted": customer_country in disrupted_countries,
                "is_main": customer == main_company
            }
        
        # Add edge
        edges.append({
            "source": supplier,
            "target": customer,
            "product": product,
            "is_disrupted": supplier_country in disrupted_countries or customer_country in disrupted_countries
        })
    
    logger.info(f"[Custom Viz] Nodes: {len(nodes)}, Edges: {len(edges)}")
    
    # Calculate tiers using BFS from main company
    tiers = {}
    if main_company in nodes:
        from collections import deque
        
        # Build adjacency list (reverse - from customers to suppliers)
        adj = {}
        for edge in edges:
            customer = edge["target"]
            supplier = edge["source"]
            if customer not in adj:
                adj[customer] = []
            adj[customer].append(supplier)
        
        # BFS from main company
        queue = deque([(main_company, 0)])
        visited = {main_company}
        tiers[main_company] = 0
        
        while queue:
            node, tier = queue.popleft()
            
            # Get suppliers of this node
            if node in adj and tier < 4:  # Limit to Tier-4
                for supplier in adj[node]:
                    if supplier not in visited:
                        visited.add(supplier)
                        tiers[supplier] = tier + 1
                        queue.append((supplier, tier + 1))
    
    # Add tier information to nodes
    for node_id, node_data in nodes.items():
        node_data["tier"] = tiers.get(node_id, 5)  # 5 = beyond scope
    
    # Filter to only nodes within Tier-4
    nodes_in_scope = {k: v for k, v in nodes.items() if v["tier"] <= 4}
    edges_in_scope = [e for e in edges if e["source"] in nodes_in_scope and e["target"] in nodes_in_scope]
    
    logger.info(f"[Custom Viz] Nodes in scope (Tier-0 to Tier-4): {len(nodes_in_scope)}")
    logger.info(f"[Custom Viz] Edges in scope: {len(edges_in_scope)}")
    
    # Convert to JSON
    nodes_json = json.dumps(list(nodes_in_scope.values()), indent=2)
    edges_json = json.dumps(edges_in_scope, indent=2)
    
    # Disruption info
    disruption_type = disruption_analysis.get("type", "Unknown") if disruption_analysis else "Unknown"
    disrupted_companies_count = len([n for n in nodes_in_scope.values() if n["is_disrupted"]])
    
    # Build HTML with custom visualization
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Supply Chain Disruption Visualization - {main_company}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
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
            overflow: hidden;
        }}
        
        .container {{
            max-width: 100%;
            height: 95vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 20px 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2em;
            margin-bottom: 8px;
            font-weight: 700;
        }}
        
        .header .info {{
            font-size: 1em;
            opacity: 0.9;
        }}
        
        .viz-container {{
            flex: 1;
            position: relative;
            background: #f8f9fa;
        }}
        
        #network {{
            width: 100%;
            height: 100%;
        }}
        
        .legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 1000;
            max-width: 280px;
        }}
        
        .legend h3 {{
            margin: 0 0 15px 0;
            color: #1e3c72;
            font-size: 1.2em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 10px 0;
            font-size: 0.95em;
        }}
        
        .legend-color {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            margin-right: 12px;
            border: 2px solid #34495e;
        }}
        
        .legend-color.main {{
            background: #FFD700;
            border-color: #DAA520;
        }}
        
        .legend-color.disrupted {{
            background: #DC143C;
            border-color: #8B0000;
        }}
        
        .legend-color.normal {{
            background: #4A90E2;
            border-color: #2E5C8A;
        }}
        
        .legend-line {{
            width: 40px;
            height: 3px;
            margin-right: 12px;
        }}
        
        .legend-line.disrupted-link {{
            background: #DC143C;
        }}
        
        .legend-line.normal-link {{
            background: #34495e;
        }}
        
        .node {{
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .node:hover {{
            filter: brightness(1.2);
        }}
        
        .node-label {{
            font-size: 12px;
            font-weight: 600;
            pointer-events: none;
            text-shadow: 1px 1px 2px white, -1px -1px 2px white, 1px -1px 2px white, -1px 1px 2px white;
        }}
        
        .link {{
            stroke-width: 3px;
            opacity: 0.8;
        }}
        
        .link-label {{
            font-size: 11px;
            fill: #2c3e50;
            font-weight: 600;
            pointer-events: none;
            text-shadow: 1px 1px 2px white, -1px -1px 2px white;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-size: 13px;
            pointer-events: none;
            z-index: 2000;
            max-width: 300px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }}
        
        .tooltip h4 {{
            margin: 0 0 8px 0;
            color: #FFD700;
            font-size: 15px;
        }}
        
        .tooltip p {{
            margin: 5px 0;
            line-height: 1.5;
        }}
        
        .controls {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 1000;
        }}
        
        .controls button {{
            padding: 10px 20px;
            margin: 5px;
            border: none;
            border-radius: 6px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }}
        
        .controls button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 Supply Chain Disruption Visualization</h1>
            <div class="info">
                <strong>{main_company}</strong> | 
                Disruption: {disruption_type} | 
                Affected: {", ".join(disrupted_countries) if disrupted_countries else "Multiple regions"} | 
                {disrupted_companies_count} Disrupted Suppliers
            </div>
        </div>
        
        <div class="viz-container">
            <svg id="network"></svg>
            
            <div class="legend">
                <h3>🎨 Legend</h3>
                <div class="legend-item">
                    <div class="legend-color main"></div>
                    <span><strong>{main_company}</strong> (Tier-0)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color disrupted"></div>
                    <span><strong>Disrupted Supplier</strong></span>
                </div>
                <div class="legend-item">
                    <div class="legend-color normal"></div>
                    <span>Normal Supplier</span>
                </div>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #ddd;">
                    <div class="legend-item">
                        <div class="legend-line disrupted-link"></div>
                        <span><strong>Disrupted Link</strong></span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line normal-link"></div>
                        <span>Normal Link</span>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <button onclick="resetZoom()">🔄 Reset View</button>
                <button onclick="fitToScreen()">📐 Fit to Screen</button>
            </div>
        </div>
    </div>
    
    <div class="tooltip" id="tooltip" style="display: none;"></div>
    
    <script>
        // Data
        const nodes = {nodes_json};
        const links = {edges_json};
        
        console.log("Nodes:", nodes.length);
        console.log("Links:", links.length);
        
        // Setup SVG
        const container = document.getElementById('network');
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        
        const svg = d3.select("#network")
            .attr("width", containerWidth)
            .attr("height", containerHeight);
        
        // Create groups for zoom
        const g = svg.append("g");
        
        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        // Tooltip
        const tooltip = d3.select("#tooltip");
        
        // Color mapping
        function getNodeColor(d) {{
            if (d.is_main) return "#FFD700";  // Gold for main company
            if (d.is_disrupted) return "#DC143C";  // Red for disrupted
            return "#4A90E2";  // Blue for normal
        }}
        
        function getNodeStroke(d) {{
            if (d.is_main) return "#DAA520";
            if (d.is_disrupted) return "#8B0000";
            return "#2E5C8A";
        }}
        
        function getLinkColor(d) {{
            return d.is_disrupted ? "#DC143C" : "#34495e";
        }}
        
        // Calculate node size based on tier
        function getNodeSize(d) {{
            if (d.is_main) return 25;
            if (d.tier === 1) return 18;
            if (d.tier === 2) return 15;
            if (d.tier === 3) return 13;
            return 11;
        }}
        
        // Force simulation
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links)
                .id(d => d.id)
                .distance(150))
            .force("charge", d3.forceManyBody().strength(-800))
            .force("center", d3.forceCenter(containerWidth / 2, containerHeight / 2))
            .force("collision", d3.forceCollide().radius(d => getNodeSize(d) + 30))
            .force("x", d3.forceX(d => {{
                // Position by tier (left to right)
                const tierX = [0.1, 0.3, 0.5, 0.7, 0.9];
                return containerWidth * tierX[Math.min(d.tier, 4)];
            }}).strength(0.3))
            .force("y", d3.forceY(containerHeight / 2).strength(0.1));
        
        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke", getLinkColor)
            .attr("stroke-width", d => d.is_disrupted ? 4 : 2)
            .attr("marker-end", "url(#arrowhead)");
        
        // Arrow markers
        svg.append("defs").selectAll("marker")
            .data(["arrowhead"])
            .enter().append("marker")
            .attr("id", "arrowhead")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#34495e");
        
        // Create link labels (products)
        const linkLabel = g.append("g")
            .selectAll("text")
            .data(links)
            .enter().append("text")
            .attr("class", "link-label")
            .attr("text-anchor", "middle")
            .text(d => {{
                const product = d.product || "Unknown";
                return product.length > 20 ? product.substring(0, 20) + "..." : product;
            }});
        
        // Create nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", getNodeSize)
            .attr("fill", getNodeColor)
            .attr("stroke", getNodeStroke)
            .attr("stroke-width", 3)
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended))
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip);
        
        // Create node labels
        const nodeLabel = g.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .attr("text-anchor", "middle")
            .attr("dy", d => getNodeSize(d) + 15)
            .text(d => {{
                const name = d.name || d.id;
                // Show company name + tier + country
                const tier = d.tier === 0 ? "T0" : `T${{d.tier}}`;
                const country = (d.country || "").substring(0, 3).toUpperCase();
                return `${{name.length > 20 ? name.substring(0, 20) + "..." : name}} [${{tier}}] 🌍${{country}}`;
            }});
        
        // Tooltip functions
        function showTooltip(event, d) {{
            tooltip.style("display", "block")
                .html(`
                    <h4>${{d.name || d.id}}</h4>
                    <p><strong>Tier:</strong> ${{d.tier === 0 ? "0 (Main Company)" : d.tier}}</p>
                    <p><strong>Country:</strong> ${{d.country || "Unknown"}} ${{d.is_disrupted ? "🔴 DISRUPTED" : "🟢"}}</p>
                    <p><strong>Industry:</strong> ${{d.industry || "Unknown"}}</p>
                    <p><strong>Status:</strong> ${{d.is_main ? "Main Company" : (d.is_disrupted ? "⚠️ In Disrupted Region" : "Normal")}}</p>
                `)
                .style("left", (event.pageX + 15) + "px")
                .style("top", (event.pageY - 15) + "px");
        }}
        
        function hideTooltip() {{
            tooltip.style("display", "none");
        }}
        
        // Simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            linkLabel
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            nodeLabel
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Control functions
        function resetZoom() {{
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity
            );
        }}
        
        function fitToScreen() {{
            const bounds = g.node().getBBox();
            const fullWidth = containerWidth;
            const fullHeight = containerHeight;
            const width = bounds.width;
            const height = bounds.height;
            const midX = bounds.x + width / 2;
            const midY = bounds.y + height / 2;
            
            const scale = 0.8 / Math.max(width / fullWidth, height / fullHeight);
            const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];
            
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
            );
        }}
        
        // Initial fit to screen after simulation stabilizes
        setTimeout(() => {{
            fitToScreen();
        }}, 2000);
        
        console.log("Visualization loaded successfully!");
    </script>
</body>
</html>
"""
    
    # Save to file
    try:
        base_dir = os.path.join(os.getcwd(), "visualizations", "network_plots")
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        company_suffix = main_company.replace(" ", "_").replace("/", "_")
        file_name = f"Professional_{company_suffix}_{timestamp}_{unique_id}.html"
        file_path = os.path.join(base_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"✅ [Custom Viz] Saved to: {file_path}")
        logger.info(f"   ✅ File size: {file_size:,} bytes")
        logger.info(f"   ✅ Nodes: {len(nodes_in_scope)}, Edges: {len(edges_in_scope)}")
        
        return html_content, file_path
        
    except Exception as e:
        logger.error(f"❌ [Custom Viz] Failed to save: {e}")
        import traceback
        traceback.print_exc()
        return html_content, ""

def custom_professional_visualization_tool_func(
    product_map: List[Dict[str, Any]],
    disrupted_countries: Optional[List[str]] = None,
    main_company: Optional[str] = None,
    disruption_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates a custom, professional supply chain visualization.
    
    Features:
    - Full disrupted chain from Tier-0 to Tier-4
    - Products on every link
    - Countries for each company
    - Clean, professional design
    - NO unnecessary filters
    - Interactive D3.js visualization
    """
    html, file_path = build_custom_professional_visualization(
        product_map=product_map,
        disrupted_countries=disrupted_countries,
        main_company=main_company,
        disruption_analysis=disruption_analysis
    )
    
    return {"html": html, "file_path": file_path}

custom_professional_visualization_tool = StructuredTool(
    name="custom_professional_supply_chain_visualizer",
    description="""
    Generates a custom, professional supply chain disruption visualization.
    
    Features:
    - Full disrupted chain from Tier-0 (main company) to Tier-4
    - Products displayed on every supply chain link
    - Countries shown for each company
    - Clean, professional design suitable for CEO presentations
    - NO unnecessary filters - just pure visualization
    - Interactive D3.js-based network diagram
    - Zoom, pan, drag nodes
    - Hover for detailed information
    """,
    func=custom_professional_visualization_tool_func,
    args_schema=CustomVisualizationInput
)



