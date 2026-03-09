# tools/html_report_tool.py
# Professional HTML Report Generator for CEO/Executive Review

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

class HTMLReportInput(BaseModel):
    """Input schema for HTML report export"""
    report_data: Dict[str, Any] = Field(..., description="Report data to export. Should contain risk_assessment, decisions, kg_results, and disruption_analysis.")
    company_name: Optional[str] = Field(None, description="Company name for the report. If not provided, will be extracted from report_data.")
    visualization_file_path: Optional[str] = Field(None, description="Path to the visualization HTML file to link in the report.")
    output_path: Optional[str] = Field(None, description="Output file path (default: reports/{company}_report_{timestamp}.html)")
    title: Optional[str] = Field(None, description="Report title")

def create_executive_html_report(
    report_data: Dict[str, Any],
    company_name: Optional[str] = None,
    visualization_file_path: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> str:
    """
    Create a professional HTML report for executive/CEO review.
    
    Args:
        report_data: Dictionary containing risk assessment, decisions, etc.
        company_name: Name of the company (optional, will be extracted from report_data if not provided)
        visualization_file_path: Path to visualization HTML file to link
        output_path: Optional custom output path
        title: Optional custom title
    
    Returns:
        Path to the created HTML file
    """
    # CRITICAL: Log what data we received
    logger.info(f"[HTML Report] Generating report with report_data keys: {list(report_data.keys())}")
    logger.info(f"[HTML Report] Company name provided: {company_name}")
    
    # Extract company_name from report_data if not provided
    if not company_name:
        risk_assessment = report_data.get("risk_assessment", {})
        if isinstance(risk_assessment, str):
            try:
                import json
                risk_assessment = json.loads(risk_assessment)
            except:
                risk_assessment = {}
        
        if isinstance(risk_assessment, dict):
            company_name = risk_assessment.get("company_name")
        
        if not company_name:
            kg_results = report_data.get("kg_results", {})
            if isinstance(kg_results, dict):
                company_name = kg_results.get("monitored_company")
        
        if not company_name:
            company_name = "Unknown Company"
            logger.warning("Company name not provided and not found in report_data, using default: 'Unknown Company'")
    
    logger.info(f"[HTML Report] Final company name: {company_name}")
    
    # Generate output path if not provided
    if not output_path:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_company_name = company_name.replace(" ", "_").replace("/", "_")
        output_path = str(reports_dir / f"{safe_company_name}_ExecutiveReport_{timestamp}.html")
    
    # Extract data with robust parsing
    risk_assessment = report_data.get("risk_assessment", {})
    if isinstance(risk_assessment, str):
        try:
            import json
            risk_assessment = json.loads(risk_assessment)
        except:
            risk_assessment = {}
    logger.info(f"[HTML Report] risk_assessment type: {type(risk_assessment)}, keys: {list(risk_assessment.keys()) if isinstance(risk_assessment, dict) else 'N/A'}")
    
    decisions = report_data.get("decisions", {})
    if isinstance(decisions, str):
        try:
            import json
            decisions = json.loads(decisions)
        except:
            decisions = {}
    logger.info(f"[HTML Report] decisions type: {type(decisions)}, keys: {list(decisions.keys()) if isinstance(decisions, dict) else 'N/A'}")
    
    kg_results = report_data.get("kg_results", {})
    if isinstance(kg_results, str):
        try:
            import json
            kg_results = json.loads(kg_results)
        except:
            kg_results = {}
    
    disruption_analysis = report_data.get("disruption_analysis", {})
    if isinstance(disruption_analysis, str):
        try:
            import json
            disruption_analysis = json.loads(disruption_analysis)
        except:
            disruption_analysis = {}
    
    logger.info(f"[HTML Report] Data extraction complete - risk_assessment items: {len(risk_assessment) if isinstance(risk_assessment, dict) else 0}")
    
    # Build HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title or f"Supply Chain Disruption Report - {company_name}"}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #2d3436;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .header .meta {{
            margin-top: 20px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .section h3 {{
            color: #764ba2;
            font-size: 1.4em;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        
        .executive-summary {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 30px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            margin-bottom: 30px;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .risk-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
            margin: 5px;
        }}
        
        .risk-critical {{
            background: #e74c3c;
            color: white;
        }}
        
        .risk-high {{
            background: #f39c12;
            color: white;
        }}
        
        .risk-medium {{
            background: #f1c40f;
            color: #2d3436;
        }}
        
        .risk-low {{
            background: #2ecc71;
            color: white;
        }}
        
        .supplier-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .supplier-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .supplier-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .supplier-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .action-plan {{
            background: #fff;
            border-left: 4px solid #2ecc71;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .action-plan h4 {{
            color: #27ae60;
            margin-bottom: 10px;
        }}
        
        .action-plan ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}
        
        .action-plan li {{
            margin: 8px 0;
        }}
        
        .visualization-link {{
            display: block;
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: bold;
            transition: transform 0.3s;
        }}
        
        .visualization-link:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }}
        
        .footer {{
            background: #2d3436;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 0.9em;
        }}
        
        .expandable {{
            cursor: pointer;
            user-select: none;
        }}
        
        .expandable:hover {{
            background: #f8f9fa;
        }}
        
        .collapsible-content {{
            display: none;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
            margin-top: 10px;
        }}
        
        .collapsible-content.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Supply Chain Disruption Risk Assessment</h1>
            <div class="subtitle">{company_name}</div>
            <div class="meta">
                Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </div>
        </div>
        
        <div class="content">
            <!-- Executive Summary -->
            <div class="section">
                <h2>📋 Executive Summary</h2>
                <div class="executive-summary">
"""
    
    # Add executive summary content
    if isinstance(risk_assessment, dict):
        exec_summary = risk_assessment.get("executive_summary", {})
        risk_metrics = risk_assessment.get("risk_metrics_summary", {})
        
        html_content += f"""
                    <h3>Key Findings</h3>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{risk_metrics.get('total_suppliers_assessed', 0)}</div>
                            <div class="metric-label">Suppliers Assessed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{risk_metrics.get('critical_count', 0)}</div>
                            <div class="metric-label">Critical Risk</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{risk_metrics.get('high_risk_count', 0)}</div>
                            <div class="metric-label">High Risk</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{risk_metrics.get('overall_risk_score', 0.0):.2f}</div>
                            <div class="metric-label">Overall Risk Score</div>
                        </div>
                    </div>
                    <p style="margin-top: 20px; font-size: 1.1em;">
                        <strong>Recommendation:</strong> {exec_summary.get('recommendation', 'Continue monitoring supply chain risks.')}
                    </p>
"""
    
    html_content += """
                </div>
            </div>
"""
    
    # Add visualization link
    if visualization_file_path and os.path.exists(visualization_file_path):
        # Convert to relative path for HTML
        rel_path = os.path.relpath(visualization_file_path, os.path.dirname(output_path))
        html_content += f"""
            <div class="section">
                <h2>🌐 Interactive Supply Chain Visualization</h2>
                <a href="{rel_path}" target="_blank" class="visualization-link">
                    📊 View Full Interactive Supply Chain Visualization →
                </a>
                <p style="text-align: center; color: #7f8c8d; margin-top: 10px;">
                    Click above to explore the complete disrupted supply chain network with interactive features
                </p>
            </div>
"""
    
    # Add disruption analysis
    if disruption_analysis:
        html_content += """
            <div class="section">
                <h2>⚠️ Disruption Analysis</h2>
"""
        if isinstance(disruption_analysis, dict):
            disruption_type = disruption_analysis.get("type", "Unknown")
            involved = disruption_analysis.get("involved", {})
            countries = involved.get("countries", [])
            industries = involved.get("industries", [])
            
            html_content += f"""
                <div class="executive-summary">
                    <h3>Disruption Type: {disruption_type}</h3>
                    <p><strong>Affected Countries:</strong> {', '.join(countries) if countries else 'None identified'}</p>
                    <p><strong>Affected Industries:</strong> {', '.join(industries) if industries else 'None identified'}</p>
                </div>
"""
        html_content += """
            </div>
"""
    
    # Add risk assessment details
    if isinstance(risk_assessment, dict):
        html_content += """
            <div class="section">
                <h2>📊 Detailed Risk Assessment</h2>
"""
        critical_suppliers = risk_assessment.get("critical_suppliers", [])
        if critical_suppliers:
            html_content += """
                <h3>Critical & High-Risk Suppliers</h3>
                <table class="supplier-table">
                    <thead>
                        <tr>
                            <th>Supplier</th>
                            <th>Risk Score</th>
                            <th>Risk Level</th>
                            <th>Tier</th>
                            <th>Country</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            for supplier in critical_suppliers[:20]:  # Limit to top 20
                if isinstance(supplier, dict):
                    html_content += f"""
                        <tr>
                            <td><strong>{supplier.get('company', 'Unknown')}</strong></td>
                            <td>{supplier.get('risk_score', 0.0):.3f}</td>
                            <td><span class="risk-badge risk-{supplier.get('risk_level', 'LOW').lower()}">{supplier.get('risk_level', 'LOW')}</span></td>
                            <td>Tier-{supplier.get('tier', '?')}</td>
                            <td>{supplier.get('country', 'Unknown')}</td>
                        </tr>
"""
            html_content += """
                    </tbody>
                </table>
"""
        html_content += """
            </div>
"""
    
    # Add decisions and action plan
    if isinstance(decisions, dict):
        decision_items = decisions.get("decisions", {})
        action_plan = decisions.get("action_plan", {})
        
        if decision_items or action_plan:
            html_content += """
            <div class="section">
                <h2>🎯 Strategic Decisions & Action Plan</h2>
"""
            if action_plan:
                for timeline, actions in action_plan.items():
                    if actions:
                        timeline_label = timeline.replace("_", " ").title()
                        html_content += f"""
                <div class="action-plan">
                    <h4>{timeline_label} Actions</h4>
                    <ul>
"""
                        for action in actions[:10]:  # Limit to 10 per timeline
                            if isinstance(action, dict):
                                supplier = action.get("supplier", "Unknown")
                                action_type = action.get("action", "Monitor")
                                html_content += f"<li><strong>{supplier}:</strong> {action_type}</li>"
                        html_content += """
                    </ul>
                </div>
"""
            html_content += """
            </div>
"""
    
    # Add methodology
    if isinstance(risk_assessment, dict):
        methodology = risk_assessment.get("methodology", {})
        if methodology:
            html_content += """
            <div class="section">
                <h2>🔬 Risk Assessment Methodology</h2>
                <div class="executive-summary">
"""
            metrics_used = methodology.get("metrics_used", [])
            if metrics_used:
                html_content += "<h3>Metrics Used:</h3><ul>"
                for metric in metrics_used:
                    html_content += f"<li>{metric}</li>"
                html_content += "</ul>"
            
            risk_formula = methodology.get("risk_formula", "")
            if risk_formula:
                html_content += f'<p style="margin-top: 15px;"><strong>Risk Formula:</strong> <code>{risk_formula}</code></p>'
            
            html_content += """
                </div>
            </div>
"""
    
    # Close HTML
    html_content += """
        </div>
        
        <div class="footer">
            <p>This report was automatically generated by the Supply Chain Disruption Monitoring Framework</p>
            <p>For questions or additional analysis, please contact the Supply Chain Risk Management team</p>
        </div>
    </div>
    
    <script>
        // Add collapsible functionality
        document.querySelectorAll('.expandable').forEach(item => {
            item.addEventListener('click', function() {
                const content = this.nextElementSibling;
                content.classList.toggle('active');
            });
        });
    </script>
</body>
</html>
"""
    
    # Write HTML file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Executive HTML report saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save HTML report: {e}")
        raise

def html_report_tool_entrypoint(
    report_data: Dict[str, Any],
    company_name: Optional[str] = None,
    visualization_file_path: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> str:
    """
    Entry point for HTML report export tool.
    
    CRITICAL: Handles both dict and JSON string inputs (LLM may pass strings).
    """
    # Handle case where LLM passes report_data as a JSON string
    if isinstance(report_data, str):
        try:
            import json
            report_data = json.loads(report_data)
            logger.info("HTML Report: Parsed report_data from JSON string")
        except Exception as e:
            logger.error(f"HTML Report: Failed to parse report_data as JSON: {e}")
            raise ValueError(f"report_data must be a dict or valid JSON string. Got: {type(report_data)}")
    
    # Ensure report_data is a dict
    if not isinstance(report_data, dict):
        logger.error(f"HTML Report: report_data must be a dict, got {type(report_data)}")
        raise ValueError(f"report_data must be a dict, got {type(report_data)}")
    
    return create_executive_html_report(report_data, company_name, visualization_file_path, output_path, title)

html_report_tool = StructuredTool(
    name="export_executive_html_report",
    description="""
    Creates a professional HTML report for executive/CEO review.
    Includes risk assessment, critical suppliers, decisions, action plans, and a link to the visualization.
    Suitable for C-suite presentations and board meetings.
    
    **IMPORTANT:** The `company_name` parameter is optional. If not provided, it will be automatically extracted from `report_data['risk_assessment']['company_name']` or `report_data['kg_results']['monitored_company']`.
    
    **Usage:**
    - You can call this tool with just `report_data` - the company name will be extracted automatically
    - Provide `visualization_file_path` to include a link to the interactive visualization
    - Or provide `company_name` explicitly if you have it
    """,
    func=html_report_tool_entrypoint,
    args_schema=HTMLReportInput
)

