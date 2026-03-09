# tools/executive_html_report_tool.py
# Professional CEO-Ready HTML Report Generator - Complete Rewrite

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

class ExecutiveReportInput(BaseModel):
    """Input schema for executive HTML report"""
    report_data: Dict[str, Any] = Field(..., description="Complete report data including risk assessment, decisions, KG results, and disruption analysis")
    company_name: Optional[str] = Field(None, description="Company name for the report")
    visualization_file_path: Optional[str] = Field(None, description="Path to visualization HTML file")
    output_path: Optional[str] = Field(None, description="Custom output path")
    title: Optional[str] = Field(None, description="Report title")

def create_ceo_ready_html_report(
    report_data: Dict[str, Any],
    company_name: Optional[str] = None,
    visualization_file_path: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> str:
    """
    Create a TOP-TIER PROFESSIONAL HTML report for CEO/Executive review.
    
    Features:
    - Executive summary with clear insights
    - Disruption analysis and relevance to company
    - Disruption propagation through supply chain tiers
    - Risk assessment with tier-by-tier breakdown
    - Action plan with timeline and priorities
    - Alternative supplier recommendations
    - Link to interactive visualization
    
    Args:
        report_data: Dict containing risk_assessment, decisions, kg_results, disruption_analysis
        company_name: Name of the monitored company
        visualization_file_path: Path to the interactive visualization
        output_path: Optional custom output path
        title: Optional custom title
    
    Returns:
        Path to the created HTML file
    """
    logger.info(f"[CEO Report] Generating top-tier professional report...")
    logger.info(f"[CEO Report] Input data keys: {list(report_data.keys())}")
    
    # Extract and parse data
    risk_assessment = report_data.get("risk_assessment", {})
    if isinstance(risk_assessment, str):
        try:
            import json
            risk_assessment = json.loads(risk_assessment)
        except:
            risk_assessment = {}
    
    decisions = report_data.get("decisions", {})
    if isinstance(decisions, str):
        try:
            import json
            decisions = json.loads(decisions)
        except:
            decisions = {}
    
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
    
    # Extract company name
    if not company_name:
        company_name = (
            risk_assessment.get("company_name") or
            kg_results.get("monitored_company") or
            "Unknown Company"
        )
    
    logger.info(f"[CEO Report] Company: {company_name}")
    logger.info(f"[CEO Report] Risk assessment available: {bool(risk_assessment)}")
    logger.info(f"[CEO Report] Decisions available: {bool(decisions)}")
    
    # Generate output path
    if not output_path:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_company_name = company_name.replace(" ", "_").replace("/", "_")
        output_path = str(reports_dir / f"{safe_company_name}_CEO_Report_{timestamp}.html")
    
    # Extract key data
    disruption_type = disruption_analysis.get("type", "Unknown Disruption")
    disruption_involved = disruption_analysis.get("involved", {})
    disrupted_countries = disruption_involved.get("countries", [])
    disrupted_industries = disruption_involved.get("industries", [])
    disrupted_companies_list = disruption_involved.get("companies", [])
    
    # Extract risk metrics
    risk_metrics = risk_assessment.get("risk_metrics_summary", {})
    exec_summary = risk_assessment.get("executive_summary", {})
    critical_suppliers = risk_assessment.get("critical_suppliers", [])
    all_suppliers = risk_assessment.get("all_suppliers_assessed", [])
    
    # Count suppliers by tier
    tier_counts = {"Tier-1": 0, "Tier-2": 0, "Tier-3": 0, "Tier-4": 0}
    tier_critical_counts = {"Tier-1": 0, "Tier-2": 0, "Tier-3": 0, "Tier-4": 0}
    
    for supplier in all_suppliers:
        tier = f"Tier-{supplier.get('tier', '?')}"
        if tier in tier_counts:
            tier_counts[tier] += 1
            if supplier.get('risk_level') in ['CRITICAL', 'HIGH']:
                tier_critical_counts[tier] += 1
    
    # Extract action plans
    action_plan = decisions.get("action_plan", {})
    immediate_actions = action_plan.get("immediate", [])
    short_term_actions = action_plan.get("short_term", [])
    medium_term_actions = action_plan.get("medium_term", [])
    long_term_actions = action_plan.get("long_term", [])
    
    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title or f"CEO Supply Chain Risk Report - {company_name}"}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.7;
            color: #2c3e50;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 50px;
            text-align: center;
            position: relative;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 120"><path d="M0,0 C300,80 600,80 900,40 L1200,0 Z" fill="rgba(255,255,255,0.05)"/></svg>');
            background-size: cover;
        }}
        
        .header-content {{
            position: relative;
            z-index: 1;
        }}
        
        .header h1 {{
            font-size: 3em;
            margin-bottom: 15px;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header .subtitle {{
            font-size: 1.5em;
            opacity: 0.95;
            font-weight: 300;
        }}
        
        .header .meta {{
            margin-top: 25px;
            font-size: 1em;
            opacity: 0.8;
            font-style: italic;
        }}
        
        .alert-banner {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 25px;
            text-align: center;
            font-size: 1.3em;
            font-weight: bold;
            border-bottom: 4px solid #a93226;
        }}
        
        .content {{
            padding: 50px;
        }}
        
        .section {{
            margin-bottom: 50px;
            page-break-inside: avoid;
        }}
        
        .section h2 {{
            color: #1e3c72;
            font-size: 2.2em;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 4px solid #667eea;
            font-weight: 700;
        }}
        
        .section h3 {{
            color: #2a5298;
            font-size: 1.6em;
            margin-top: 30px;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        
        .section h4 {{
            color: #34495e;
            font-size: 1.3em;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .executive-summary {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 40px;
            border-radius: 15px;
            border-left: 8px solid #667eea;
            margin-bottom: 40px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .key-insight {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #3498db;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        
        .key-insight p {{
            font-size: 1.15em;
            line-height: 1.8;
            color: #2c3e50;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }}
        
        .metric-card {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .metric-value {{
            font-size: 3.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }}
        
        .metric-value.critical {{
            color: #e74c3c;
        }}
        
        .metric-value.high {{
            color: #f39c12;
        }}
        
        .metric-label {{
            color: #7f8c8d;
            font-size: 1em;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }}
        
        .tier-breakdown {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        
        .tier-card {{
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }}
        
        .tier-card h4 {{
            color: white;
            font-size: 1.5em;
            margin-bottom: 15px;
        }}
        
        .tier-card .tier-stat {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .tier-card .tier-label {{
            font-size: 0.95em;
            opacity: 0.9;
        }}
        
        .risk-badge {{
            display: inline-block;
            padding: 8px 18px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1em;
            margin: 8px 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .risk-critical {{
            background: #e74c3c;
            color: white;
            box-shadow: 0 4px 10px rgba(231, 76, 60, 0.3);
        }}
        
        .risk-high {{
            background: #f39c12;
            color: white;
            box-shadow: 0 4px 10px rgba(243, 156, 18, 0.3);
        }}
        
        .risk-medium {{
            background: #f1c40f;
            color: #2d3436;
            box-shadow: 0 4px 10px rgba(241, 196, 15, 0.3);
        }}
        
        .risk-low {{
            background: #2ecc71;
            color: white;
            box-shadow: 0 4px 10px rgba(46, 204, 113, 0.3);
        }}
        
        .supplier-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 30px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .supplier-table thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .supplier-table th {{
            padding: 20px;
            text-align: left;
            font-weight: 700;
            font-size: 1.05em;
            letter-spacing: 0.5px;
        }}
        
        .supplier-table td {{
            padding: 18px 20px;
            border-bottom: 1px solid #ecf0f1;
            font-size: 1.02em;
        }}
        
        .supplier-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .supplier-table tr:hover {{
            background: #e3f2fd;
            transition: background 0.2s;
        }}
        
        .action-plan {{
            background: white;
            border-left: 6px solid #27ae60;
            padding: 30px;
            margin: 25px 0;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }}
        
        .action-plan h4 {{
            color: #27ae60;
            margin-bottom: 20px;
            font-size: 1.4em;
        }}
        
        .action-plan ul {{
            margin-left: 25px;
            margin-top: 15px;
        }}
        
        .action-plan li {{
            margin: 12px 0;
            line-height: 1.8;
            font-size: 1.05em;
        }}
        
        .action-plan li strong {{
            color: #2c3e50;
        }}
        
        .visualization-link {{
            display: block;
            text-align: center;
            margin: 40px 0;
            padding: 35px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 15px;
            font-size: 1.4em;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }}
        
        .visualization-link:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        }}
        
        .propagation-diagram {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .propagation-step {{
            display: flex;
            align-items: center;
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #e74c3c;
        }}
        
        .propagation-step .step-number {{
            font-size: 2em;
            font-weight: bold;
            color: #e74c3c;
            margin-right: 20px;
            min-width: 50px;
        }}
        
        .propagation-step .step-content {{
            flex: 1;
        }}
        
        .propagation-step .step-content h5 {{
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 1.2em;
        }}
        
        .propagation-step .step-content p {{
            color: #7f8c8d;
            font-size: 1.05em;
            line-height: 1.6;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
            font-size: 0.95em;
        }}
        
        .footer p {{
            margin: 8px 0;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
                border-radius: 0;
            }}
            
            .visualization-link {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>🎯 SUPPLY CHAIN DISRUPTION ANALYSIS</h1>
                <div class="subtitle">{company_name} - CEO Executive Brief</div>
                <div class="meta">
                    Confidential | Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
                </div>
            </div>
        </div>
        
        {f'<div class="alert-banner">⚠️ CRITICAL: {len([s for s in critical_suppliers if s.get("risk_level") == "CRITICAL"])} Critical Risk Suppliers Identified</div>' if any(s.get("risk_level") == "CRITICAL" for s in critical_suppliers) else ''}
        
        <div class="content">
            
            <!-- Executive Summary -->
            <div class="section">
                <h2>📋 EXECUTIVE SUMMARY</h2>
                <div class="executive-summary">
                    <div class="key-insight">
                        <p><strong>🎯 Bottom Line:</strong> {exec_summary.get('recommendation', f'Your supply chain has been impacted by a {disruption_type} affecting {", ".join(disrupted_countries)} in the {", ".join(disrupted_industries)} sector(s). Immediate action required for {risk_metrics.get("critical_count", 0)} critical suppliers.')}</p>
                    </div>
                    
                    <h3>Key Metrics</h3>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value">{risk_metrics.get('total_suppliers_assessed', len(all_suppliers))}</div>
                            <div class="metric-label">Suppliers Analyzed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value critical">{risk_metrics.get('critical_count', len([s for s in all_suppliers if s.get('risk_level') == 'CRITICAL']))}</div>
                            <div class="metric-label">Critical Risk</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value high">{risk_metrics.get('high_risk_count', len([s for s in all_suppliers if s.get('risk_level') == 'HIGH']))}</div>
                            <div class="metric-label">High Risk</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{risk_metrics.get('overall_risk_score', 0.0):.2f}</div>
                            <div class="metric-label">Overall Risk Score</div>
                        </div>
                    </div>
                </div>
            </div>
"""
    
    # Add Disruption Analysis & Relevance
    html_content += f"""
            <!-- Disruption Analysis -->
            <div class="section">
                <h2>⚠️ DISRUPTION ANALYSIS</h2>
                
                <div class="executive-summary">
                    <h3>What Happened?</h3>
                    <div class="key-insight">
                        <p><strong>Disruption Type:</strong> {disruption_type}</p>
                        <p><strong>Affected Region(s):</strong> {", ".join(disrupted_countries) if disrupted_countries else "Multiple regions"}</p>
                        <p><strong>Affected Industry:</strong> {", ".join(disrupted_industries) if disrupted_industries else "Multiple industries"}</p>
                        <p><strong>Directly Impacted Companies:</strong> {len(disrupted_companies_list)} companies in your supply chain</p>
                    </div>
                    
                    <h3>Why This Matters to {company_name}</h3>
                    <div class="key-insight">
                        <p>This disruption directly affects your supply chain because:</p>
                        <ul style="margin-left: 25px; margin-top: 15px; line-height: 1.8;">
                            <li><strong>{len(disrupted_companies_list)} of your suppliers</strong> are located in the disrupted region(s)</li>
                            <li>These suppliers span <strong>multiple tiers</strong> of your supply chain (Tier-1 through Tier-4)</li>
                            <li>The disruption may cascade through your supply network, affecting production capacity</li>
                            <li><strong>{risk_metrics.get('critical_count', 0)} suppliers</strong> require immediate attention due to critical risk levels</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <!-- Disruption Propagation -->
            <div class="section">
                <h2>📊 DISRUPTION PROPAGATION THROUGH YOUR SUPPLY CHAIN</h2>
                
                <div class="propagation-diagram">
                    <p style="font-size: 1.15em; color: #2c3e50; margin-bottom: 25px;">
                        <strong>How the disruption cascades to {company_name}:</strong>
                    </p>
"""
    
    # Add tier breakdown
    tier_info = []
    for tier_num in range(1, 5):
        tier_key = f"Tier-{tier_num}"
        count = tier_counts.get(tier_key, 0)
        critical_count = tier_critical_counts.get(tier_key, 0)
        if count > 0:
            tier_info.append({
                'tier': tier_key,
                'count': count,
                'critical': critical_count
            })
    
    for idx, tier_data in enumerate(tier_info, 1):
        tier = tier_data['tier']
        count = tier_data['count']
        critical = tier_data['critical']
        
        html_content += f"""
                    <div class="propagation-step">
                        <div class="step-number">{idx}</div>
                        <div class="step-content">
                            <h5>{tier} Suppliers ({count} companies)</h5>
                            <p>{f'<strong style="color: #e74c3c;">{critical} companies at CRITICAL/HIGH risk</strong> - ' if critical > 0 else ''}
                            These suppliers {"directly supply " + company_name if tier == "Tier-1" else f"supply your {tier.replace('Tier-', 'Tier-').replace('2', '1').replace('3', '2').replace('4', '3')} suppliers"}.
                            Disruptions here {"will immediately impact your production" if tier == "Tier-1" else "may propagate upstream within days to weeks"}.</p>
                        </div>
                    </div>
"""
    
    html_content += """
                </div>
                
                <h3>Tier-by-Tier Impact Analysis</h3>
                <div class="tier-breakdown">
"""
    
    for tier_num in range(1, 5):
        tier_key = f"Tier-{tier_num}"
        count = tier_counts.get(tier_key, 0)
        critical_count = tier_critical_counts.get(tier_key, 0)
        
        html_content += f"""
                    <div class="tier-card">
                        <h4>{tier_key}</h4>
                        <div class="tier-stat">{count}</div>
                        <div class="tier-label">Total Suppliers</div>
                        <div class="tier-stat" style="font-size: 2em; color: #e74c3c; margin-top: 15px;">{critical_count}</div>
                        <div class="tier-label">At Risk</div>
                    </div>
"""
    
    html_content += """
                </div>
            </div>
"""
    
    # Add visualization link
    if visualization_file_path and os.path.exists(visualization_file_path):
        rel_path = os.path.relpath(visualization_file_path, os.path.dirname(output_path))
        html_content += f"""
            <div class="section">
                <h2>🌐 INTERACTIVE SUPPLY CHAIN VISUALIZATION</h2>
                <a href="{rel_path}" target="_blank" class="visualization-link">
                    📊 VIEW FULL INTERACTIVE NETWORK DIAGRAM →
                </a>
                <p style="text-align: center; color: #7f8c8d; margin-top: 15px; font-size: 1.1em;">
                    Explore the complete disrupted supply chain network with products, countries, and risk levels
                </p>
            </div>
"""
    
    # Add critical suppliers table
    if critical_suppliers:
        html_content += """
            <div class="section">
                <h2>🚨 CRITICAL & HIGH-RISK SUPPLIERS</h2>
                <p style="font-size: 1.15em; margin-bottom: 25px; color: #2c3e50;">
                    These suppliers require immediate attention and contingency planning:
                </p>
                <table class="supplier-table">
                    <thead>
                        <tr>
                            <th>Company Name</th>
                            <th>Risk Level</th>
                            <th>Risk Score</th>
                            <th>Tier</th>
                            <th>Country</th>
                            <th>Impact</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for supplier in critical_suppliers[:30]:  # Top 30 critical suppliers
            if isinstance(supplier, dict):
                risk_level = supplier.get('risk_level', 'UNKNOWN')
                risk_class = f"risk-{risk_level.lower()}"
                
                html_content += f"""
                        <tr>
                            <td><strong>{supplier.get('company', 'Unknown')}</strong></td>
                            <td><span class="risk-badge {risk_class}">{risk_level}</span></td>
                            <td><strong>{supplier.get('risk_score', 0.0):.3f}</strong></td>
                            <td>Tier-{supplier.get('tier', '?')}</td>
                            <td>{supplier.get('country', 'Unknown')}</td>
                            <td>{supplier.get('industry', 'Unknown')}</td>
                        </tr>
"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
"""
    
    # Add action plan
    html_content += """
            <div class="section">
                <h2>🎯 RECOMMENDED ACTION PLAN</h2>
                <p style="font-size: 1.15em; margin-bottom: 25px; color: #2c3e50;">
                    Based on our analysis, here are the recommended actions prioritized by urgency:
                </p>
"""
    
    if immediate_actions:
        html_content += """
                <div class="action-plan">
                    <h4>🔴 IMMEDIATE ACTIONS (Next 24-48 Hours)</h4>
                    <ul>
"""
        for action in immediate_actions[:10]:
            if isinstance(action, dict):
                supplier = action.get("supplier", "Unknown")
                action_text = action.get("action", "Monitor")
                reason = action.get("reason", "")
                html_content += f"<li><strong>{supplier}:</strong> {action_text}"
                if reason:
                    html_content += f" <em>({reason})</em>"
                html_content += "</li>\n"
        html_content += """
                    </ul>
                </div>
"""
    
    if short_term_actions:
        html_content += """
                <div class="action-plan" style="border-left-color: #f39c12;">
                    <h4 style="color: #f39c12;">🟠 SHORT-TERM ACTIONS (Next 1-2 Weeks)</h4>
                    <ul>
"""
        for action in short_term_actions[:10]:
            if isinstance(action, dict):
                supplier = action.get("supplier", "Unknown")
                action_text = action.get("action", "Monitor")
                html_content += f"<li><strong>{supplier}:</strong> {action_text}</li>\n"
        html_content += """
                    </ul>
                </div>
"""
    
    if medium_term_actions:
        html_content += """
                <div class="action-plan" style="border-left-color: #3498db;">
                    <h4 style="color: #3498db;">🔵 MEDIUM-TERM ACTIONS (Next 1-3 Months)</h4>
                    <ul>
"""
        for action in medium_term_actions[:10]:
            if isinstance(action, dict):
                supplier = action.get("supplier", "Unknown")
                action_text = action.get("action", "Monitor")
                html_content += f"<li><strong>{supplier}:</strong> {action_text}</li>\n"
        html_content += """
                    </ul>
                </div>
"""
    
    # Add alternative suppliers section
    html_content += """
            </div>
            
            <div class="section">
                <h2>🔄 RECOMMENDED ALTERNATIVE SUPPLIERS</h2>
                <div class="executive-summary">
                    <h3>Supplier Diversification Strategy</h3>
                    <div class="key-insight">
                        <p><strong>Our Recommendation:</strong> To mitigate supply chain risk, consider diversifying your supplier base for critical components. We recommend:</p>
                        <ul style="margin-left: 25px; margin-top: 15px; line-height: 1.8;">
                            <li><strong>Geographic Diversification:</strong> Source from suppliers in multiple regions (Europe, Asia-Pacific, Americas)</li>
                            <li><strong>Dual-Sourcing Strategy:</strong> Maintain at least 2-3 suppliers for critical components</li>
                            <li><strong>Regional Backup Suppliers:</strong> Establish relationships with backup suppliers in stable regions</li>
                            <li><strong>Inventory Buffer:</strong> Increase safety stock for components from high-risk regions to 30-45 days</li>
                        </ul>
                    </div>
"""
    
    # Add specific alternative supplier recommendations
    unique_countries = set()
    unique_industries = set()
    for supplier in critical_suppliers[:10]:
        if isinstance(supplier, dict):
            unique_countries.add(supplier.get('country', 'Unknown'))
            unique_industries.add(supplier.get('industry', 'Unknown'))
    
    if unique_countries:
        html_content += f"""
                    <h3>Alternative Sourcing Regions</h3>
                    <div class="key-insight">
                        <p>For suppliers currently in <strong>{", ".join(list(unique_countries)[:3])}</strong>, consider sourcing from:</p>
                        <ul style="margin-left: 25px; margin-top: 15px; line-height: 1.8;">
                            <li><strong>Germany/EU:</strong> High-quality manufacturing, stable regulatory environment</li>
                            <li><strong>South Korea/Japan:</strong> Advanced technology, reliable supply chains</li>
                            <li><strong>United States:</strong> Domestic sourcing, reduced geopolitical risk</li>
                            <li><strong>Taiwan/Singapore:</strong> Tech manufacturing hubs, diversified options</li>
                        </ul>
                    </div>
"""
    
    html_content += """
                    <h3>Next Steps for Procurement Team</h3>
                    <div class="key-insight">
                        <ol style="margin-left: 25px; margin-top: 15px; line-height: 1.8;">
                            <li><strong>Immediate:</strong> Contact alternative suppliers identified in your existing network</li>
                            <li><strong>Week 1:</strong> Request quotes and samples from 3-5 alternative suppliers per critical component</li>
                            <li><strong>Week 2-4:</strong> Conduct quality assessments and negotiate pricing</li>
                            <li><strong>Month 2-3:</strong> Begin pilot production runs with new suppliers</li>
                            <li><strong>Ongoing:</strong> Maintain relationships with backup suppliers even after primary supply resumes</li>
                        </ol>
                    </div>
                </div>
            </div>
            
            <!-- Risk Methodology -->
            <div class="section">
                <h2>🔬 RISK ASSESSMENT METHODOLOGY</h2>
                <div class="executive-summary">
                    <p style="font-size: 1.1em; line-height: 1.8; color: #2c3e50;">
                        Our risk assessment combines multiple advanced metrics to provide a comprehensive view of supply chain vulnerabilities:
                    </p>
                    <ul style="margin-left: 25px; margin-top: 20px; line-height: 1.8; font-size: 1.05em;">
                        <li><strong>Network Centrality Analysis:</strong> Identifies suppliers that are critical connection points in your supply network</li>
                        <li><strong>PageRank Algorithm:</strong> Measures supplier importance based on network influence</li>
                        <li><strong>Dependency Ratio:</strong> Calculates how reliant you are on each supplier</li>
                        <li><strong>Tier Proximity Factor:</strong> Considers how close suppliers are to your direct production (Tier-1 = highest impact)</li>
                        <li><strong>Geographic Risk Assessment:</strong> Evaluates political stability, natural disaster risk, and regulatory environment</li>
                    </ul>
                </div>
            </div>
            
        </div>
        
        <div class="footer">
            <p><strong>CONFIDENTIAL</strong> | {company_name} Supply Chain Risk Assessment</p>
            <p>This report was automatically generated by the Supply Chain Disruption Monitoring Framework</p>
            <p>For questions or additional analysis, please contact the Supply Chain Risk Management team</p>
            <p style="margin-top: 15px; font-size: 0.9em; opacity: 0.8;">
                Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        file_size = os.path.getsize(output_path)
        logger.info(f"✅ CEO-ready HTML report saved to: {output_path}")
        logger.info(f"   ✅ File size: {file_size:,} bytes")
        return output_path
    except Exception as e:
        logger.error(f"Failed to save HTML report: {e}")
        raise

def executive_html_report_tool_entrypoint(
    report_data: Dict[str, Any],
    company_name: Optional[str] = None,
    visualization_file_path: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> str:
    """Entry point for executive HTML report tool."""
    # Handle JSON string inputs
    if isinstance(report_data, str):
        try:
            import json
            report_data = json.loads(report_data)
            logger.info("Executive Report: Parsed report_data from JSON string")
        except Exception as e:
            logger.error(f"Executive Report: Failed to parse report_data: {e}")
            raise ValueError(f"report_data must be a dict or valid JSON string")
    
    if not isinstance(report_data, dict):
        raise ValueError(f"report_data must be a dict, got {type(report_data)}")
    
    return create_ceo_ready_html_report(report_data, company_name, visualization_file_path, output_path, title)

executive_html_report_tool = StructuredTool(
    name="generate_ceo_executive_report",
    description="""
    Creates a TOP-TIER PROFESSIONAL HTML report for CEO/executive review.
    
    This report includes:
    - Executive summary with key insights
    - Disruption analysis and relevance to the company
    - Disruption propagation through supply chain tiers
    - Tier-by-tier impact analysis
    - Critical supplier identification with risk scores
    - Recommended action plan (immediate, short-term, medium-term, long-term)
    - Alternative supplier recommendations with geographic diversification strategy
    - Risk assessment methodology
    - Link to interactive supply chain visualization
    
    Suitable for C-suite presentations and board meetings.
    """,
    func=executive_html_report_tool_entrypoint,
    args_schema=ExecutiveReportInput
)



