# pdf_export_tool.py
# Professional PDF export for executive reports

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logging.warning("reportlab not available. Install with: pip install reportlab")

logger = logging.getLogger(__name__)

class PDFExportInput(BaseModel):
    """Input schema for PDF export"""
    report_data: Dict[str, Any] = Field(..., description="Report data to export. Should contain risk_assessment, decisions, kg_results, and disruption_analysis. If company_name is not provided, it will be extracted from report_data['risk_assessment']['company_name'] if available.")
    company_name: Optional[str] = Field(None, description="Company name for the report. If not provided, will be extracted from report_data['risk_assessment']['company_name'] or defaults to 'Unknown Company'")
    output_path: Optional[str] = Field(None, description="Output file path (default: reports/{company}_report_{timestamp}.pdf)")
    title: Optional[str] = Field(None, description="Report title")

def create_executive_pdf(
    report_data: Dict[str, Any],
    company_name: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> str:
    """
    Create a professional PDF report for executive/CEO review.
    
    Args:
        report_data: Dictionary containing risk assessment, decisions, etc.
        company_name: Name of the company (optional, will be extracted from report_data if not provided)
        output_path: Optional custom output path
        title: Optional custom title
    
    Returns:
        Path to the created PDF file
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")
    
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
        
        # If still not found, try to get from kg_results
        if not company_name:
            kg_results = report_data.get("kg_results", {})
            if isinstance(kg_results, dict):
                company_name = kg_results.get("monitored_company")
        
        # Default fallback
        if not company_name:
            company_name = "Unknown Company"
            logger.warning("Company name not provided and not found in report_data, using default: 'Unknown Company'")
    
    # Generate output path if not provided
    if not output_path:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_company_name = company_name.replace(" ", "_").replace("/", "_")
        output_path = str(reports_dir / f"{safe_company_name}_SupplyChainRiskReport_{timestamp}.pdf")
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    # Title page
    report_title = title or f"Supply Chain Disruption Risk Assessment"
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph(f"<b>Company:</b> {company_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Paragraph(f"<b>Time:</b> {datetime.now().strftime('%I:%M %p')}", styles['Normal']))
    
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    # Log what data we received for debugging
    logger.info(f"PDF Export - Received report_data keys: {list(report_data.keys())}")
    
    risk_assessment = report_data.get("risk_assessment", {})
    if not risk_assessment:
        logger.warning("PDF Export: No risk_assessment in report_data")
        # Try to get it from top level
        risk_assessment = report_data
    
    if isinstance(risk_assessment, str):
        try:
            import json
            risk_assessment = json.loads(risk_assessment)
        except:
            risk_assessment = {}
    
    logger.info(f"PDF Export - risk_assessment type: {type(risk_assessment)}, keys: {list(risk_assessment.keys()) if isinstance(risk_assessment, dict) else 'N/A'}")
    
    risk_summary = risk_assessment.get("risk_metrics_summary", {}) if isinstance(risk_assessment, dict) else {}
    
    # Get disruption analysis for additional context
    disruption_analysis = report_data.get("disruption_analysis", {})
    if isinstance(disruption_analysis, str):
        try:
            import json
            disruption_analysis = json.loads(disruption_analysis)
        except:
            disruption_analysis = {}
    
    # Get kg_results
    kg_results = report_data.get("kg_results", {})
    if isinstance(kg_results, str):
        try:
            import json
            kg_results = json.loads(kg_results)
        except:
            kg_results = {}
    
    logger.info(f"PDF Export - Data availability:")
    logger.info(f"  - risk_assessment: {'Yes' if risk_assessment else 'No'}")
    logger.info(f"  - disruption_analysis: {'Yes' if disruption_analysis else 'No'}")
    logger.info(f"  - kg_results: {'Yes' if kg_results else 'No'}")
    logger.info(f"  - decisions: {'Yes' if report_data.get('decisions') else 'No'}")
    
    summary_text = f"""
    This report provides a comprehensive assessment of supply chain disruption risks for {company_name}.
    The analysis covers up to Tier-4 suppliers and identifies critical risk factors that may impact operations.
    """
    
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Key Metrics Table
    # Get supplier_risk_scores to calculate metrics if risk_summary is empty
    supplier_risk_scores = risk_assessment.get("supplier_risk_scores", {}) if isinstance(risk_assessment, dict) else {}
    
    if risk_summary:
        total_assessed = risk_summary.get("total_suppliers_assessed", len(supplier_risk_scores) if supplier_risk_scores else 0)
        critical_count = risk_summary.get("critical_count", len([s for s in supplier_risk_scores.values() if s >= 0.8]) if supplier_risk_scores else 0)
        high_count = risk_summary.get("high_risk_count", len([s for s in supplier_risk_scores.values() if 0.6 <= s < 0.8]) if supplier_risk_scores else 0)
        medium_count = risk_summary.get("medium_risk_count", len([s for s in supplier_risk_scores.values() if 0.4 <= s < 0.6]) if supplier_risk_scores else 0)
        low_count = risk_summary.get("low_risk_count", len([s for s in supplier_risk_scores.values() if s < 0.4]) if supplier_risk_scores else 0)
        overall_risk = risk_summary.get("overall_risk_score", max(supplier_risk_scores.values()) if supplier_risk_scores else 0.0)
        avg_risk = risk_summary.get("average_risk_score", sum(supplier_risk_scores.values()) / len(supplier_risk_scores) if supplier_risk_scores else 0.0)
        
        metrics_data = [
            ["Metric", "Value"],
            ["Total Suppliers Assessed", str(total_assessed)],
            ["Critical Risk Suppliers", str(critical_count)],
            ["High Risk Suppliers", str(high_count)],
            ["Medium Risk Suppliers", str(medium_count)],
            ["Low Risk Suppliers", str(low_count)],
            ["Overall Risk Score", f"{overall_risk:.4f}"],
            ["Average Risk Score", f"{avg_risk:.4f}"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[4*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
    elif supplier_risk_scores:
        # Fallback: Create metrics table from supplier_risk_scores
        total_assessed = len(supplier_risk_scores)
        critical_count = len([s for s in supplier_risk_scores.values() if s >= 0.8])
        high_count = len([s for s in supplier_risk_scores.values() if 0.6 <= s < 0.8])
        medium_count = len([s for s in supplier_risk_scores.values() if 0.4 <= s < 0.6])
        low_count = len([s for s in supplier_risk_scores.values() if s < 0.4])
        overall_risk = max(supplier_risk_scores.values()) if supplier_risk_scores else 0.0
        avg_risk = sum(supplier_risk_scores.values()) / len(supplier_risk_scores) if supplier_risk_scores else 0.0
        
        metrics_data = [
            ["Metric", "Value"],
            ["Total Suppliers Assessed", str(total_assessed)],
            ["Critical Risk Suppliers", str(critical_count)],
            ["High Risk Suppliers", str(high_count)],
            ["Medium Risk Suppliers", str(medium_count)],
            ["Low Risk Suppliers", str(low_count)],
            ["Overall Risk Score", f"{overall_risk:.4f}"],
            ["Average Risk Score", f"{avg_risk:.4f}"]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[4*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.3*inch))
    else:
        # No data available
        story.append(Paragraph("<b>Warning:</b> No risk assessment data available.", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Disruption Summary
    disruption_summary = risk_assessment.get("disruption_summary", {}) if isinstance(risk_assessment, dict) else {}
    if not disruption_summary and isinstance(disruption_analysis, dict):
        # Extract from disruption_analysis if not in risk_assessment
        disruption_type = disruption_analysis.get("type", "Unknown")
        involved = disruption_analysis.get("involved", {})
        affected_countries = involved.get("countries", []) if isinstance(involved, dict) else []
        affected_industries = involved.get("industries", []) if isinstance(involved, dict) else []
        disruption_summary = {
            "type": disruption_type,
            "affected_countries": affected_countries,
            "affected_industries": affected_industries
        }
    
    if disruption_summary:
        story.append(Paragraph("Disruption Summary", heading_style))
        
        disruption_type = disruption_summary.get('type', 'Unknown')
        affected_countries = disruption_summary.get('affected_countries', [])
        if not affected_countries and isinstance(disruption_summary, dict):
            affected_countries = disruption_summary.get('affected_countries', [])
        
        affected_industries = disruption_summary.get('affected_industries', [])
        disrupted_companies_count = disruption_summary.get('disrupted_companies_count', 0)
        
        disruption_text = f"""
        <b>Disruption Type:</b> {disruption_type}<br/>
        <b>Affected Countries:</b> {', '.join(affected_countries) if affected_countries else 'Unknown'}<br/>
        <b>Affected Industries:</b> {', '.join(affected_industries) if affected_industries else 'Unknown'}<br/>
        <b>Disrupted Companies Count:</b> {disrupted_companies_count}
        """
        story.append(Paragraph(disruption_text, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Critical Suppliers - Show ALL suppliers with risk levels
    critical_suppliers = risk_assessment.get("critical_suppliers", []) if isinstance(risk_assessment, dict) else []
    all_suppliers = risk_assessment.get("all_suppliers_assessed", []) if isinstance(risk_assessment, dict) else []
    
    # Also try to get from supplier_risk_scores if available
    supplier_risk_scores = risk_assessment.get("supplier_risk_scores", {}) if isinstance(risk_assessment, dict) else {}
    
    # Use all_suppliers_assessed if available, otherwise use critical_suppliers
    # If neither exists, create from supplier_risk_scores
    suppliers_to_display = all_suppliers if all_suppliers else critical_suppliers
    
    # Fallback: Create supplier list from supplier_risk_scores if suppliers_to_display is empty
    if not suppliers_to_display and supplier_risk_scores:
        logger.info("Creating supplier list from supplier_risk_scores")
        suppliers_to_display = []
        for company, score in supplier_risk_scores.items():
            # Determine risk level
            if score >= 0.8:
                risk_level = "CRITICAL"
            elif score >= 0.6:
                risk_level = "HIGH"
            elif score >= 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Try to get country from kg_results
            country = "Unknown"
            kg_results = report_data.get("kg_results", {})
            if isinstance(kg_results, dict):
                # Search for company in kg_results chains
                for tier_key in ["tier_1", "tier_2", "tier_3", "tier_4"]:
                    chains = kg_results.get(tier_key, [])
                    for chain in chains:
                        for node in chain:
                            if node.get("company") == company:
                                country = node.get("country", "Unknown")
                                break
                        if country != "Unknown":
                            break
                    if country != "Unknown":
                        break
            
            suppliers_to_display.append({
                "company": company,
                "risk_score": score,
                "risk_level": risk_level,
                "country": country,
                "recommendation": f"Risk score {score:.4f} indicates {risk_level.lower()} risk level"
            })
    
    logger.info(f"PDF Export: Found {len(suppliers_to_display)} suppliers to display")
    
    if suppliers_to_display:
        story.append(Paragraph("Supplier Risk Assessment", heading_style))
        
        # Group by risk level
        critical = [s for s in suppliers_to_display if s.get("risk_level") == "CRITICAL"]
        high = [s for s in suppliers_to_display if s.get("risk_level") == "HIGH"]
        medium = [s for s in suppliers_to_display if s.get("risk_level") == "MEDIUM"]
        low = [s for s in suppliers_to_display if s.get("risk_level") == "LOW"]
        
        # Show CRITICAL suppliers
        if critical:
            story.append(Paragraph("<b>CRITICAL RISK SUPPLIERS</b>", subheading_style))
            supplier_data = [["Company", "Risk Score", "Country", "Recommendation"]]
            for supplier in critical[:15]:  # Top 15 critical
                supplier_data.append([
                    supplier.get("company", "Unknown"),
                    f"{supplier.get('risk_score', 0.0):.4f}",
                    supplier.get("country", "Unknown"),
                    supplier.get("recommendation", "Immediate action required")[:50] + "..." if len(supplier.get("recommendation", "")) > 50 else supplier.get("recommendation", "Immediate action required")
                ])
            
            supplier_table = Table(supplier_data, colWidths=[2.5*inch, 1*inch, 1.5*inch, 2*inch])
            supplier_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c0392b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffe6e6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(supplier_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Show HIGH risk suppliers
        if high:
            story.append(Paragraph("<b>HIGH RISK SUPPLIERS</b>", subheading_style))
            supplier_data = [["Company", "Risk Score", "Country", "Recommendation"]]
            for supplier in high[:15]:  # Top 15 high risk
                supplier_data.append([
                    supplier.get("company", "Unknown"),
                    f"{supplier.get('risk_score', 0.0):.4f}",
                    supplier.get("country", "Unknown"),
                    supplier.get("recommendation", "Increased monitoring required")[:50] + "..." if len(supplier.get("recommendation", "")) > 50 else supplier.get("recommendation", "Increased monitoring required")
                ])
            
            supplier_table = Table(supplier_data, colWidths=[2.5*inch, 1*inch, 1.5*inch, 2*inch])
            supplier_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e67e22')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fff4e6')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(supplier_table)
            story.append(Spacer(1, 0.2*inch))
        
        # Summary of all risk levels
        story.append(Paragraph("<b>Risk Level Summary</b>", subheading_style))
        summary_data = [
            ["Risk Level", "Count"],
            ["CRITICAL", str(len(critical))],
            ["HIGH", str(len(high))],
            ["MEDIUM", str(len(medium))],
            ["LOW", str(len(low))],
            ["TOTAL", str(len(suppliers_to_display))]
        ]
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Decisions and Action Plan
    decisions_data = report_data.get("decisions", {})
    if decisions_data:
        story.append(PageBreak())
        story.append(Paragraph("Recommended Actions and Decision Plan", heading_style))
        
        # Extract decisions and action_plan
        if isinstance(decisions_data, dict):
            decisions = decisions_data.get("decisions", {})
            action_plan = decisions_data.get("action_plan", {})
            executive_summary = decisions_data.get("executive_summary", {})
            
            # Executive Summary of Decisions
            if executive_summary:
                story.append(Paragraph("<b>Executive Summary</b>", subheading_style))
                exec_text = f"""
                <b>Total Suppliers Analyzed:</b> {executive_summary.get('total_suppliers_analyzed', 0)}<br/>
                <b>Critical Suppliers:</b> {executive_summary.get('critical_suppliers_count', 0)}<br/>
                <b>High Risk Suppliers:</b> {executive_summary.get('high_risk_suppliers_count', 0)}<br/>
                <b>Requires Immediate Attention:</b> {'Yes' if executive_summary.get('requires_immediate_attention', False) else 'No'}<br/>
                <b>Overall Recommendation:</b> {executive_summary.get('overall_recommendation', 'Continue monitoring')}
                """
                story.append(Paragraph(exec_text, styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
            
            # Action Plan by Timeline
            if action_plan and isinstance(action_plan, dict):
                story.append(Paragraph("<b>Action Plan by Timeline</b>", subheading_style))
                
                # Immediate Actions
                if action_plan.get("immediate_actions"):
                    story.append(Paragraph("<b>IMMEDIATE ACTIONS (0-7 days)</b>", subheading_style))
                    for action in action_plan["immediate_actions"][:10]:  # Top 10
                        if isinstance(action, dict):
                            supplier = action.get("supplier", "Unknown")
                            action_type = action.get("action", "Unknown")
                            story.append(Paragraph(f"• <b>{supplier}:</b> {action_type}", styles['Normal']))
                    story.append(Spacer(1, 0.15*inch))
                
                # Short-term Actions
                if action_plan.get("short_term_actions"):
                    story.append(Paragraph("<b>SHORT-TERM ACTIONS (within 30 days)</b>", subheading_style))
                    for action in action_plan["short_term_actions"][:10]:  # Top 10
                        if isinstance(action, dict):
                            supplier = action.get("supplier", "Unknown")
                            action_type = action.get("action", "Unknown")
                            story.append(Paragraph(f"• <b>{supplier}:</b> {action_type}", styles['Normal']))
                    story.append(Spacer(1, 0.15*inch))
                
                # Medium-term Actions
                if action_plan.get("medium_term_actions"):
                    story.append(Paragraph("<b>MEDIUM-TERM ACTIONS (ongoing)</b>", subheading_style))
                    for action in action_plan["medium_term_actions"][:10]:  # Top 10
                        if isinstance(action, dict):
                            supplier = action.get("supplier", "Unknown")
                            action_type = action.get("action", "Unknown")
                            story.append(Paragraph(f"• <b>{supplier}:</b> {action_type}", styles['Normal']))
                    story.append(Spacer(1, 0.15*inch))
            
            # Detailed Decisions for Critical Suppliers
            if decisions and isinstance(decisions, dict):
                story.append(PageBreak())
                story.append(Paragraph("Detailed Supplier Decisions", heading_style))
                
                # Show detailed decisions for CRITICAL and HIGH risk suppliers
                critical_decisions = {k: v for k, v in decisions.items() if isinstance(v, dict) and v.get("risk_level") in ["CRITICAL", "HIGH"]}
                
                for supplier, decision in list(critical_decisions.items())[:10]:  # Top 10
                    if isinstance(decision, dict):
                        story.append(Paragraph(f"<b>{supplier}</b>", subheading_style))
                        decision_text = f"""
                        <b>Risk Level:</b> {decision.get('risk_level', 'UNKNOWN')}<br/>
                        <b>Risk Score:</b> {decision.get('risk_score', 0.0):.4f}<br/>
                        <b>Action:</b> {decision.get('action', 'Unknown')}<br/>
                        <b>Priority:</b> {decision.get('priority', 'Unknown')}<br/>
                        <b>Timeline:</b> {decision.get('timeline', 'Unknown')}<br/>
                        <b>Urgency:</b> {decision.get('urgency', 'Unknown')}<br/>
                        <b>Justification:</b> {decision.get('justification', 'No justification provided')}<br/>
                        """
                        story.append(Paragraph(decision_text, styles['Normal']))
                        
                        # Actions Required
                        actions_required = decision.get("actions_required", [])
                        if actions_required:
                            story.append(Paragraph("<b>Actions Required:</b>", styles['Normal']))
                            for action in actions_required[:5]:  # Top 5 actions
                                story.append(Paragraph(f"  • {action}", styles['Normal']))
                        
                        story.append(Spacer(1, 0.2*inch))
    
    # Methodology
    methodology = risk_assessment.get("methodology", {}) if isinstance(risk_assessment, dict) else {}
    if methodology:
        story.append(PageBreak())
        story.append(Paragraph("Methodology and Risk Calculation", heading_style))
        
        metrics_used = methodology.get("metrics_used", [])
        if metrics_used:
            story.append(Paragraph("<b>Metrics Used:</b>", subheading_style))
            for metric in metrics_used:
                story.append(Paragraph(f"• {metric}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        risk_formula = methodology.get("risk_formula", "")
        if risk_formula:
            story.append(Paragraph(f"<b>Risk Calculation Formula:</b>", subheading_style))
            story.append(Paragraph(risk_formula, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        tier_proximity = methodology.get("tier_proximity", {})
        if tier_proximity:
            story.append(Paragraph("<b>Tier Proximity Factors:</b>", subheading_style))
            for tier, factor in tier_proximity.items():
                story.append(Paragraph(f"• {tier}: {factor}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        geographic_boost = methodology.get("geographic_risk_boost", "")
        if geographic_boost:
            story.append(Paragraph(f"<b>Geographic Risk Boost:</b> {geographic_boost}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
    
    # Supply Chain Overview (from kg_results)
    kg_results = report_data.get("kg_results", {})
    if kg_results and isinstance(kg_results, dict):
        story.append(PageBreak())
        story.append(Paragraph("Supply Chain Overview", heading_style))
        
        summary = kg_results.get("summary", {})
        if summary:
            sc_text = f"""
            <b>Total Disrupted Chains:</b> {summary.get('total_disrupted_chains', 0)}<br/>
            <b>Tier-1 Chains:</b> {summary.get('tier_1_count', 0)}<br/>
            <b>Tier-2 Chains:</b> {summary.get('tier_2_count', 0)}<br/>
            <b>Tier-3 Chains:</b> {summary.get('tier_3_count', 0)}<br/>
            <b>Tier-4 Chains:</b> {summary.get('tier_4_count', 0)}<br/>
            <b>Monitored Company:</b> {kg_results.get('monitored_company', company_name)}<br/>
            <b>Disrupted Countries:</b> {', '.join(kg_results.get('disrupted_countries', []))}
            """
            story.append(Paragraph(sc_text, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
    
    # Appendices
    story.append(PageBreak())
    story.append(Paragraph("Appendices", heading_style))
    
    # Full Supplier List (if available)
    if suppliers_to_display and len(suppliers_to_display) > 20:
        story.append(Paragraph("<b>Complete Supplier List</b>", subheading_style))
        story.append(Paragraph(f"Total suppliers assessed: {len(suppliers_to_display)}", styles['Normal']))
        story.append(Paragraph("(Top 20 critical/high-risk suppliers shown in detail above)", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Disruption Analysis Details
    if disruption_analysis and isinstance(disruption_analysis, dict):
        story.append(Paragraph("<b>Disruption Analysis Details</b>", subheading_style))
        disruption_type = disruption_analysis.get("type", "Unknown")
        involved = disruption_analysis.get("involved", {})
        if isinstance(involved, dict):
            countries = involved.get("countries", [])
            industries = involved.get("industries", [])
            companies = involved.get("companies", [])
            
            story.append(Paragraph(f"<b>Type:</b> {disruption_type}", styles['Normal']))
            if countries:
                story.append(Paragraph(f"<b>Countries:</b> {', '.join(countries)}", styles['Normal']))
            if industries:
                story.append(Paragraph(f"<b>Industries:</b> {', '.join(industries)}", styles['Normal']))
            if companies:
                story.append(Paragraph(f"<b>Companies Mentioned:</b> {', '.join(companies[:10])}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(
        f"<i>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    ))
    
    # Build PDF
    doc.build(story)
    
    # Log summary of what was included
    logger.info(f"✅ PDF report created: {output_path}")
    logger.info(f"   Report includes:")
    logger.info(f"   - Risk assessment: {'Yes' if risk_assessment else 'No'}")
    logger.info(f"   - Suppliers: {len(suppliers_to_display) if suppliers_to_display else 0}")
    logger.info(f"   - Decisions: {'Yes' if decisions_data else 'No'}")
    logger.info(f"   - KG results: {'Yes' if kg_results else 'No'}")
    logger.info(f"   - Disruption analysis: {'Yes' if disruption_analysis else 'No'}")
    logger.info(f"   - Methodology: {'Yes' if methodology else 'No'}")
    
    return output_path

def pdf_export_tool_entrypoint(
    report_data: Dict[str, Any],
    company_name: Optional[str] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> str:
    """
    Entry point for PDF export tool - company_name is optional and will be extracted from report_data if not provided.
    
    CRITICAL: Handles both dict and JSON string inputs (LLM may pass strings).
    """
    # Handle case where LLM passes report_data as a JSON string
    if isinstance(report_data, str):
        try:
            import json
            report_data = json.loads(report_data)
            logger.info("PDF Export: Parsed report_data from JSON string")
        except Exception as e:
            logger.error(f"PDF Export: Failed to parse report_data as JSON: {e}")
            raise ValueError(f"report_data must be a dict or valid JSON string. Got: {type(report_data)}")
    
    # Ensure report_data is a dict
    if not isinstance(report_data, dict):
        logger.error(f"PDF Export: report_data must be a dict, got {type(report_data)}")
        raise ValueError(f"report_data must be a dict, got {type(report_data)}")
    
    return create_executive_pdf(report_data, company_name, output_path, title)

pdf_export_tool = StructuredTool(
    name="export_executive_pdf",
    description="""
    Creates a professional PDF report for executive/CEO review.
    Includes risk assessment, critical suppliers, and recommended actions.
    Suitable for C-suite presentations and board meetings.
    
    **IMPORTANT:** The `company_name` parameter is optional. If not provided, it will be automatically extracted from `report_data['risk_assessment']['company_name']` or `report_data['kg_results']['monitored_company']`.
    
    **Usage:**
    - You can call this tool with just `report_data` - the company name will be extracted automatically
    - Or provide `company_name` explicitly if you have it
    """,
    func=pdf_export_tool_entrypoint,
    args_schema=PDFExportInput
)

