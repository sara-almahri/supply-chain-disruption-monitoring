"""
Executive HTML Report Generator
================================
Generates C-suite level executive reports from CSCO agent decisions.

Professional HTML reports with:
- Executive summary
- Risk overview with statistics
- Supplier-by-supplier analysis
- Action recommendations
- Visual formatting for board presentation
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from collections import Counter

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Report output directory
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "executive_reports"


class ExecutiveReportInput(BaseModel):
    """Input schema for executive report generation"""
    csco_decision_data: Dict[str, Any] = Field(..., description="Complete CSCO decision JSON with justifications")
    risk_assessment: Dict[str, Any] = Field(..., description="Risk assessment data from risk manager")
    disruption_analysis: Dict[str, Any] = Field(..., description="Disruption analysis from monitoring agent")
    company_name: str = Field(..., description="Monitored company name (e.g., 'Tesla')")


class ExecutiveReportGenerator(BaseTool):
    name: str = "ExecutiveReportGenerator"
    description: str = "Generates professional C-suite HTML executive report from CSCO decisions"
    args_schema: type = ExecutiveReportInput

    def _run(
        self,
        csco_decision_data: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        disruption_analysis: Dict[str, Any],
        company_name: str
    ) -> Dict[str, Any]:
        """Generate professional HTML executive report"""
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Generate report HTML
            html_content = self._generate_html_report(
                csco_decision_data,
                risk_assessment,
                disruption_analysis,
                company_name
            )
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{company_name}_Executive_Report_{timestamp}.html"
            filepath = REPORTS_DIR / filename
            
            with filepath.open("w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"Executive report generated: {filepath}")
            
            return {
                "success": True,
                "report_path": str(filepath),
                "report_filename": filename,
                "html_source": html_content,
                "message": f"Executive report generated successfully at {filepath}"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate executive report: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "html_source": None
            }
    
    def _generate_html_report(
        self,
        csco_data: Dict[str, Any],
        risk_data: Dict[str, Any],
        disruption_data: Dict[str, Any],
        company_name: str
    ) -> str:
        """Generate the HTML report content"""
        
        # Extract key data
        decisions = csco_data.get("decisions", {})
        decision_report = csco_data.get("decision_report", {})
        risk_summary = csco_data.get("risk_summary", {})
        
        # Calculate statistics
        stats = self._calculate_statistics(decisions, risk_data, disruption_data)
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company_name} Supply Chain Risk Executive Report</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header(company_name, csco_data.get("decision_timestamp"))}
        {self._generate_executive_summary(stats, disruption_data, decision_report)}
        {self._generate_risk_overview(stats, risk_data)}
        {self._generate_disruption_context(disruption_data, stats)}
        {self._generate_supplier_analysis(decisions, stats, company_name)}
        {self._generate_action_plan(decisions, stats)}
        {self._generate_recommendations(decision_report, risk_summary)}
        {self._generate_footer()}
    </div>
</body>
</html>
"""
        return html
    
    def _calculate_statistics(
        self,
        decisions: Dict[str, Any],
        risk_data: Dict[str, Any],
        disruption_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate key statistics for the report"""
        
        # Action counts
        action_counter = Counter()
        risk_level_counter = Counter()
        industry_counter = Counter()
        
        for supplier, decision in decisions.items():
            action = decision.get("action", "UNKNOWN")
            risk_level = decision.get("risk_level", "UNKNOWN")
            action_counter[action] += 1
            risk_level_counter[risk_level] += 1
        
        # Extract industry data from risk profiles
        tier1_profiles = risk_data.get("tier1_risk_profiles", [])
        for profile in tier1_profiles:
            supplier_name = profile.get("supplier", "")
            # Try to infer industry from component breakdown or use generic
            industry_counter["Automotive Components"] += 1  # Placeholder
        
        # Disruption statistics - handle empty or missing data gracefully
        disruption_involved = disruption_data.get("involved", {}) if disruption_data else {}
        disruption_countries = disruption_involved.get("countries", []) if isinstance(disruption_involved, dict) else []
        disruption_industries = disruption_involved.get("industries", []) if isinstance(disruption_involved, dict) else []
        disruption_companies = disruption_involved.get("companies", []) if isinstance(disruption_involved, dict) else []
        
        # Aggregate exposure from risk_data if available (most accurate)
        total_disrupted_nodes = 0
        tier_breakdown = {"tier_2": 0, "tier_3": 0, "tier_4": 0}
        
        # Try to get from risk_data first (most accurate source)
        if risk_data:
            tier1_profiles = risk_data.get("tier1_risk_profiles", [])
            if tier1_profiles:
                # Aggregate unique counts across all suppliers
                for profile in tier1_profiles:
                    disrupted_counts = profile.get("disrupted_counts_by_tier", {})
                    tier_breakdown["tier_2"] = max(tier_breakdown["tier_2"], disrupted_counts.get("tier_2", 0))
                    tier_breakdown["tier_3"] = max(tier_breakdown["tier_3"], disrupted_counts.get("tier_3", 0))
                    tier_breakdown["tier_4"] = max(tier_breakdown["tier_4"], disrupted_counts.get("tier_4", 0))
                
                # Get total from risk_metrics_summary if available
                metrics_summary = risk_data.get("risk_metrics_summary", {})
                total_disrupted_nodes = metrics_summary.get("total_unique_disrupted_suppliers", 0)
        
        # Fallback: aggregate from decisions if risk_data not available
        if total_disrupted_nodes == 0:
            for supplier, decision in decisions.items():
                exposure = decision.get("justification", {}).get("exposure_details", {})
                by_tier = exposure.get("by_tier", {})
                for tier_key, count in by_tier.items():
                    if tier_key in tier_breakdown:
                        tier_breakdown[tier_key] = max(tier_breakdown[tier_key], count)
            total_disrupted_nodes = sum(tier_breakdown.values())
        
        return {
            "total_suppliers": len(decisions),
            "action_counts": dict(action_counter),
            "risk_level_counts": dict(risk_level_counter),
            "industry_counts": dict(industry_counter),
            "disruption_countries": disruption_countries,
            "disruption_industries": disruption_industries,
            "disruption_companies": disruption_companies,
            "total_disrupted_nodes": total_disrupted_nodes,
            "tier_breakdown": tier_breakdown,
            "high_suppliers": [s for s, d in decisions.items() if d.get("risk_level") == "HIGH"],
            "medium_suppliers": [s for s, d in decisions.items() if d.get("risk_level") == "MEDIUM"],
            "low_suppliers": [s for s, d in decisions.items() if d.get("risk_level") == "LOW"],
        }
    
    def _get_css_styles(self) -> str:
        """Professional CSS styling for C-suite report"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            font-size: 18px;
            opacity: 0.9;
        }
        
        .header .meta {
            margin-top: 20px;
            font-size: 14px;
            opacity: 0.8;
        }
        
        .section {
            padding: 40px;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .section-title {
            font-size: 24px;
            font-weight: 600;
            color: #1e3a8a;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #3b82f6;
        }
        
        .executive-summary {
            background: #eff6ff;
            border-left: 5px solid #3b82f6;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            text-align: center;
        }
        
        .stat-card .number {
            font-size: 36px;
            font-weight: bold;
            color: #1e3a8a;
            margin-bottom: 5px;
        }
        
        .stat-card .label {
            font-size: 14px;
            color: #6b7280;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .stat-card .sub-label {
            font-size: 11px;
            color: #9ca3af;
            margin-top: 4px;
            text-transform: none;
            font-weight: 400;
        }
        
        .stat-card.high {
            border-left: 5px solid #ea580c;
            background: linear-gradient(135deg, #fff7ed 0%, #ffffff 100%);
        }
        
        .stat-card.medium {
            border-left: 5px solid #ca8a04;
            background: linear-gradient(135deg, #fefce8 0%, #ffffff 100%);
        }
        
        .stat-card.low {
            border-left: 5px solid #22c55e;
            background: linear-gradient(135deg, #f0fdf4 0%, #ffffff 100%);
        }
        
        .stat-card.action {
            border-left: 5px solid #3b82f6;
            background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
        }
        
        .supplier-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .supplier-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .supplier-name {
            font-size: 20px;
            font-weight: 600;
            color: #111827;
        }
        
        .risk-badge {
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }
        
        .risk-badge.critical {
            background: #fee2e2;
            color: #991b1b;
        }
        
        .risk-badge.high {
            background: #ffedd5;
            color: #9a3412;
        }
        
        .risk-badge.medium {
            background: #fef3c7;
            color: #92400e;
        }
        
        .risk-badge.low {
            background: #d1fae5;
            color: #065f46;
        }
        
        .action-badge {
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 12px;
            margin-left: 10px;
        }
        
        .action-badge.replace {
            background: #dbeafe;
            color: #1e40af;
        }
        
        .action-badge.monitor {
            background: #e0e7ff;
            color: #4338ca;
        }
        
        .action-badge.maintain {
            background: #f3f4f6;
            color: #374151;
        }
        
        .justification {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-left: 3px solid #3b82f6;
        }
        
        .justification h4 {
            font-size: 14px;
            font-weight: 600;
            color: #374151;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        
        .justification p {
            font-size: 14px;
            color: #4b5563;
            margin-bottom: 10px;
        }
        
        .exposure-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin: 10px 0;
        }
        
        .exposure-item {
            text-align: center;
            padding: 10px;
            background: #f3f4f6;
            border-radius: 6px;
        }
        
        .exposure-item .count {
            font-size: 20px;
            font-weight: bold;
            color: #1e3a8a;
        }
        
        .exposure-item .tier-label {
            font-size: 13px;
            color: #374151;
            font-weight: 600;
            margin-top: 5px;
        }
        
        .exposure-item .tier-description {
            font-size: 11px;
            color: #9ca3af;
            margin-top: 4px;
        }
        
        .exposure-item.tier-priority {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 2px solid #fca5a5;
        }
        
        .exposure-item.tier-affected {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border: 2px solid #93c5fd;
        }
        
        .action-list {
            list-style: none;
            padding: 0;
        }
        
        .action-item {
            padding: 15px;
            margin-bottom: 10px;
            background: white;
            border-left: 4px solid #3b82f6;
            border-radius: 4px;
        }
        
        .action-item strong {
            color: #1e3a8a;
        }
        
        .footer {
            padding: 30px;
            text-align: center;
            background: #f9fafb;
            color: #6b7280;
            font-size: 12px;
        }
        
        .alert {
            padding: 15px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 5px solid;
        }
        
        .alert.critical {
            background: #fee2e2;
            border-color: #dc2626;
            color: #991b1b;
        }
        
        .alert.warning {
            background: #fef3c7;
            border-color: #f59e0b;
            color: #92400e;
        }
        
        .alert.info {
            background: #dbeafe;
            border-color: #3b82f6;
            color: #1e40af;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }
        
        th {
            background: #f3f4f6;
            font-weight: 600;
            color: #374151;
        }
        
        tr:hover {
            background: #f9fafb;
        }
        """
    
    def _generate_header(self, company_name: str, timestamp: str) -> str:
        """Generate report header"""
        formatted_time = timestamp or datetime.now().strftime("%B %d, %Y %H:%M UTC")
        return f"""
        <div class="header">
            <h1>Supply Chain Risk Executive Report</h1>
            <div class="subtitle">{company_name}</div>
            <div class="meta">Generated: {formatted_time}</div>
            <div class="meta">Classification: CONFIDENTIAL - C-Suite Distribution Only</div>
        </div>
        """
    
    def _generate_executive_summary(
        self,
        stats: Dict[str, Any],
        disruption_data: Dict[str, Any],
        decision_report: Dict[str, Any]
    ) -> str:
        """Generate executive summary section"""
        disruption_type = disruption_data.get("type", "Unknown") if disruption_data else "Unknown"
        involved = disruption_data.get("involved", {}) if disruption_data else {}
        countries = ", ".join(involved.get("countries", [])) if isinstance(involved, dict) else "N/A"
        industries = ", ".join(involved.get("industries", [])) if isinstance(involved, dict) else "N/A"
        
        summary_text = decision_report.get("executive_summary", 
            f"This report analyzes the impact of {disruption_type} disruptions affecting {stats['total_suppliers']} Tier-1 suppliers."
        )
        
        # Get human-readable action counts
        replace_count = stats['action_counts'].get('REPLACE_SUPPLIER', 0)
        monitor_count = stats['action_counts'].get('INCREASE_MONITORING', 0)
        maintain_count = stats['action_counts'].get('MAINTAIN_STANDARD', 0)
        
        return f"""
        <div class="section executive-summary">
            <h2 class="section-title">Executive Summary</h2>
            <p style="font-size: 16px; line-height: 1.8; color: #374151; margin-bottom: 30px;">
                {summary_text}
            </p>
            
            <div class="stat-grid">
                <div class="stat-card high">
                    <div class="number">{stats['risk_level_counts'].get('HIGH', 0)}</div>
                    <div class="label">High Risk Suppliers</div>
                    <div class="sub-label">Require Immediate Action</div>
                </div>
                <div class="stat-card medium">
                    <div class="number">{stats['risk_level_counts'].get('MEDIUM', 0)}</div>
                    <div class="label">Medium Risk Suppliers</div>
                    <div class="sub-label">Enhanced Monitoring</div>
                </div>
                <div class="stat-card low">
                    <div class="number">{stats['risk_level_counts'].get('LOW', 0)}</div>
                    <div class="label">Low Risk Suppliers</div>
                    <div class="sub-label">Standard Protocol</div>
                </div>
                <div class="stat-card action">
                    <div class="number">{replace_count}</div>
                    <div class="label">Supplier Replacements</div>
                    <div class="sub-label">Immediate Action Required</div>
                </div>
            </div>
        </div>
        """
    
    def _generate_risk_overview(self, stats: Dict[str, Any], risk_data: Dict[str, Any]) -> str:
        """Generate risk overview section"""
        total_nodes = stats['total_disrupted_nodes']
        tier_breakdown = stats['tier_breakdown']
        
        # Get tier data from risk_metrics_summary if available (most accurate)
        tier2_total = tier_breakdown.get('tier_2', 0)
        tier3_total = tier_breakdown.get('tier_3', 0)
        tier4_total = tier_breakdown.get('tier_4', 0)
        
        # Try to get more accurate data from risk_data
        if risk_data:
            metrics_summary = risk_data.get("risk_metrics_summary", {})
            if metrics_summary:
                # Get chain counts by tier if available
                tier2_total = metrics_summary.get("tier_2_chain_count", tier2_total)
                tier3_total = metrics_summary.get("tier_3_chain_count", tier3_total)
                tier4_total = metrics_summary.get("tier_4_chain_count", tier4_total)
            
            # Fallback: aggregate from tier1_profiles
            if tier2_total == 0 and tier3_total == 0 and tier4_total == 0:
                tier1_profiles = risk_data.get("tier1_risk_profiles", [])
                for profile in tier1_profiles:
                    disrupted_counts = profile.get("disrupted_counts_by_tier", {})
                    tier2_total = max(tier2_total, disrupted_counts.get("tier_2", 0))
                    tier3_total = max(tier3_total, disrupted_counts.get("tier_3", 0))
                    tier4_total = max(tier4_total, disrupted_counts.get("tier_4", 0))
        
        return f"""
        <div class="section">
            <h2 class="section-title">Risk Overview & Exposure Analysis</h2>
            
            <div class="alert info">
                <strong>Aggregate Exposure:</strong> {total_nodes} disrupted companies identified across our extended supply chain (Tier-2 through Tier-4).
            </div>
            
            <h3 style="margin: 30px 0 20px 0; color: #374151; font-size: 18px; font-weight: 600;">Disruption Propagation by Tier</h3>
            <div class="exposure-grid" style="grid-template-columns: repeat(4, 1fr);">
                <div class="exposure-item tier-priority">
                    <div class="count">{tier2_total}</div>
                    <div class="tier-label">Tier-2 Disruptions</div>
                    <div class="tier-description">Direct suppliers to our Tier-1</div>
                </div>
                <div class="exposure-item">
                    <div class="count">{tier3_total}</div>
                    <div class="tier-label">Tier-3 Disruptions</div>
                    <div class="tier-description">Suppliers to Tier-2</div>
                </div>
                <div class="exposure-item">
                    <div class="count">{tier4_total}</div>
                    <div class="tier-label">Tier-4 Disruptions</div>
                    <div class="tier-description">Furthest tier</div>
                </div>
                <div class="exposure-item tier-affected">
                    <div class="count">{stats['total_suppliers']}</div>
                    <div class="tier-label">Tier-1 Affected</div>
                    <div class="tier-description">Our direct suppliers</div>
                </div>
            </div>
            
            <div style="margin-top: 30px; padding: 20px; background: #f0f9ff; border-left: 4px solid #0ea5e9; border-radius: 6px;">
                <p style="color: #0c4a6e; margin: 0; font-size: 15px; line-height: 1.6;">
                    <strong style="color: #0369a1;">Key Insight:</strong> Tier-2 disruptions represent the highest immediate risk, as they are only one tier away from directly impacting our Tier-1 suppliers and propagating to our operations. These require the most urgent attention.
                </p>
            </div>
        </div>
        """
    
    def _generate_disruption_context(self, disruption_data: Dict[str, Any], stats: Dict[str, Any]) -> str:
        """Generate disruption context section"""
        # Safely extract disruption data
        if not disruption_data or not isinstance(disruption_data, dict):
            disruption_type = "Not Specified"
            countries_list = []
            industries_list = []
            companies_list = []
            summary = "Disruption context data not available."
        else:
            disruption_type = disruption_data.get("type", "Not Specified")
            involved = disruption_data.get("involved", {})
            if isinstance(involved, dict):
                countries_list = involved.get("countries", [])
                industries_list = involved.get("industries", [])
                companies_list = involved.get("companies", [])
            else:
                countries_list = []
                industries_list = []
                companies_list = []
            summary = disruption_data.get("summary", "No additional context available.")
        
        # Use stats as fallback if available
        if not countries_list and stats.get("disruption_countries"):
            countries_list = stats["disruption_countries"] if isinstance(stats["disruption_countries"], list) else []
        if not industries_list and stats.get("disruption_industries"):
            industries_list = stats["disruption_industries"] if isinstance(stats["disruption_industries"], list) else []
        
        countries_str = ", ".join(countries_list) if countries_list else "Not Specified"
        industries_str = ", ".join(industries_list) if industries_list else "Not Specified"
        companies_str = ", ".join(companies_list[:10]) if companies_list else "Not Specified"  # Limit to 10 companies
        if companies_list and len(companies_list) > 10:
            companies_str += f" (and {len(companies_list) - 10} more)"
        
        return f"""
        <div class="section">
            <h2 class="section-title">Disruption Context</h2>
            
            <div style="background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <table style="margin: 0;">
                    <tr style="background: #f9fafb;">
                        <th style="width: 30%; padding: 16px; font-weight: 600; color: #374151; border-bottom: 2px solid #e5e7eb;">Disruption Type</th>
                        <td style="padding: 16px; color: #111827; font-weight: 500; border-bottom: 2px solid #e5e7eb;">{disruption_type}</td>
                    </tr>
                    <tr>
                        <th style="width: 30%; padding: 16px; font-weight: 600; color: #374151; border-bottom: 2px solid #e5e7eb;">Affected Regions</th>
                        <td style="padding: 16px; color: #111827; border-bottom: 2px solid #e5e7eb;">{countries_str}</td>
                    </tr>
                    <tr style="background: #f9fafb;">
                        <th style="width: 30%; padding: 16px; font-weight: 600; color: #374151; border-bottom: 2px solid #e5e7eb;">Affected Industries</th>
                        <td style="padding: 16px; color: #111827; border-bottom: 2px solid #e5e7eb;">{industries_str}</td>
                    </tr>
                    <tr>
                        <th style="width: 30%; padding: 16px; font-weight: 600; color: #374151;">Key Companies Mentioned</th>
                        <td style="padding: 16px; color: #111827;">{companies_str}</td>
                    </tr>
                </table>
            </div>
            
            <div style="margin-top: 25px; padding: 20px; background: #f9fafb; border-left: 4px solid #6366f1; border-radius: 8px;">
                <p style="color: #4b5563; margin: 0; font-size: 15px; line-height: 1.7;">
                    {summary}
                </p>
            </div>
        </div>
        """
    
    def _generate_supplier_analysis(self, decisions: Dict[str, Any], stats: Dict[str, Any], company_name: str) -> str:
        """Generate detailed supplier analysis"""
        
        # Sort suppliers by risk score (highest first)
        sorted_suppliers = sorted(
            decisions.items(),
            key=lambda x: x[1].get("risk_score_rounded", 0),
            reverse=True
        )
        
        suppliers_html = ""
        for supplier_name, decision in sorted_suppliers:
            risk_level = decision.get("risk_level", "UNKNOWN").lower()
            risk_score_raw = decision.get("risk_score_raw", 0)
            risk_score_rounded = decision.get("risk_score_rounded", 0)
            action = decision.get("action", "UNKNOWN")
            
            justification = decision.get("justification", {})
            risk_analysis = justification.get("risk_analysis", "No analysis provided")
            exposure_details = justification.get("exposure_details", {})
            industry_impact = justification.get("industry_impact", "No industry impact analysis provided")
            dependency_analysis = justification.get("dependency_analysis", "No dependency analysis provided")
            production_impact = justification.get("production_impact", "No production impact analysis provided")
            propagation_mechanics = justification.get("propagation_mechanics", "No propagation mechanics provided")
            action_rationale = justification.get("action_rationale", "No action rationale provided")
            
            # Format action badge with human-readable text
            action_mapping = {
                "REPLACE_SUPPLIER": ("replace", "Replace Supplier"),
                "INCREASE_MONITORING": ("monitor", "Increase Monitoring"),
                "MAINTAIN_STANDARD": ("maintain", "Maintain Standard Protocol")
            }
            action_class, action_display = action_mapping.get(action, ("maintain", "Standard Protocol"))
            
            # Exposure breakdown
            by_tier = exposure_details.get("by_tier", {})
            tier2 = by_tier.get("tier_2", 0)
            tier3 = by_tier.get("tier_3", 0)
            tier4 = by_tier.get("tier_4", 0)
            total_nodes = exposure_details.get("total_disrupted_nodes", 0)
            countries = exposure_details.get("top_disrupted_countries", [])
            countries_str = ", ".join(countries) if countries else "N/A"
            physical_propagation = exposure_details.get("physical_propagation", "")
            
            suppliers_html += f"""
            <div class="supplier-card">
                <div class="supplier-header">
                    <div>
                        <div class="supplier-name">{supplier_name}</div>
                        <div style="margin-top: 5px;">
                            <span class="risk-badge {risk_level}">Risk: {risk_level.upper()} ({risk_score_rounded})</span>
                            <span class="action-badge {action_class}">{action_display}</span>
                        </div>
                    </div>
                </div>
                
                <div class="justification">
                    <h4>Risk Score Analysis</h4>
                    <p>{risk_analysis}</p>
                    
                    <h4>Exposure Details & Physical Propagation</h4>
                    <p><strong>Total Disrupted Nodes:</strong> {total_nodes} | <strong>Geographic Concentration:</strong> {countries_str}</p>
                    <div class="exposure-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 15px;">
                        <div class="exposure-item">
                            <div class="count">{tier2}</div>
                            <div class="tier-label">Tier-2</div>
                        </div>
                        <div class="exposure-item">
                            <div class="count">{tier3}</div>
                            <div class="tier-label">Tier-3</div>
                        </div>
                        <div class="exposure-item">
                            <div class="count">{tier4}</div>
                            <div class="tier-label">Tier-4</div>
                        </div>
                    </div>
                    {f'<p style="margin-top: 10px; color: #4b5563;"><strong>Physical Propagation Timeline:</strong> {physical_propagation}</p>' if physical_propagation else ''}
                    
                    <h4>Industry Impact Analysis</h4>
                    <p>{industry_impact}</p>
                    
                    <h4>Dependency & Network Criticality</h4>
                    <p>{dependency_analysis}</p>
                    
                    <h4>Production Impact on {company_name}</h4>
                    <p>{production_impact}</p>
                    
                    <h4>Physical Supply Chain Propagation Mechanics</h4>
                    <p>{propagation_mechanics}</p>
                    
                    <h4>Action Rationale & Strategic Decision</h4>
                    <p>{action_rationale}</p>
                </div>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2 class="section-title">Detailed Supplier Analysis (Tier-1)</h2>
            <p style="margin-bottom: 20px; color: #6b7280;">
                Comprehensive risk assessment and strategic recommendations for each affected Tier-1 supplier, ordered by risk severity.
            </p>
            {suppliers_html}
        </div>
        """
    
    def _generate_action_plan(self, decisions: Dict[str, Any], stats: Dict[str, Any]) -> str:
        """Generate action plan summary"""
        
        # Group by action type
        replace_suppliers = []
        monitor_suppliers = []
        maintain_suppliers = []
        
        for supplier, decision in decisions.items():
            action = decision.get("action", "UNKNOWN")
            risk_score = decision.get("risk_score_rounded", 0)
            if action == "REPLACE_SUPPLIER":
                replace_suppliers.append((supplier, risk_score))
            elif action == "INCREASE_MONITORING":
                monitor_suppliers.append((supplier, risk_score))
            else:
                maintain_suppliers.append((supplier, risk_score))
        
        # Sort by risk score
        replace_suppliers.sort(key=lambda x: x[1], reverse=True)
        monitor_suppliers.sort(key=lambda x: x[1], reverse=True)
        maintain_suppliers.sort(key=lambda x: x[1], reverse=True)
        
        replace_html = "<ul class='action-list'>"
        for supplier, score in replace_suppliers:
            replace_html += f"<li class='action-item'><strong>{supplier}</strong> <span style='color: #6b7280;'>(Risk Score: {score})</span> - Initiate replacement sourcing immediately</li>"
        replace_html += "</ul>"
        
        monitor_html = "<ul class='action-list'>"
        for supplier, score in monitor_suppliers:
            monitor_html += f"<li class='action-item'><strong>{supplier}</strong> <span style='color: #6b7280;'>(Risk Score: {score})</span> - Enhance monitoring and prepare contingencies</li>"
        monitor_html += "</ul>"
        
        return f"""
        <div class="section">
            <h2 class="section-title">Prioritized Action Plan</h2>
            
            {f'<div class="alert warning" style="margin-bottom: 30px;"><strong style="font-size: 16px;">⚠️ URGENT ACTION REQUIRED:</strong> {len(replace_suppliers)} supplier{"s" if len(replace_suppliers) != 1 else ""} require{"s" if len(replace_suppliers) == 1 else ""} immediate replacement</div>' if replace_suppliers else ''}
            
            <div style="margin-bottom: 40px;">
                <h3 style="margin: 0 0 20px 0; color: #374151; font-size: 18px; font-weight: 600; padding-bottom: 10px; border-bottom: 2px solid #e5e7eb;">Immediate Actions: Supplier Replacement</h3>
                {replace_html if replace_suppliers else '<div style="padding: 20px; background: #f9fafb; border-radius: 6px;"><p style="color: #6b7280; margin: 0;">✓ No suppliers require immediate replacement at this time.</p></div>'}
            </div>
            
            <div style="margin-bottom: 40px;">
                <h3 style="margin: 0 0 20px 0; color: #374151; font-size: 18px; font-weight: 600; padding-bottom: 10px; border-bottom: 2px solid #e5e7eb;">Short-Term Actions: Enhanced Monitoring</h3>
                {monitor_html if monitor_suppliers else '<div style="padding: 20px; background: #f9fafb; border-radius: 6px;"><p style="color: #6b7280; margin: 0;">✓ No suppliers require enhanced monitoring at this time.</p></div>'}
            </div>
            
            <div>
                <h3 style="margin: 0 0 20px 0; color: #374151; font-size: 18px; font-weight: 600; padding-bottom: 10px; border-bottom: 2px solid #e5e7eb;">Standard Protocol</h3>
                <div style="padding: 20px; background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 6px;">
                    <p style="color: #166534; margin: 0; font-size: 15px;">
                        <strong>{len(maintain_suppliers)} supplier{"s" if len(maintain_suppliers) != 1 else ""}</strong> assessed as low risk - maintain standard monitoring protocols.
                    </p>
                </div>
            </div>
        </div>
        """
    
    def _generate_recommendations(self, decision_report: Dict[str, Any], risk_summary: Dict[str, Any]) -> str:
        """Generate strategic recommendations"""
        recommendation = risk_summary.get("recommendation", "Continue monitoring supply chain exposure.")
        attention_flag = risk_summary.get("attention_flag", False)
        
        return f"""
        <div class="section">
            <h2 class="section-title">Strategic Recommendations</h2>
            
            {f'<div class="alert critical"><strong>⚠️ ATTENTION REQUIRED:</strong> This situation requires immediate executive attention and action.</div>' if attention_flag else ''}
            
            <div style="background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0;">
                <h3 style="color: #1e3a8a; margin-bottom: 15px;">Executive Recommendation</h3>
                <p style="font-size: 16px; color: #374151; line-height: 1.8;">
                    {recommendation}
                </p>
            </div>
            
            <h3 style="margin: 20px 0 15px 0; color: #374151;">Next Steps</h3>
            <ul style="color: #4b5563; line-height: 2;">
                <li>Review and approve recommended supplier replacements</li>
                <li>Authorize enhanced monitoring protocols for at-risk suppliers</li>
                <li>Engage procurement and sourcing teams for immediate action</li>
                <li>Schedule follow-up review in 2 weeks to assess mitigation progress</li>
                <li>Prepare board communication on supply chain resilience measures</li>
            </ul>
        </div>
        """
    
    def _generate_footer(self) -> str:
        """Generate report footer"""
        return """
        <div class="footer">
            <p><strong>CONFIDENTIAL</strong> - This report contains proprietary supply chain intelligence.</p>
            <p>Distribution limited to C-Suite executives and authorized personnel only.</p>
            <p style="margin-top: 10px;">Generated by AI-Powered Supply Chain Risk Management System</p>
        </div>
        """


# Create tool instance
executive_report_tool = ExecutiveReportGenerator()

