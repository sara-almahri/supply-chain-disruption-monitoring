# agents/chief_supply_chain_agent.py

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from crewai import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChiefSupplyChainAgent(Agent):
    """
    Chief Supply Chain Agent:
    Analyzes the risk report from the RiskManagerAgent and makes strategic decisions
    to mitigate supply chain risks based on supplier risk scores.
    
    This agent:
    1. Filters risk_assessment to TOP 10 RISKIEST suppliers BEFORE LLM call (for efficiency)
    2. Uses LLM to generate expert decisions with board-level justifications
    3. Returns structured JSON output only (no PDF/HTML/visualization reports)
    """
    def __init__(self, **config):
        """
        Initialize the ChiefSupplyChainAgent with configuration.
        
        Args:
            **config: Configuration parameters from agents.yaml (e.g., role, goal, backstory).
        """
        super().__init__(**config)
        logger.info("🚀 ChiefSupplyChainAgent initialized (JSON output only, no PDF/HTML/visualization)")
    
    def _filter_risk_assessment_to_top10(self, context, task):
        """
        Filter risk_assessment to only include TOP 10 RISKIEST suppliers.
        This happens BEFORE the LLM call, so the LLM only processes 10 suppliers.
        
        Returns filtered context with risk_assessment containing only top 10 suppliers.
        """
        import json
        
        # Extract risk_assessment from context or previous task outputs
        risk_assessment = None
        
        # Try task.input_data FIRST (CrewAI uses this for prompt building)
        if hasattr(task, 'input_data') and isinstance(task.input_data, dict):
            risk_assessment = task.input_data.get("risk_assessment")
            if risk_assessment:
                logger.info("   ✅ Found risk_assessment in task.input_data")
        
        # Try context second
        if not risk_assessment and context:
            if isinstance(context, dict):
                risk_assessment = context.get("risk_assessment")
            elif isinstance(context, list):
                for item in context:
                    if isinstance(item, dict) and ("supplier_risk_scores" in item or "tier1_risk_profiles" in item):
                        risk_assessment = item
                        break
        
        # Try previous task outputs if not in context or input_data
        if not risk_assessment and hasattr(task, 'crew') and task.crew and hasattr(task.crew, 'tasks'):
            for prev_task in task.crew.tasks:
                if prev_task == task:
                    break
                if hasattr(prev_task, 'output') and prev_task.output:
                    output = prev_task.output
                    if hasattr(output, 'raw'):
                        output = output.raw
                    if isinstance(output, str):
                        try:
                            output = json.loads(output)
                        except:
                            import re
                            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', output, re.DOTALL)
                            if json_match:
                                try:
                                    output = json.loads(json_match.group(1))
                                except:
                                    continue
                    if isinstance(output, dict) and ("supplier_risk_scores" in output or "tier1_risk_profiles" in output):
                        risk_assessment = output
                        break
        
        if not risk_assessment:
            logger.warning("⚠️ No risk_assessment found to filter - proceeding with original context")
            return context
        
        # Extract supplier risk scores
        supplier_risk_scores = risk_assessment.get("supplier_risk_scores", {}) or {}
        if not supplier_risk_scores:
            logger.warning("⚠️ No supplier_risk_scores found - proceeding with original context")
            return context
        
        # FILTER TO TOP 10 RISKIEST SUPPLIERS
        sorted_suppliers = sorted(
            supplier_risk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top_10_suppliers = dict(sorted_suppliers[:10])
        total_count = len(supplier_risk_scores)
        
        logger.info(
            f"🔍 Filtering risk_assessment to TOP 10 RISKIEST suppliers "
            f"(out of {total_count} total Tier-1 suppliers)"
        )
        logger.info(
            f"   Top 10 risk scores: {[f'{s}={score:.4f}' for s, score in list(top_10_suppliers.items())[:5]]}..."
        )
        
        # Create filtered risk_assessment
        filtered_risk_assessment = risk_assessment.copy()
        filtered_risk_assessment["supplier_risk_scores"] = top_10_suppliers
        
        # Filter tier1_risk_profiles to only include top 10 suppliers
        tier1_profiles = risk_assessment.get("tier1_risk_profiles", []) or []
        top_10_supplier_set = set(top_10_suppliers.keys())
        filtered_profiles = [
            profile for profile in tier1_profiles
            if profile.get("supplier") in top_10_supplier_set
        ]
        filtered_risk_assessment["tier1_risk_profiles"] = filtered_profiles
        
        # CRITICAL: Update task.input_data if it exists (CrewAI uses this for prompt building)
        if hasattr(task, 'input_data') and isinstance(task.input_data, dict):
            task.input_data["risk_assessment"] = filtered_risk_assessment
            logger.info("   ✅ Updated task.input_data['risk_assessment'] with filtered version")
        
        # Update context with filtered risk_assessment
        if isinstance(context, dict):
            filtered_context = context.copy()
            filtered_context["risk_assessment"] = filtered_risk_assessment
            return filtered_context
        elif isinstance(context, list):
            filtered_context = []
            for item in context:
                if isinstance(item, dict) and ("supplier_risk_scores" in item or "tier1_risk_profiles" in item):
                    filtered_context.append(filtered_risk_assessment)
                else:
                    filtered_context.append(item)
            return filtered_context
        else:
            # Create new context dict
            return {"risk_assessment": filtered_risk_assessment}
    
    def execute_task(self, task, context=None, tools=None):
        """
        Override execute_task to:
        1. Pass ALL supplier risk scores to the LLM (no top-10 filter)
        2. Return JSON decision data only (no PDF/HTML/visualization reports)
        """
        import json
        
        # Log what we're receiving
        logger.info("🔍 CSCO Agent: Inspecting task inputs...")
        if hasattr(task, 'input_data'):
            logger.info(f"   Task input_data keys: {list(task.input_data.keys()) if isinstance(task.input_data, dict) else 'N/A'}")
        if context:
            logger.info(f"   Context type: {type(context)}")
            if isinstance(context, dict):
                logger.info(f"   Context keys: {list(context.keys())}")
                if "risk_assessment" in context:
                    risk_assessment = context["risk_assessment"]
                    if isinstance(risk_assessment, dict):
                        scores = risk_assessment.get("supplier_risk_scores", {})
                        logger.info(f"   Processing ALL {len(scores)} suppliers (no top-10 filter)")
        
        # Let CrewAI's LLM generate the decision JSON with ALL suppliers
        logger.info("🚀 CSCO Agent: Starting LLM decision generation for ALL suppliers...")
        logger.info(f"   Task description length: {len(getattr(task, 'description', ''))} chars")
        try:
            result = super().execute_task(task, context, tools)
            logger.info("✅ CSCO Agent: LLM decision generation completed")
        except Exception as e:
            logger.error(f"❌ CSCO Agent: LLM call failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Extract raw value if result is a TaskOutput object
        raw_result = result
        if hasattr(result, 'raw'):
            raw_result = result.raw
        
        # Return JSON decision data only (no PDF/HTML/visualization reports)
        # Parse JSON if it's a string, otherwise return as-is
        if isinstance(raw_result, str):
            try:
                decision_data = json.loads(raw_result)
                logger.info("✅ Returning JSON decision data (PDF/HTML/visualization generation disabled)")
                return json.dumps(decision_data, indent=2, default=str)
            except:
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_result, re.DOTALL)
                if json_match:
                    try:
                        decision_data = json.loads(json_match.group(1))
                        logger.info("✅ Returning JSON decision data (extracted from markdown)")
                        return json.dumps(decision_data, indent=2, default=str)
                    except:
                        pass
                # Return as-is if not JSON
                logger.info("✅ Returning raw result (not JSON)")
                return raw_result
        elif isinstance(raw_result, dict):
            logger.info("✅ Returning JSON decision data (converted from dict)")
            return json.dumps(raw_result, indent=2, default=str)
        else:
            logger.info("✅ Returning result as-is")
            return str(raw_result) if raw_result else ""

    def _execute_DEPRECATED(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task: analyze the risk assessment, make decisions, and export PDF report.
        
        Args:
            inputs: Dictionary containing:
                - 'risk_assessment' from the RiskManagerAgent (metadata with risk_payload_path)
                - 'kg_results' from KGQueryAgent (optional, for PDF)
                - 'disruption_analysis' from DisruptionMonitoringAgent (optional, for PDF)
                - 'company_name' for PDF export
        
        Returns:
            Dictionary with 'decisions' key mapping suppliers to actions and priorities, and 'pdf_path' if exported.
        """
        logging.info(f"ChiefSupplyChainAgent.execute() called with input keys: {list(inputs.keys())}")
        
        # Handle nested structure: inputs may contain {"risk_assessment": {...}}
        risk_assessment = inputs.get("risk_assessment", {})
        
        # CRITICAL: Handle nested risk_assessment structure from return_direct=True
        if isinstance(risk_assessment, dict) and "risk_assessment" in risk_assessment:
            # Check if it's a nested structure
            inner_risk = risk_assessment.get("risk_assessment")
            if isinstance(inner_risk, dict) and "supplier_risk_scores" in inner_risk:
                logging.info("[ChiefSupplyChainAgent] Unwrapping nested risk_assessment structure")
                risk_assessment = inner_risk
        
        # CrewAI may pass risk_assessment as a string (JSON) or dict
        if isinstance(risk_assessment, str):
            try:
                import json
                risk_assessment = json.loads(risk_assessment)
            except:
                logging.warning("Could not parse risk_assessment as JSON")
                risk_assessment = {}
        
        if not risk_assessment and isinstance(inputs, dict):
            # Try direct access if risk_assessment is the input itself
            # Also check if inputs itself is the risk_assessment
            if "supplier_risk_scores" in inputs or "risk_metrics_summary" in inputs:
                risk_assessment = inputs
            else:
                # Try to get from any nested structure
                for key in ["risk_assessment", "risk_analysis", "risk_report"]:
                    if key in inputs:
                        risk_assessment = inputs[key]
                        break
        
        if not risk_assessment:
            logging.error("No risk assessment provided for analysis.")
            logging.error(f"Available input keys: {list(inputs.keys())}")
            return {"error": "No risk assessment provided."}
        
        logging.info(f"Risk assessment keys: {list(risk_assessment.keys()) if isinstance(risk_assessment, dict) else type(risk_assessment)}")

        # Extract supplier risk scores - handle different structures
        # NOTE: Filtering to top 10 already happened in execute_task() before LLM call
        # This code path is for backward compatibility (if _make_decisions is called directly)
        supplier_risk_scores = {}
        if "supplier_risk_scores" in risk_assessment:
            supplier_risk_scores = risk_assessment.get("supplier_risk_scores", {})
        elif "risk_analysis" in risk_assessment:
            supplier_risk_scores = (
                risk_assessment.get("risk_analysis", {}).get("supplier_risk_scores", {})
            )
        
        if not supplier_risk_scores:
            logging.warning("Risk assessment contains no supplier risk scores.")
            # Return empty decisions but with executive summary
            return {
                "decisions": {},
                "executive_summary": "No suppliers identified in disrupted regions requiring immediate action.",
                "recommendation": "Continue standard monitoring procedures."
            }

        # Get executive summary from risk assessment if available
        executive_summary = risk_assessment.get("executive_summary", {})
        critical_suppliers = risk_assessment.get("critical_suppliers", [])
        
        # Get company name for PDF export
        company_name = risk_assessment.get("company_name") or inputs.get("company_name") or "Tesla Inc"
        
        tier1_profiles = risk_assessment.get("tier1_risk_profiles", [])
        
        decisions = self._make_decisions(
            supplier_risk_scores,
            critical_suppliers,
            tier1_profiles,
        )
        
        # Create comprehensive decision report
        decision_report = {
            "decisions": decisions,
            "executive_summary": {
                "total_suppliers_analyzed": len(supplier_risk_scores),
                "high_risk_suppliers_count": len([s for s in critical_suppliers if s.get("risk_level") == "HIGH"]),
                "medium_risk_suppliers_count": len([s for s in critical_suppliers if s.get("risk_level") == "MEDIUM"]),
                "requires_immediate_attention": executive_summary.get("requires_immediate_attention", False),
                "overall_recommendation": executive_summary.get("recommendation", "Continue monitoring.")
            },
            "action_plan": self._create_action_plan(decisions, critical_suppliers)
        }
        
        logging.info(f"✅ Generated decisions for {len(decisions)} suppliers.")

        # Return JSON-friendly response without generating external files
        response = {
            "company_name": company_name,
            "decision_timestamp": datetime.now().isoformat(),
            "decisions": decision_report["decisions"],
            "decision_report": decision_report,
            "risk_summary": {
                "total_suppliers_analyzed": decision_report["executive_summary"]["total_suppliers_analyzed"],
                "critical_suppliers_count": decision_report["executive_summary"]["critical_suppliers_count"],
                "high_risk_suppliers_count": decision_report["executive_summary"]["high_risk_suppliers_count"],
                "overall_recommendation": decision_report["executive_summary"]["overall_recommendation"],
                "requires_immediate_attention": decision_report["executive_summary"]["requires_immediate_attention"]
            }
        }

        return response

    def _make_decisions(
        self,
        supplier_risk_scores: Dict[str, float],
        critical_suppliers: List[Dict] = None,
        tier1_profiles: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Determine mitigation actions based on supplier risk scores with comprehensive analysis.
        
        Uses configurable thresholds:
        - Critical risk (>=0.8): Immediate replacement required
        - High risk (>=0.6): Increase inventory + identify alternatives
        - Medium risk (>=0.4): Increase monitoring + buffer inventory
        - Low risk (<0.4): Standard monitoring
        
        Args:
            supplier_risk_scores: Dictionary mapping supplier names to risk scores (0.0-1.0).
            critical_suppliers: List of critical suppliers with risk levels from risk assessment.
        
        Returns:
            Dictionary mapping suppliers to actions, priorities, and justifications.
        """
        # Load thresholds from config (DEPRECATED - using new 3-stage thresholds)
        import math
        high_threshold = 0.6  # HIGH: ≥ 0.6 (when rounded)
        medium_threshold = 0.45  # MEDIUM: 0.45-0.59 (when rounded)
        # LOW: < 0.45 (when rounded)
        
        # Create a map of suppliers to risk levels for easy lookup
        supplier_risk_levels = {}
        if critical_suppliers:
            for supplier_info in critical_suppliers:
                if isinstance(supplier_info, dict):
                    company = supplier_info.get("company")
                    risk_level = supplier_info.get("risk_level")
                    if company:
                        supplier_risk_levels[company] = risk_level
        profile_lookup = {
            profile.get("supplier"): profile for profile in (tier1_profiles or [])
        }

        decisions = {}
        for supplier, score in supplier_risk_scores.items():
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                logging.warning(f"Invalid risk score for {supplier}: {score}. Skipping.")
                continue
            
            # Round score UP to 1 decimal place FIRST, then determine risk level
            rounded_score = math.ceil(score * 10) / 10.0
            
            # Get risk level from critical_suppliers if available, otherwise determine from rounded score
            risk_level = supplier_risk_levels.get(supplier)
            if not risk_level:
                if rounded_score >= high_threshold:
                    risk_level = "HIGH"
                elif rounded_score >= medium_threshold:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
            
            supplier_profile = profile_lookup.get(supplier, {})
            disrupted_counts = supplier_profile.get("disrupted_counts_by_tier", {})
            max_disruption_tier = supplier_profile.get("max_disruption_tier")
            component_breakdown = supplier_profile.get("component_breakdown", {})
            downstream_count = supplier_profile.get("downstream_supplier_count")

            # HIGH risk (≥0.6 when rounded)
            if rounded_score >= high_threshold or risk_level == "HIGH":
                decisions[supplier] = {
                    "supplier": supplier,
                    "risk_score": round(score, 4),
                    "risk_level": "HIGH",
                    "action": "REPLACE_SUPPLIER",
                    "priority": "HIGH",
                    "justification": f"Risk score {rounded_score:.1f} (rounded from {score:.4f}) exceeds high threshold (≥0.6). This supplier requires replacement or increased monitoring to prevent supply chain disruption.",
                    "timeline": "IMMEDIATE (0-7 days)",
                    "urgency": "EXTREME",
                    "actions_required": [
                        "Contact supplier immediately for status update",
                        "Activate emergency supplier search and qualification process",
                        "Identify and qualify alternative suppliers in non-disrupted regions",
                        "Assess lead time for supplier transition (target: <30 days)",
                        "Evaluate cost impact of replacement",
                        "Ensure quality standards are maintained with new supplier",
                        "Notify C-Suite and executive team",
                        "Implement contingency inventory management"
                    ],
                    "success_metrics": [
                        "Alternative supplier identified and qualified within 14 days",
                        "Supplier transition plan established within 7 days",
                        "Zero production disruption",
                        "Cost increase < 15%"
                    ],
                    "disrupted_counts_by_tier": disrupted_counts,
                    "max_disruption_tier": max_disruption_tier,
                    "downstream_supplier_count": downstream_count,
                    "component_breakdown": component_breakdown,
                }
            # MEDIUM risk (0.45-0.59 when rounded)
            elif rounded_score >= medium_threshold and rounded_score < high_threshold or risk_level == "MEDIUM":
                decisions[supplier] = {
                    "supplier": supplier,
                    "risk_score": round(score, 4),
                    "risk_level": "MEDIUM",
                    "action": "INCREASE_MONITORING",
                    "priority": "MEDIUM",
                    "justification": f"Risk score {rounded_score:.1f} (rounded from {score:.4f}) is in MEDIUM range (0.45-0.59). Enhanced monitoring recommended.",
                    "timeline": "ONGOING (continuous monitoring)",
                    "urgency": "MODERATE",
                    "actions_required": [
                        "Increase monitoring frequency to bi-weekly",
                        "Maintain standard inventory levels with 20% buffer",
                        "Develop contingency plans",
                        "Track supplier performance metrics",
                        "Identify potential alternative suppliers",
                        "Regular status updates with supplier"
                    ],
                    "success_metrics": [
                        "Bi-weekly monitoring reports",
                        "Contingency plan documented",
                        "Performance metrics tracked",
                        "Alternative suppliers identified"
                    ],
                    "disrupted_counts_by_tier": disrupted_counts,
                    "max_disruption_tier": max_disruption_tier,
                    "downstream_supplier_count": downstream_count,
                    "component_breakdown": component_breakdown,
                }
            # LOW risk (<0.45 when rounded)
            else:
                decisions[supplier] = {
                    "supplier": supplier,
                    "risk_score": round(score, 4),
                    "risk_level": "LOW",
                    "action": "MAINTAIN_STANDARD",
                    "priority": "LOW",
                    "justification": f"Risk score {rounded_score:.1f} (rounded from {score:.4f}) is LOW (<0.45). Standard monitoring procedures apply.",
                    "timeline": "STANDARD (quarterly reviews)",
                    "urgency": "LOW",
                    "actions_required": [
                        "Continue standard monitoring procedures",
                        "Review quarterly risk assessments",
                        "Maintain current inventory levels",
                        "Track supplier performance"
                    ],
                    "success_metrics": [
                        "Quarterly risk assessments completed",
                        "No disruption events",
                        "Standard performance metrics met"
                    ],
                    "disrupted_counts_by_tier": disrupted_counts,
                    "max_disruption_tier": max_disruption_tier,
                    "downstream_supplier_count": downstream_count,
                    "component_breakdown": component_breakdown,
                }
        return decisions
    
    def _create_action_plan(self, decisions: Dict[str, Dict[str, Any]], critical_suppliers: List[Dict] = None) -> Dict[str, Any]:
        """Create a comprehensive action plan from decisions"""
        action_plan = {
            "immediate_actions": [],
            "short_term_actions": [],
            "medium_term_actions": [],
            "long_term_actions": []
        }
        
        for supplier, decision in decisions.items():
            risk_level = decision.get("risk_level", "LOW")
            timeline = decision.get("timeline", "")
            
            if risk_level == "HIGH":
                action_plan["immediate_actions"].append({
                    "supplier": supplier,
                    "action": decision.get("action"),
                    "timeline": timeline
                })
            elif risk_level == "HIGH":
                action_plan["short_term_actions"].append({
                    "supplier": supplier,
                    "action": decision.get("action"),
                    "timeline": timeline
                })
            elif risk_level == "MEDIUM":
                action_plan["medium_term_actions"].append({
                    "supplier": supplier,
                    "action": decision.get("action"),
                    "timeline": timeline
                })
            else:
                action_plan["long_term_actions"].append({
                    "supplier": supplier,
                    "action": decision.get("action"),
                    "timeline": timeline
                })
        
        return action_plan

# Example usage (for testing purposes)
if __name__ == "__main__":
    agent = ChiefSupplyChainAgent(
        role="Chief Supply Chain Officer",
        goal="Mitigate supply chain risks through strategic decisions",
        backstory="Experienced executive specializing in supply chain resilience"
    )
    sample_input = {
        "risk_report": {
            "risk_analysis": {
                "supplier_risk_scores": {
                    "SupplierA": 0.9,
                    "SupplierB": 0.6,
                    "SupplierC": 0.3
                }
            }
        }
    }
    result = agent.execute(sample_input)
    print(result)