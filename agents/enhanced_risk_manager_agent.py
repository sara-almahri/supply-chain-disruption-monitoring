# agents/enhanced_risk_manager_agent.py

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from crewai import Agent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for Tier-1 aggregation
# ---------------------------------------------------------------------------


@dataclass
class ExposureTotals:
    total_chains: int = 0
    chains_per_tier: Dict[int, int] = field(
        default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0}
    )
    unique_companies: Set[str] = field(default_factory=set)
    unique_countries: Set[str] = field(default_factory=set)


@dataclass
class TierOneExposure:
    supplier: str
    downstream_nodes: Set[str] = field(default_factory=set)
    final_nodes_by_tier: Dict[int, Set[Tuple[str, str]]] = field(
        default_factory=lambda: {1: set(), 2: set(), 3: set(), 4: set()}
    )
    final_countries: Counter = field(default_factory=Counter)
    chain_examples: List[List[str]] = field(default_factory=list)

    def register_chain(self, chain: List[Dict[str, Any]]) -> None:
        if len(chain) < 2:
            return

        chain_companies: List[str] = []
        for idx, node in enumerate(chain):
            company = str(node.get("company", "")).strip()
            if not company:
                return
            chain_companies.append(company)

            if idx > 0:
                self.downstream_nodes.add(company)

            if idx == len(chain) - 1:
                tier = idx
                country = (
                    str(node.get("country", "Unknown")).strip() or "Unknown"
                )
                self.final_nodes_by_tier.setdefault(tier, set()).add(
                    (company, country)
                )
                self.final_countries[country] += 1

        if len(self.chain_examples) < 10:
            self.chain_examples.append(chain_companies)

    @property
    def disrupted_counts_by_tier(self) -> Dict[int, int]:
        return {tier: len(nodes) for tier, nodes in self.final_nodes_by_tier.items()}

    @property
    def total_unique_disrupted(self) -> int:
        return sum(self.disrupted_counts_by_tier.values())

    @property
    def max_disruption_tier(self) -> int:
        for tier in range(4, 0, -1):
            if self.disrupted_counts_by_tier.get(tier):
                return tier
        return 0


# ---------------------------------------------------------------------------
# Tier-1 focused risk manager
# ---------------------------------------------------------------------------


class EnhancedRiskManagerAgent(Agent):
    """
    Computes Tier-1 supplier risk scores by aggregating downstream disruption
    impact (Tier-2..Tier-4) into actionable metrics for decision-makers.
    
    CRITICAL: This agent uses direct computation (not LLM tools) to calculate
    risk scores for ALL Tier-1 suppliers without context overflow.
    """

    def __init__(self, company_name: str = "", **config):
        super().__init__(**config)
        object.__setattr__(self, "company_name", company_name)
        logger.info("🚀 EnhancedRiskManagerAgent initialised for %s", company_name)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        disruption_analysis = self._ensure_dict(inputs.get("disruption_analysis", {}))
        kg_results = self._ensure_dict(inputs.get("kg_results", {}))

        if not kg_results:
            logger.error("Missing `kg_results` for risk assessment.")
            return {"error": "Missing kg_results for risk assessment"}

        exposures, totals = self._build_tier1_exposures(kg_results)
        if not exposures:
            logger.warning(
                "No Tier-1 exposures detected in kg_results; returning baseline assessment."
            )
            return {
                "risk_assessment": self._empty_assessment(disruption_analysis, totals),
                "company_name": inputs.get("company_name", self.company_name),
            }

        disrupted_companies = sorted(totals.unique_companies)
        disrupted_countries = sorted(totals.unique_countries)

        resolved_company_name = self._resolve_company_name(self.company_name)
        
        # Get list of Tier-1 suppliers for focused metrics calculation
        tier1_supplier_names = list(exposures.keys())
        
        metrics_data = self._compute_graph_metrics(
            resolved_company_name, disrupted_companies, tier1_supplier_names
        )

        risk_scores, components = self._score_tier1_suppliers(exposures, metrics_data)
        tier1_rankings = self._build_ranked_profiles(
            exposures, risk_scores, components, metrics_data
        )

        thresholds = self._load_thresholds()
        critical_suppliers = [
            {
                "company": profile["supplier"],
                "risk_score": profile["risk_score"],
                "risk_level": profile["risk_level"],
                "component_breakdown": profile["component_breakdown"],
                "disrupted_counts_by_tier": profile["disrupted_counts_by_tier"],
            }
            for profile in tier1_rankings
            if profile["risk_level"] in {"CRITICAL", "HIGH"}
        ]

        critical_supplier_names = [
            supplier["company"] for supplier in critical_suppliers
        ]

        # Return ALL Tier-1 supplier risk scores (no top-10 filter)
        # Full coverage enables complete evaluation and fairer recall metrics
        all_supplier_scores = {
            supplier: round(score, 4) for supplier, score in risk_scores.items()
        }
        
        risk_assessment = {
            "company_name": self.company_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "supplier_risk_scores": all_supplier_scores,
            "tier1_risk_profiles": tier1_rankings,
            "critical_suppliers": critical_suppliers,
            "expected_critical_suppliers": [s.get("company") for s in critical_suppliers],
            "risk_metrics_summary": self._build_risk_summary(
                all_supplier_scores, tier1_rankings, thresholds
            ),
            "kg_summary": self._build_kg_summary(
                resolved_company_name, totals, disrupted_countries
            ),
            "disruption_summary": self._build_disruption_summary(
                disruption_analysis, disrupted_countries, disrupted_companies
            ),
            "methodology": self._methodology_section(),
            "recommendation": self._get_executive_recommendation(critical_suppliers),
        }

        logger.info(
            "✅ Risk assessment completed: %d Tier-1 suppliers scored (ALL returned).",
            len(all_supplier_scores),
        )

        return {
            "risk_assessment": risk_assessment,
            "company_name": inputs.get("company_name", self.company_name),
        }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _ensure_dict(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                import json

                return json.loads(value)
            except json.JSONDecodeError:
                logger.warning(
                    "Could not parse JSON string while preparing inputs; defaulting to empty dict."
                )
        return {}

    def _build_tier1_exposures(
        self, kg_results: Dict[str, Any]
    ) -> Tuple[Dict[str, TierOneExposure], ExposureTotals]:
        exposures: Dict[str, TierOneExposure] = {}
        totals = ExposureTotals()

        for tier_key in ("tier_1", "tier_2", "tier_3", "tier_4"):
            tier_data = kg_results.get(tier_key, [])
            if not isinstance(tier_data, list):
                continue

            tier_num = int(tier_key.split("_")[1])

            for chain in tier_data:
                if (
                    not isinstance(chain, list)
                    or len(chain) < 2
                    or not chain[0].get("company")
                ):
                    continue

                totals.total_chains += 1
                totals.chains_per_tier[tier_num] += 1

                final_node = chain[-1]
                final_company = str(final_node.get("company", "")).strip()
                final_country = (
                    str(final_node.get("country", "Unknown")).strip() or "Unknown"
                )

                if final_company:
                    totals.unique_companies.add(final_company)
                totals.unique_countries.add(final_country)

                tier1_company = str(chain[1].get("company", "")).strip()
                if not tier1_company:
                    continue

                profile = exposures.setdefault(
                    tier1_company, TierOneExposure(supplier=tier1_company)
                )
                profile.register_chain(chain)

        logger.info(
            "Tier-1 exposure map built: %d Tier-1 suppliers, %d unique disrupted companies.",
            len(exposures),
            len(totals.unique_companies),
        )

        return exposures, totals

    def _compute_graph_metrics(
        self, resolved_company_name: str, disrupted_companies: List[str], tier1_suppliers: List[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate graph metrics ONLY for Tier-1 suppliers (efficient).
        This avoids calculating metrics for 1,077 companies when we only need ~30.
        """
        from tools.tier1_metrics_calculator import calculate_tier1_metrics

        if not tier1_suppliers:
            logger.warning("No Tier-1 suppliers provided for metrics calculation")
            return {
                "centrality_metrics": {},
                "pagerank": {},
                "dependency_ratios": {},
            }

        try:
            metrics_data = calculate_tier1_metrics(
                monitored_company=resolved_company_name,
                tier1_suppliers=tier1_suppliers,
                disrupted_companies=disrupted_companies
            )
            if "error" in metrics_data:
                logger.warning(
                    "Tier-1 metrics calculator returned an error: %s",
                    metrics_data.get("error"),
                )
                return {
                    "centrality_metrics": {},
                    "pagerank": {},
                    "dependency_ratios": {},
                }
            return metrics_data
        except Exception as exc:
            logger.error("Failed to compute Tier-1 metrics: %s", exc)
            return {
                "centrality_metrics": {},
                "pagerank": {},
                "dependency_ratios": {},
            }

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    def _score_tier1_suppliers(
        self,
        exposures: Dict[str, TierOneExposure],
        metrics_data: Dict[str, Any],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Dict[str, float]]]]:
        centrality_metrics = metrics_data.get("centrality_metrics", {})
        dependency_ratios = metrics_data.get("dependency_ratios", {})
        pagerank = metrics_data.get("pagerank", {})

        exposure_feature: Dict[str, float] = {}
        downstream_centrality_feature: Dict[str, float] = {}
        tier1_centrality_feature: Dict[str, float] = {}
        dependency_feature: Dict[str, float] = {}
        depth_feature: Dict[str, float] = {}
        downstream_pagerank_feature: Dict[str, float] = {}

        EXPOSURE_WEIGHTS = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.35}

        for tier1, profile in exposures.items():
            counts = profile.disrupted_counts_by_tier
            exposure_raw = sum(
                counts.get(tier, 0) * EXPOSURE_WEIGHTS[tier] for tier in range(1, 5)
            )
            exposure_feature[tier1] = exposure_raw

            dependency_raw = min(max(dependency_ratios.get(tier1, 0.0), 0.0), 1.0)
            dependency_feature[tier1] = dependency_raw

            tier1_centrality_raw = self._calculate_centrality_score(
                centrality_metrics.get(tier1, {})
            )
            tier1_centrality_feature[tier1] = tier1_centrality_raw

            downstream_nodes = profile.downstream_nodes
            if downstream_nodes:
                centrality_scores = [
                    self._calculate_centrality_score(centrality_metrics.get(node, {}))
                    for node in downstream_nodes
                    if node in centrality_metrics
                ]
                if centrality_scores:
                    downstream_centrality_feature[tier1] = sum(centrality_scores) / len(
                        centrality_scores
                    )
                else:
                    downstream_centrality_feature[tier1] = 0.0

                pagerank_scores = [
                    pagerank.get(node, 0.0) for node in downstream_nodes if node in pagerank
                ]
                downstream_pagerank_feature[tier1] = (
                    sum(pagerank_scores) / len(pagerank_scores) if pagerank_scores else 0.0
                )
            else:
                downstream_centrality_feature[tier1] = 0.0
                downstream_pagerank_feature[tier1] = 0.0

            depth_feature[tier1] = profile.max_disruption_tier

        exposure_norm = self._normalize_map(exposure_feature)
        downstream_centrality_norm = self._normalize_map(downstream_centrality_feature)
        tier1_centrality_norm = self._normalize_map(tier1_centrality_feature)
        downstream_pagerank_norm = self._normalize_map(downstream_pagerank_feature)

        risk_scores: Dict[str, float] = {}
        component_breakdown: Dict[str, Dict[str, Dict[str, float]]] = {}

        for tier1 in exposures.keys():
            exposure_score = exposure_norm.get(tier1, 0.0)
            dependency_score = dependency_feature.get(tier1, 0.0)
            downstream_centrality_score = downstream_centrality_norm.get(tier1, 0.0)
            tier1_centrality_score = tier1_centrality_norm.get(tier1, 0.0)
            depth_score = depth_feature.get(tier1, 0) / 4.0
            downstream_pagerank_score = downstream_pagerank_norm.get(tier1, 0.0)

            risk_score = (
                0.35 * exposure_score
                + 0.25 * dependency_score
                + 0.2 * max(downstream_centrality_score, downstream_pagerank_score)
                + 0.1 * tier1_centrality_score
                + 0.1 * depth_score
            )
            risk_scores[tier1] = round(min(max(risk_score, 0.0), 1.0), 4)

            component_breakdown[tier1] = {
                "exposure": {
                    "raw": exposure_feature.get(tier1, 0.0),
                    "normalized": round(exposure_score, 4),
                },
                "dependency_ratio": {
                    "raw": dependency_feature.get(tier1, 0.0),
                    "normalized": round(dependency_score, 4),
                },
                "downstream_centrality": {
                    "raw": downstream_centrality_feature.get(tier1, 0.0),
                    "normalized": round(downstream_centrality_score, 4),
                },
                "downstream_pagerank": {
                    "raw": downstream_pagerank_feature.get(tier1, 0.0),
                    "normalized": round(downstream_pagerank_score, 4),
                },
                "tier1_centrality": {
                    "raw": tier1_centrality_feature.get(tier1, 0.0),
                    "normalized": round(tier1_centrality_score, 4),
                },
                "depth": {
                    "raw": depth_feature.get(tier1, 0.0),
                    "normalized": round(depth_score, 4),
                },
            }

        return risk_scores, component_breakdown

    def _normalize_map(self, values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        min_v = min(values.values())
        max_v = max(values.values())
        if max_v == min_v:
            return {k: (1.0 if v > 0 else 0.0) for k, v in values.items()}
        return {k: (v - min_v) / (max_v - min_v) for k, v in values.items()}

    def _build_ranked_profiles(
        self,
        exposures: Dict[str, TierOneExposure],
        risk_scores: Dict[str, float],
        components: Dict[str, Dict[str, Dict[str, float]]],
        metrics_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        thresholds = self._load_thresholds()
        pagerank = metrics_data.get("pagerank", {})

        rankings = []
        for supplier, profile in exposures.items():
            risk_score = risk_scores.get(supplier, 0.0)
            risk_level = self._risk_level(risk_score, thresholds)

            rankings.append(
                {
                    "supplier": supplier,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "disrupted_counts_by_tier": profile.disrupted_counts_by_tier,
                    "total_unique_disrupted_suppliers": profile.total_unique_disrupted,
                    "max_disruption_tier": profile.max_disruption_tier,
                    "downstream_supplier_count": len(profile.downstream_nodes),
                    "top_disrupted_countries": profile.final_countries.most_common(5),
                    "component_breakdown": components.get(supplier, {}),
                    "pagerank": round(pagerank.get(supplier, 0.0), 6),
                }
            )

        rankings.sort(key=lambda item: item["risk_score"], reverse=True)
        return rankings

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _build_disruption_summary(
        self,
        disruption_analysis: Dict[str, Any],
        disrupted_countries: List[str],
        disrupted_companies: List[str],
    ) -> Dict[str, Any]:
        return {
            "type": disruption_analysis.get("type", "Unknown"),
            "affected_countries": disrupted_countries
            or disruption_analysis.get("affected_countries", []),
            "affected_industries": self._extract_industries(disruption_analysis),
            "disrupted_companies_count": len(disrupted_companies),
            "disrupted_companies_sample": disrupted_companies[:25],
        }

    def _build_risk_summary(
        self,
        risk_scores: Dict[str, float],
        rankings: List[Dict[str, Any]],
        thresholds: Dict[str, float],
    ) -> Dict[str, Any]:
        counts = Counter(r["risk_level"] for r in rankings)
        return {
            "total_tier1_suppliers": len(risk_scores),
            "critical_suppliers": counts.get("CRITICAL", 0),
            "high_risk_suppliers": counts.get("HIGH", 0),
            "medium_risk_suppliers": counts.get("MEDIUM", 0),
            "low_risk_suppliers": counts.get("LOW", 0),
            "risk_thresholds": thresholds,
            "highest_risk_score": max(risk_scores.values()) if risk_scores else 0.0,
            "average_risk_score": (
                sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0.0
            ),
        }

    def _build_executive_summary(
        self, rankings: List[Dict[str, Any]], thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        critical_high = [
            r for r in rankings if r["risk_level"] in {"CRITICAL", "HIGH"}
        ]
        recommendation = self._get_executive_recommendation(critical_high)
        return {
            "requires_immediate_attention": any(
                r["risk_level"] == "CRITICAL" for r in rankings
            ),
            "total_critical_and_high_risk_suppliers": len(critical_high),
            "recommendation": recommendation,
        }

    def _methodology_section(self) -> Dict[str, Any]:
        return {
            "focus": "Tier-1 supplier risk, aggregated from downstream disruptions up to Tier-4.",
            "metrics": {
                "Disruption Exposure": (
                    "Weighted count of unique disrupted suppliers reachable from each Tier-1 "
                    "(Tier-1→Tier-4 weights: 1.0, 0.75, 0.5, 0.35)."
                ),
                "Dependency Ratio": (
                    "Fraction of a Tier-1's subnetwork impacted by disruptions "
                    "(enhanced_graph_metrics_tool)."
                ),
                "Downstream Criticality": (
                    "Average centrality/PageRank scores of disrupted downstream suppliers."
                ),
                "Tier-1 Centrality": (
                    "Importance of the Tier-1 node within the monitored company's network."
                ),
                "Depth": "Maximum tier at which disruptions are observed in the supplier's chain.",
            },
            "risk_formula": (
                "Risk = 0.35×Exposure + 0.25×Dependency + 0.20×Downstream Criticality "
                "(max of centrality & pagerank signals) + 0.10×Tier-1 Centrality + 0.10×Depth."
            ),
            "control_assumption": (
                "Operational decisions apply only to Tier-1 suppliers; downstream metrics inform urgency."
            ),
        }

    def _build_kg_summary(
        self,
        monitored_company: str,
        totals: ExposureTotals,
        disrupted_countries: List[str],
    ) -> Dict[str, Any]:
        return {
            "monitored_company": monitored_company,
            "disrupted_countries": disrupted_countries,
            "summary": {
                "total_disrupted_chains": totals.total_chains,
                "tier_1_chain_count": totals.chains_per_tier[1],
                "tier_2_chain_count": totals.chains_per_tier[2],
                "tier_3_chain_count": totals.chains_per_tier[3],
                "tier_4_chain_count": totals.chains_per_tier[4],
                "unique_disrupted_companies": len(totals.unique_companies),
            },
        }

    def _empty_assessment(
        self, disruption_analysis: Dict[str, Any], totals: ExposureTotals
    ) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "supplier_risk_scores": {},
            "tier1_risk_profiles": [],
            "critical_suppliers": [],
            "expected_critical_suppliers": [],
            "risk_metrics_summary": {
                "total_tier1_suppliers": 0,
                "critical_suppliers": 0,
                "high_risk_suppliers": 0,
                "medium_risk_suppliers": 0,
                "low_risk_suppliers": 0,
                "risk_thresholds": self._load_thresholds(),
                "highest_risk_score": 0.0,
                "average_risk_score": 0.0,
            },
            "kg_summary": self._build_kg_summary(self.company_name, totals, []),
            "disruption_summary": self._build_disruption_summary(
                disruption_analysis, [], []
            ),
            "methodology": self._methodology_section(),
            "recommendation": "No Tier-1 suppliers identified in disrupted chains.",
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _resolve_company_name(self, company_name: str) -> str:
        from tools.neo4j_setup import graph

        try:
            resolve_query = """
            MATCH (c:Company)
            WHERE toLower(c.name) CONTAINS toLower($name)
               OR toLower($name) CONTAINS toLower(c.name)
            RETURN c.name AS exact_name
            ORDER BY
              CASE WHEN toLower(c.name) = toLower($name) THEN 0 ELSE 1 END,
              size(c.name)
            LIMIT 1
            """
            result = graph.query(resolve_query, {"name": company_name})
            if result and result[0].get("exact_name"):
                resolved = result[0]["exact_name"]
                logger.info(
                    "Resolved monitored company '%s' to '%s'", company_name, resolved
                )
                return resolved
        except Exception as exc:
            logger.error("Failed to resolve company name '%s': %s", company_name, exc)
        return company_name

    def _calculate_centrality_score(self, centrality: Dict[str, float]) -> float:
        if not centrality:
            return 0.0

        betweenness = min(max(centrality.get("betweenness", 0.0), 0.0), 1.0)
        closeness = min(max(centrality.get("closeness", 0.0), 0.0), 1.0)
        eigenvector = min(max(centrality.get("eigenvector", 0.0), 0.0), 1.0)

        degree_centrality = centrality.get("degree_centrality")
        if degree_centrality is None:
            in_degree = min(max(centrality.get("in_degree", 0.0), 0.0), 1.0)
            out_degree = min(max(centrality.get("out_degree", 0.0), 0.0), 1.0)
            degree_centrality = (in_degree + out_degree) / 2.0
        else:
            degree_centrality = min(max(degree_centrality, 0.0), 1.0)

        score = (betweenness + closeness + eigenvector + degree_centrality) / 4.0
        return min(max(score, 0.0), 1.0)

    def _extract_industries(self, disruption_analysis: Dict[str, Any]) -> List[str]:
        involved = (
            disruption_analysis.get("involved", {})
            if isinstance(disruption_analysis, dict)
            else {}
        )
        if isinstance(involved, dict):
            industries = involved.get("industries")
            if industries:
                return industries
        return (
            disruption_analysis.get("affected_industries", [])
            if isinstance(disruption_analysis, dict)
            else []
        )

    def _load_thresholds(self) -> Dict[str, float]:
        try:
            from crew import load_company_config

            config = load_company_config()
            return config.get(
                "settings",
                {},
            ).get("risk_thresholds", {"critical": 0.8, "high": 0.6, "medium": 0.4, "low": 0.0})
        except Exception:
            return {"critical": 0.8, "high": 0.6, "medium": 0.4, "low": 0.0}

    def _risk_level(self, score: float, thresholds: Dict[str, float]) -> str:
        if score >= thresholds.get("critical", 0.8):
            return "CRITICAL"
        if score >= thresholds.get("high", 0.6):
            return "HIGH"
        if score >= thresholds.get("medium", 0.4):
            return "MEDIUM"
        return "LOW"

    def _get_executive_recommendation(self, critical_suppliers: List[Dict[str, Any]]) -> str:
        critical_count = len([s for s in critical_suppliers if s.get("risk_level") == "CRITICAL"])
        high_count = len([s for s in critical_suppliers if s.get("risk_level") == "HIGH"])

        if critical_count > 0:
            return (
                f"URGENT: {critical_count} Tier-1 supplier(s) require immediate action. "
                "Accelerate mitigation and activate alternative sourcing."
            )
        if high_count > 0:
            return (
                f"HIGH PRIORITY: {high_count} Tier-1 supplier(s) show elevated exposure. "
                "Increase monitoring, buffer inventory, and prepare contingency sourcing."
            )
        return "LOW RISK: No Tier-1 suppliers exceed high-risk thresholds. Maintain standard monitoring cadence."

