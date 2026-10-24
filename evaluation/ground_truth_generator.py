# ground_truth_generator.py
# Generates ground truth answers by hardcoding KG queries for each scenario

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from dotenv import load_dotenv

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def register_chain(self, chain: List[Dict[str, Any]]) -> None:
        if len(chain) < 2:
            return

        for idx, node in enumerate(chain):
            company = str(node.get("company", "")).strip()
            if not company:
                return

            if idx > 0:
                self.downstream_nodes.add(company)

            if idx == len(chain) - 1:
                tier = idx
                country = str(node.get("country", "Unknown")).strip() or "Unknown"
                self.final_nodes_by_tier.setdefault(tier, set()).add(
                    (company, country)
                )
                self.final_countries[country] += 1

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


class GroundTruthGenerator:
    """Generates ground truth answers by hardcoding KG queries"""
    
    def __init__(self, monitored_company: str = "Tesla Inc"):
        self.monitored_company = monitored_company
        logger.info(f"Ground truth generator initialized for {monitored_company}")

    _network = None
    _resolver = None

    @classmethod
    def network(cls):
        if cls._network is None:
            from offline import kg
            cls._network = kg.load()
        return cls._network
    
    # ------------------------------------------------------------------
    # Shared Tier-1 risk utilities (mirrors production agent logic)
    # ------------------------------------------------------------------

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

        return exposures, totals

    def _compute_graph_metrics(
        self, disrupted_companies: List[str]
    ) -> Dict[str, Any]:
        from tools.enhanced_graph_metrics_tool import calculate_comprehensive_metrics

        return calculate_comprehensive_metrics(
            target_company=self.monitored_company,
            disrupted_companies=disrupted_companies,
            include_centrality=True,
            include_pagerank=True,
            include_dependency=True,
        )

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

            dependency_feature[tier1] = min(
                max(dependency_ratios.get(tier1, 0.0), 0.0), 1.0
            )

            tier1_centrality_feature[tier1] = self._calculate_centrality_score(
                centrality_metrics.get(tier1, {})
            )

            downstream_nodes = profile.downstream_nodes
            if downstream_nodes:
                centrality_scores = [
                    self._calculate_centrality_score(centrality_metrics.get(node, {}))
                    for node in downstream_nodes
                    if node in centrality_metrics
                ]
                downstream_centrality_feature[tier1] = (
                    sum(centrality_scores) / len(centrality_scores)
                    if centrality_scores
                    else 0.0
                )

                pagerank_scores = [
                    pagerank.get(node, 0.0) for node in downstream_nodes if node in pagerank
                ]
                downstream_pagerank_feature[tier1] = (
                    sum(pagerank_scores) / len(pagerank_scores)
                    if pagerank_scores
                    else 0.0
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
        components: Dict[str, Dict[str, Dict[str, float]]] = {}

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

            components[tier1] = {
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

        return risk_scores, components

    def _normalize_map(self, values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        min_v = min(values.values())
        max_v = max(values.values())
        if max_v == min_v:
            return {k: (1.0 if v > 0 else 0.0) for k, v in values.items()}
        return {k: (v - min_v) / (max_v - min_v) for k, v in values.items()}

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
        return min(
            max((betweenness + closeness + eigenvector + degree_centrality) / 4.0, 0.0),
            1.0,
        )

    def _methodology_section(self) -> Dict[str, Any]:
        return {
            "focus": "Tier-1 supplier risk, aggregated from downstream disruptions up to Tier-4.",
            "metrics": {
                "Disruption Exposure": (
                    "Weighted count of unique disrupted suppliers reachable from each Tier-1 "
                    "(Tier-1→Tier-4 weights: 1.0, 0.75, 0.5, 0.35)."
                ),
                "Dependency Ratio": (
                    "Share of the supplier's own upstream network (the supplier and companies "
                    "on simple paths of length 1-3 from it) that is disrupted."
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
        }

    def _build_kg_summary(
        self, totals: ExposureTotals, disrupted_countries: List[str]
    ) -> Dict[str, Any]:
        return {
            "monitored_company": self.monitored_company,
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

    @staticmethod
    def _get_executive_recommendation(level_counts: Dict[str, int]) -> str:
        high_count = level_counts.get("HIGH", 0)
        medium_count = level_counts.get("MEDIUM", 0)
        parts = []
        if high_count:
            parts.append(
                f"{high_count} Tier-1 supplier(s) score 0.60 or above (HIGH): REPLACE_SUPPLIER."
            )
        if medium_count:
            parts.append(
                f"{medium_count} Tier-1 supplier(s) score 0.45 to below 0.60 (MEDIUM): INCREASE_MONITORING."
            )
        if not sum(level_counts.values()):
            return "NO EXPOSURE: No Tier-1 supplier has a disrupted supply path. MAINTAIN_STANDARD for all."
        if not parts:
            return "LOW RISK: No Tier-1 supplier scores 0.45 or above. MAINTAIN_STANDARD for all."
        prefix = "HIGH PRIORITY: " if high_count else "MODERATE RISK: "
        return prefix + " ".join(parts) + " All other Tier-1 suppliers: MAINTAIN_STANDARD."

    TYPE_LABELS = {
        "Cybersecurity Incident": "Cyber Attack",
        "Labor Strike": "Labour Strike",
    }

    def _risk_level(self, score: float) -> str:
        """Raw score against the stated thresholds, without rounding. HIGH ≥ 0.60, MEDIUM ≥ 0.45, else LOW."""
        from offline.risk import risk_level
        return risk_level(score)
    
    def generate_ground_truth(self, scenario: Dict) -> Dict[str, Any]:
        """
        Generate ground truth for a scenario by hardcoding KG queries.
        
        Returns the ground truth for the four evaluated agents:
        - disruption_analysis
        - kg_results (Tier-0 to Tier-4 chains)
        - risk_assessment (Tier-1 focused)
        - scenario_metadata
        """
        logger.info(f"Generating ground truth for {scenario['scenario_id']}")
        
        disrupted_countries = scenario.get("affected_countries", [])
        disrupted_industries = scenario.get("affected_industries", [])
        disrupted_companies = scenario.get("affected_companies", [])
        is_true_positive = scenario.get("is_true_positive", True)
        
        # 2. Ground truth for KG results (hardcoded query) - MUST RUN FIRST to get ALL companies
        scope = scenario.get("scope", "country")
        from tools.full_supply_chain_path_tool import build_disrupted_supply_chains

        kg_results_gt = build_disrupted_supply_chains(
            monitored_company=self.monitored_company,
            disrupted_countries=disrupted_countries,
            disrupted_companies=disrupted_companies,
            disrupted_industries=disrupted_industries,
            scope=scope,
        )
        
        # For agent 1 ground truth, limit disrupted companies to those explicitly mentioned
        # in the scenario (i.e., the news article text). This ensures the ground truth matches
        # what the disruption monitoring agent can infer without KG lookups.
        final_disrupted_companies = sorted({
            company for company in disrupted_companies or []
            if company and isinstance(company, str)
        })
        
        # 1. Ground truth for disruption analysis - NOW includes ALL companies found
        disruption_analysis_gt = {
            "scope": scope,
            "type": self.TYPE_LABELS.get(
                scenario.get("disruption_type", "Unknown"),
                scenario.get("disruption_type", "Unknown"),
            ),
            "involved": {
                "countries": disrupted_countries,
                "industries": disrupted_industries,
                "companies": final_disrupted_companies  # Limited to entities explicitly referenced in article text
            },
            "summary": f"{scenario.get('disruption_type')} in {', '.join(disrupted_countries)} affecting {', '.join(disrupted_industries)}. "
                      f"Found {len(final_disrupted_companies)} affected companies in {self.monitored_company}'s supply chain."
        }
        
        # 3. Ground truth risk assessment using identical Tier-1 logic
        risk_assessment_gt = self._calculate_risk_assessment(
            kg_results_gt,
            disruption_analysis_gt,
            scenario
        )
        
        return {
            "disruption_analysis": disruption_analysis_gt,
            "kg_results": kg_results_gt,
            "risk_assessment": risk_assessment_gt,
            "scenario_metadata": {
                "scenario_id": scenario.get("scenario_id"),
                "is_true_positive": is_true_positive,
                "target_tier": scenario.get("target_tier", 1)
            }
        }
    
    def _calculate_risk_assessment(
        self,
        kg_results: Dict,
        disruption_analysis: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute ground truth risk metrics mirroring Tier-1 production logic."""
        exposures, totals = self._build_tier1_exposures(kg_results)
        
        from offline import kg as offline_kg
        from offline import risk as offline_risk

        affected_tier1_suppliers = sorted(exposures.keys())
        logger.info(
            f"[_calculate_risk_assessment] Computing metrics for {len(affected_tier1_suppliers)} "
            f"affected Tier-1 suppliers (out of {len(totals.unique_companies)} total disrupted companies)"
        )
        from offline import names as offline_names

        involved = disruption_analysis.get("involved", {})
        net = self.network()
        if GroundTruthGenerator._resolver is None:
            GroundTruthGenerator._resolver = offline_names.Resolver(net)
        match = GroundTruthGenerator._resolver.disruption(
            involved.get("countries", []), involved.get("industries", []), involved.get("companies", []),
            scope=disruption_analysis.get("scope", "country"),
        )["match"]
        edges = offline_kg.metrics_graph_edges(net, self.monitored_company, "simple")
        risk_scores, components = offline_risk.risk_scores(
            kg_results, self.monitored_company, edges,
            net=net, countries=match["countries"], companies=match["companies"], industries=match["industries"],
        )

        tier1_profiles = []
        for supplier, profile in exposures.items():
            risk_score = risk_scores.get(supplier, 0.0)
            risk_level = self._risk_level(risk_score)
            tier1_profiles.append(
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
                }
            )

        tier1_profiles.sort(key=lambda item: item["risk_score"], reverse=True)
        critical_suppliers = [
            {
                "company": profile["supplier"],
                "risk_score": profile["risk_score"],
                "risk_level": profile["risk_level"],
                "component_breakdown": profile["component_breakdown"],
                "disrupted_counts_by_tier": profile["disrupted_counts_by_tier"],
            }
            for profile in tier1_profiles
            if profile["risk_level"] == "HIGH"  # Only HIGH risk suppliers are critical
        ]

        counts = Counter(profile["risk_level"] for profile in tier1_profiles)
        metrics_summary = {
            "total_tier1_suppliers": len(tier1_profiles),
            "high_risk_suppliers": counts.get("HIGH", 0),
            "medium_risk_suppliers": counts.get("MEDIUM", 0),
            "low_risk_suppliers": counts.get("LOW", 0),
            "risk_thresholds": {"high": 0.6, "medium": 0.45, "low": 0.0},
            "highest_risk_score": max(risk_scores.values()) if risk_scores else 0.0,
            "average_risk_score": (
                sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0.0
            ),
        }

        scenario["expected_risk_score_range"] = (
            round(min(risk_scores.values()), 4) if risk_scores else 0.0,
            round(max(risk_scores.values()), 4) if risk_scores else 0.0,
        )

        kg_summary = self._build_kg_summary(totals, involved.get("countries", []))

        return {
            "company_name": self.monitored_company,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "supplier_risk_scores": risk_scores,
            "tier1_risk_profiles": tier1_profiles,
            "critical_suppliers": critical_suppliers,
            "expected_critical_suppliers": [
                supplier["company"] for supplier in critical_suppliers
            ],
            "risk_metrics_summary": metrics_summary,
            "kg_summary": kg_summary,
            "disruption_summary": disruption_analysis,
            "methodology": self._methodology_section(),
            "recommendation": self._get_executive_recommendation(counts),
        }

def generate_all_ground_truth(scenarios_file: str, output_location: str):
    """Generate ground truth for all scenarios.

    output_location can be either a JSON file path (legacy aggregated format)
    or a directory path (preferred) where one JSON file per scenario will be written.
    """
    load_dotenv()
    
    # Load scenarios
    with open(scenarios_file, 'r') as f:
        scenarios = json.load(f)
    
    output_path = Path(output_location)
    ground_truths = {}
    
    if output_path.suffix.lower() == ".json":
        # Legacy aggregated format
        for scenario in scenarios:
            # Create generator with scenario's monitored company
            monitored_company = scenario.get("monitored_company", "Tesla Inc")
            generator = GroundTruthGenerator(monitored_company=monitored_company)
            gt = generator.generate_ground_truth(scenario)
            ground_truths[scenario["scenario_id"]] = gt
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(ground_truths, f, indent=2, default=str)
        
        logger.info(f"✅ Generated ground truth for {len(ground_truths)} scenarios")
        logger.info(f"   Saved aggregated file to: {output_path}")
    else:
        # Preferred directory-per-scenario format
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        index = {}
        
        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            # Create generator with scenario's monitored company
            monitored_company = scenario.get("monitored_company", "Tesla Inc")
            generator = GroundTruthGenerator(monitored_company=monitored_company)
            gt = generator.generate_ground_truth(scenario)
            scenario_file = output_dir / f"{scenario_id}.json"
            with open(scenario_file, 'w') as f:
                json.dump(gt, f, indent=2, default=str)
            index[scenario_id] = scenario_file.name
            logger.info(f"   ↳ Wrote {scenario_file} (company: {monitored_company})")
        
        # Write index file for quick lookup
        index_file = output_dir / "index.json"
        with open(index_file, 'w') as f:
            json.dump({"scenarios": index, "generated_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
        
        logger.info(f"✅ Generated ground truth for {len(index)} scenarios")
        logger.info(f"   Saved scenario files to directory: {output_dir}")

if __name__ == "__main__":
    generate_all_ground_truth(
        "evaluation/synthesized_scenarios.json",
        "evaluation/ground_truth"
    )

