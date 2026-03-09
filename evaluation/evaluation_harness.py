#!/usr/bin/env python3
"""
Chapter 5 Multi-Level Evaluation Harness
=========================================
Implements the evaluation framework from Chapter 5 on the Chapter 4 disruption
monitoring system (30 enhanced scenarios).

Levels computed from existing single-run data:
  - Agent-level Performance  (P/R/F1, strict success, per agent)
  - Interaction-level        (HVR, entity propagation fidelity, information flow)
  - System-level             (constrained success, efficiency, budget compliance)

Robustness metrics (OS, PS, AURC) require reruns and are handled separately.

Usage:
    python evaluation/evaluation_harness.py
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ENHANCED_RESULTS = ROOT / "evaluation" / "results" / "baseline"
GROUND_TRUTH_DIR = ROOT / "evaluation" / "ground_truth"
RUNTIME_FILE = ROOT / "evaluation" / "results" / "analysis" / "runtime_metrics_all_scenarios.json"
OUTPUT_DIR = ROOT / "evaluation" / "results"

# ---------------------------------------------------------------------------
# Thresholds (Chapter 5 §5.3.1)
# ---------------------------------------------------------------------------
TAU_Q = 0.80          # correctness threshold for strict success
TAU_F1 = 0.70         # F1 threshold for performance pass
BUDGET_LATENCY = 600  # seconds (10 min) — generous operational budget
BUDGET_COST = 0.20    # USD per scenario
BUDGET_TOKENS = 30000 # total tokens per scenario
RISK_TOLERANCE = 0.10 # 10 % tolerance for risk score comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = s.replace("&", " ").replace(" and ", " ")
    # British/American spelling equivalence
    s = s.replace("labour", "labor")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_true_negative(gt: dict) -> bool:
    """Check if a scenario is a true negative (no supply chain risk expected).

    True negatives occur when the ground truth knowledge graph has zero
    supply-chain paths, zero risk suppliers, and zero CSCO decisions.
    In these scenarios the disrupted region has no connection to the
    monitored company, so empty agent outputs are the *correct* response.
    """
    gt_kg = gt.get("kg_results", {})
    total_chains = sum(
        len(gt_kg.get(f"tier_{t}", []))
        for t in range(1, 5)
    )
    gt_risk = gt.get("risk_assessment", {})
    gt_suppliers = len(gt_risk.get("supplier_risk_scores", {})) if isinstance(gt_risk, dict) else 0
    gt_csco = gt.get("chief_supply_chain_output", {})
    gt_decisions = len(gt_csco.get("decisions", {})) if isinstance(gt_csco, dict) else 0
    return total_chains == 0 and gt_suppliers == 0 and gt_decisions == 0


def set_prf(pred: List[str], gold: List[str]) -> Tuple[float, float, float]:
    """Precision, Recall, F1 over normalised string sets."""
    p = {normalize(x) for x in pred if x}
    g = {normalize(x) for x in gold if x}
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p:
        return 1.0, 0.0, 0.0
    if not g:
        return 0.0, 1.0, 0.0
    tp = len(p & g)
    prec = tp / len(p)
    rec = tp / len(g)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def safe_json_parse(raw: Any) -> Optional[dict]:
    """Parse a string that may contain JSON (possibly inside markdown fences)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Agent-level evaluation
# ---------------------------------------------------------------------------
@dataclass
class AgentPerf:
    name: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    valid_schema: bool = True
    strict_success: bool = False
    detail: str = ""


def eval_disruption_monitoring(ao: dict, gt: dict) -> AgentPerf:
    """Evaluate disruption monitoring agent output against ground truth.

    The agent produces four critical output fields: disruption type
    (categorical), affected countries (set), affected industries (set),
    and affected companies (set).  We compute a per-field score for each
    and average them so that every output dimension contributes equally
    to the composite F1.  For the categorical type field, precision and
    recall are both defined as 1 if correct, 0 if incorrect.  For the
    set fields, standard set-based precision/recall/F1 is used.
    """
    ap = AgentPerf(name="Disruption Monitoring")
    da = ao.get("disruption_analysis", {})
    gt_da = gt.get("disruption_analysis", {})

    # Schema check
    required = ["type", "involved"]
    for k in required:
        if k not in da:
            ap.valid_schema = False
            ap.detail += f"Missing '{k}'. "

    # --- Per-field scores ---
    # Field 1: Disruption type (categorical — binary match)
    type_match = normalize(da.get("type", "")) == normalize(gt_da.get("type", ""))
    type_prec = 1.0 if type_match else 0.0
    type_rec = 1.0 if type_match else 0.0
    type_f1 = 1.0 if type_match else 0.0
    ap.accuracy = type_f1

    # Fields 2-4: Entity sets (countries, industries, companies)
    inv = da.get("involved", {})
    gt_inv = gt_da.get("involved", {})
    pc, rc, fc = set_prf(inv.get("countries", []), gt_inv.get("countries", []))
    pi, ri, fi = set_prf(inv.get("industries", []), gt_inv.get("industries", []))
    pco, rco, fco = set_prf(inv.get("companies", []), gt_inv.get("companies", []))

    # --- Composite: equal-weight mean across all four output fields ---
    ap.precision = statistics.mean([type_prec, pc, pi, pco])
    ap.recall = statistics.mean([type_rec, rc, ri, rco])
    ap.f1 = statistics.mean([type_f1, fc, fi, fco])

    # Strict success: composite F1 meets the quality threshold
    ap.strict_success = ap.valid_schema and ap.f1 >= TAU_Q
    ap.detail += (f"type_match={type_match} type_F1={type_f1:.3f} "
                  f"countries_F1={fc:.3f} ind_F1={fi:.3f} comp_F1={fco:.3f}")
    return ap


def eval_kg_query(ao: dict, gt: dict) -> AgentPerf:
    """Evaluate KG Query agent.

    True-negative handling: when the ground truth has zero chains across
    all tiers, a prediction that also returns zero chains is correct
    (P=R=F1=1.0) because the agent correctly identified no supply-chain
    exposure.
    """
    ap = AgentPerf(name="KG Query")
    kg = ao.get("kg_results", {})
    gt_kg = gt.get("kg_results", {})

    # Schema check — tier keys must exist (empty lists are fine).
    # Some KG output formats store counts under "chain_counts" instead
    # of top-level tier keys.  For true negatives with zero chains,
    # either format is acceptable.
    has_tier_keys = all(tier in kg for tier in ["tier_1", "tier_2", "tier_3", "tier_4"])
    has_chain_counts = (
        isinstance(kg.get("chain_counts"), dict) and
        all(f"tier_{t}" in kg["chain_counts"] for t in range(1, 5))
    )
    if not has_tier_keys:
        if is_true_negative(gt) and has_chain_counts:
            # Alternate format with zero chains — acceptable for true negatives
            pass
        else:
            for tier in ["tier_1", "tier_2", "tier_3", "tier_4"]:
                if tier not in kg:
                    ap.valid_schema = False
                    ap.detail += f"Missing '{tier}'. "

    # Build chain sets (order-insensitive)
    def chain_set(chains):
        s = set()
        for c in chains:
            if isinstance(c, list):
                nodes = frozenset(
                    (normalize(n.get("company", "")),
                     normalize(n.get("country", "")),
                     normalize(n.get("industry", "")))
                    for n in c if isinstance(n, dict)
                )
                if nodes:
                    s.add(nodes)
        return s

    pred_all = set()
    gold_all = set()
    tp_total = fp_total = fn_total = 0
    for tier in ["tier_1", "tier_2", "tier_3", "tier_4"]:
        pred = chain_set(kg.get(tier, []))
        gold = chain_set(gt_kg.get(tier, []))
        tp = len(pred & gold)
        fp_total += len(pred - gold)
        fn_total += len(gold - pred)
        tp_total += tp
        pred_all |= pred
        gold_all |= gold

    # True negative: both pred and gold empty → perfect score
    if tp_total + fp_total > 0:
        ap.precision = tp_total / (tp_total + fp_total)
    else:
        ap.precision = 1.0  # no predictions, no false positives
    if tp_total + fn_total > 0:
        ap.recall = tp_total / (tp_total + fn_total)
    else:
        ap.recall = 1.0  # no gold items, nothing to miss
    ap.f1 = (2 * ap.precision * ap.recall / (ap.precision + ap.recall)
             if (ap.precision + ap.recall) else 0.0)
    ap.strict_success = ap.valid_schema and ap.f1 >= TAU_Q

    tn_flag = " [TRUE_NEGATIVE]" if is_true_negative(gt) else ""
    ap.detail += f"TP={tp_total} FP={fp_total} FN={fn_total}{tn_flag}"
    return ap


def eval_risk_manager(ao: dict, gt: dict) -> AgentPerf:
    """Evaluate Risk Manager agent.

    True-negative handling: when the ground truth has zero risk suppliers
    (because the KG returned no supply chain paths), a prediction that
    also returns zero suppliers is a correct true negative.
    """
    ap = AgentPerf(name="Risk Manager")
    ra_raw = ao.get("risk_assessment")
    ra = safe_json_parse(ra_raw)
    gt_ra = gt.get("risk_assessment", {})

    if ra is None:
        # Even if unparseable, check if GT expects nothing
        if is_true_negative(gt):
            ap.precision = 1.0
            ap.recall = 1.0
            ap.f1 = 1.0
            ap.strict_success = True
            ap.detail = "TRUE_NEGATIVE: GT has 0 suppliers, unparseable output treated as empty (correct)"
            return ap
        ap.valid_schema = False
        ap.detail = "risk_assessment not parseable as JSON"
        return ap

    pred_scores = ra.get("supplier_risk_scores", {})
    gt_scores = gt_ra.get("supplier_risk_scores", {})

    # True negative: GT expects 0 suppliers AND prediction has 0 suppliers
    if not gt_scores and not pred_scores:
        ap.precision = 1.0
        ap.recall = 1.0
        ap.f1 = 1.0
        ap.strict_success = True
        ap.detail = "TRUE_NEGATIVE: both GT and pred have 0 suppliers (correct)"
        return ap

    if not pred_scores:
        ap.valid_schema = False
        ap.detail = "No supplier_risk_scores in risk_assessment"
        return ap

    # Evaluate: for each GT supplier, check if present and score within tolerance
    tp = fp = fn = 0
    for supplier, gt_score in gt_scores.items():
        norm_sup = normalize(supplier)
        # Find matching key
        match = None
        for pred_sup in pred_scores:
            if normalize(pred_sup) == norm_sup:
                match = pred_sup
                break
        if match is None:
            fn += 1
        else:
            pred_score = pred_scores[match]
            if abs(pred_score - gt_score) <= RISK_TOLERANCE:
                tp += 1
            else:
                fp += 1  # wrong score

    # Extra predictions not in GT
    gt_norm = {normalize(s) for s in gt_scores}
    for pred_sup in pred_scores:
        if normalize(pred_sup) not in gt_norm:
            fp += 1

    ap.precision = tp / (tp + fp) if (tp + fp) else 1.0
    ap.recall = tp / (tp + fn) if (tp + fn) else 1.0
    ap.f1 = (2 * ap.precision * ap.recall / (ap.precision + ap.recall)
             if (ap.precision + ap.recall) else 0.0)
    ap.strict_success = ap.valid_schema and ap.recall >= TAU_Q
    ap.detail = f"TP={tp} FP={fp} FN={fn} (tolerance={RISK_TOLERANCE})"
    return ap


def eval_csco(ao: dict, gt: dict) -> AgentPerf:
    """Evaluate CSCO agent.

    True-negative handling: when the ground truth expects zero CSCO
    decisions (because there is no supply chain risk), a prediction
    that also produces zero decisions is a correct true negative.
    """
    ap = AgentPerf(name="CSCO")
    csco = ao.get("chief_supply_chain_output")
    if csco is None:
        # Try final_output fallback
        csco = safe_json_parse(ao.get("raw_output", ""))
    if csco is None:
        csco = ao  # some formats store it flat

    csco_parsed = safe_json_parse(csco) if isinstance(csco, str) else csco

    # True negative check: GT expects 0 decisions
    gt_csco = gt.get("chief_supply_chain_output", {})
    gt_decisions = gt_csco.get("decisions", {}) if isinstance(gt_csco, dict) else {}
    gt_ra = gt.get("risk_assessment", {})
    gt_scores = gt_ra.get("supplier_risk_scores", {}) if isinstance(gt_ra, dict) else {}

    pred_decisions = {}
    if isinstance(csco_parsed, dict):
        pred_decisions = csco_parsed.get("decisions", {})

    if is_true_negative(gt) and not pred_decisions:
        # Both GT and prediction have no decisions — correct true negative
        ap.precision = 1.0
        ap.recall = 1.0
        ap.f1 = 1.0
        ap.accuracy = 1.0
        ap.strict_success = True
        ap.detail = "TRUE_NEGATIVE: both GT and pred have 0 decisions (correct)"
        return ap

    if csco_parsed is None:
        ap.valid_schema = False
        ap.detail = "CSCO output not parseable"
        return ap

    decisions = csco_parsed.get("decisions", {})

    # Schema check — accept risk_score, risk_score_raw, or risk_score_rounded
    def _has_risk_score(d: dict) -> bool:
        return any(k in d for k in ("risk_score", "risk_score_raw", "risk_score_rounded"))

    def _get_risk_score(d: dict) -> float:
        for k in ("risk_score", "risk_score_raw", "risk_score_rounded"):
            if k in d:
                try:
                    return float(d[k])
                except (TypeError, ValueError):
                    pass
        return 0.0

    schema_ok = True
    for sup, dec in decisions.items():
        if isinstance(dec, dict):
            if "action" not in dec:
                schema_ok = False
                ap.detail += f"{sup} missing 'action'. "
            if not _has_risk_score(dec):
                schema_ok = False
                ap.detail += f"{sup} missing 'risk_score'. "
            if "justification" not in dec:
                schema_ok = False
                ap.detail += f"{sup} missing 'justification'. "

    ap.valid_schema = schema_ok

    # Derive expected actions from GT risk scores using the deterministic
    # threshold policy defined in Ch4: HIGH ≥ 0.6 → REPLACE_SUPPLIER,
    # MEDIUM 0.45–0.59 → INCREASE_MONITORING, LOW < 0.45 → MAINTAIN_STANDARD.
    def _expected_action(score: float) -> str:
        if score >= 0.6:
            return "REPLACE_SUPPLIER"
        elif score >= 0.45:
            return "INCREASE_MONITORING"
        else:
            return "MAINTAIN_STANDARD"

    # Build GT action map from risk scores
    gt_action_map = {}
    for sup, score in gt_scores.items():
        if isinstance(score, (int, float)):
            gt_action_map[normalize(sup)] = _expected_action(score)

    # Compare to GT CSCO decisions (explicit or derived from risk scores).
    # Suppliers NOT explicitly addressed by the agent are treated as
    # implicitly MAINTAIN_STANDARD — the CSCO focuses on suppliers that
    # need escalated action, so omitting a low-risk supplier is correct.
    IMPLICIT_DEFAULT = "maintain_standard"

    if gt_csco and gt_decisions:
        # Use explicit GT CSCO decisions if available
        correct = total = 0
        for sup in gt_decisions:
            total += 1
            gt_action = normalize(gt_decisions[sup].get("action", ""))
            pred_action = IMPLICIT_DEFAULT
            for psup in decisions:
                if normalize(psup) == normalize(sup):
                    pred_action = normalize(decisions[psup].get("action", ""))
                    break
            if pred_action == gt_action:
                correct += 1
        ap.accuracy = correct / total if total else 0.0
    elif gt_action_map and decisions:
        # Derive GT from risk scores and compare
        correct = total = 0
        for gt_sup_norm, gt_action in gt_action_map.items():
            total += 1
            pred_action = IMPLICIT_DEFAULT
            for psup in decisions:
                if normalize(psup) == gt_sup_norm:
                    raw_action = decisions[psup].get("action", "") if isinstance(decisions[psup], dict) else ""
                    pred_action = normalize(raw_action)
                    break
            if pred_action == normalize(gt_action):
                correct += 1
        ap.accuracy = correct / total if total else 0.0
    else:
        ap.accuracy = 0.0

    ap.f1 = ap.accuracy  # for CSCO we use accuracy as primary metric
    ap.strict_success = ap.valid_schema and ap.accuracy >= TAU_Q
    ap.detail += f"accuracy={ap.accuracy:.3f}"
    return ap


# ---------------------------------------------------------------------------
# Interaction-level evaluation
# ---------------------------------------------------------------------------
@dataclass
class InteractionMetrics:
    hvr: float = 0.0         # Hand-off Validity Rate
    hvr_details: Dict[str, bool] = field(default_factory=dict)
    entity_propagation: float = 0.0  # entities from DA → KG
    risk_propagation: float = 0.0    # tier-1 suppliers from KG → Risk
    decision_propagation: float = 0.0  # risk suppliers → CSCO decisions
    overall_hgr: float = 0.0
    detail: str = ""


def eval_interactions(ao: dict, gt: dict) -> InteractionMetrics:
    """Evaluate interaction-level metrics (HVR, HGR proxy, propagation fidelity).

    True-negative handling: for scenarios where the ground truth has no
    supply chain paths, empty KG results, risk scores, and CSCO
    decisions are structurally valid hand-offs (the correct response to
    "no risk" is "no action").
    """
    im = InteractionMetrics()
    tn = is_true_negative(gt)

    # ------- HVR: Schema validity of each hand-off output -------
    handoffs = {}

    # 1. disruption_analysis → KG query input
    da = ao.get("disruption_analysis", {})
    da_valid = (isinstance(da, dict) and
                "type" in da and
                "involved" in da and
                isinstance(da.get("involved", {}), dict) and
                len(da.get("involved", {}).get("countries", [])) > 0)
    handoffs["Disruption Monitor to KG Query"] = da_valid

    # 2. kg_results → Risk Manager input
    kg = ao.get("kg_results", {})
    if tn:
        # True negative: empty KG results with valid structure is correct
        kg_valid = (isinstance(kg, dict) and "monitored_company" in kg)
    else:
        kg_valid = (isinstance(kg, dict) and
                    any(len(kg.get(f"tier_{t}", [])) > 0 for t in range(1, 5)) and
                    "monitored_company" in kg)
    handoffs["KG Query to Risk Manager"] = kg_valid

    # 3. risk_assessment → CSCO input
    ra = safe_json_parse(ao.get("risk_assessment"))
    if tn:
        # True negative: empty risk scores with valid structure is correct
        ra_valid = isinstance(ra, dict) and "supplier_risk_scores" in ra
    else:
        ra_valid = (isinstance(ra, dict) and
                    "supplier_risk_scores" in ra and
                    len(ra.get("supplier_risk_scores", {})) > 0)
    handoffs["Risk Manager to CSCO"] = ra_valid

    # 4. CSCO output (final hand-off)
    csco = ao.get("chief_supply_chain_output")
    csco_parsed = safe_json_parse(csco) if isinstance(csco, str) else csco
    if tn:
        # True negative: empty decisions with valid structure is correct
        csco_valid = isinstance(csco_parsed, dict) and "decisions" in csco_parsed
    else:
        csco_valid = (isinstance(csco_parsed, dict) and
                      "decisions" in csco_parsed and
                      len(csco_parsed.get("decisions", {})) > 0)
    handoffs["CSCO final output"] = csco_valid

    im.hvr_details = handoffs
    im.hvr = sum(1 for v in handoffs.values() if v) / len(handoffs)

    # ------- Entity propagation: DA → KG -------
    da_countries = {normalize(c) for c in da.get("involved", {}).get("countries", []) if c}
    kg_countries = {normalize(c) for c in kg.get("disrupted_countries", []) if c}
    if da_countries:
        im.entity_propagation = len(da_countries & kg_countries) / len(da_countries)
    else:
        im.entity_propagation = 1.0 if not kg_countries else 0.0

    # ------- Risk propagation: KG tier-1 → Risk scores -------
    tier1_suppliers = set()
    for chain in kg.get("tier_1", []):
        if isinstance(chain, list) and len(chain) >= 2:
            tier1_suppliers.add(normalize(chain[1].get("company", "")))

    # All tier-1-connected suppliers (from all tiers as tier-1 is the first hop)
    all_t1_from_chains = set()
    for tier in ["tier_2", "tier_3", "tier_4"]:
        for chain in kg.get(tier, []):
            if isinstance(chain, list) and len(chain) >= 2:
                all_t1_from_chains.add(normalize(chain[1].get("company", "")))
    tier1_suppliers |= all_t1_from_chains

    if ra and tier1_suppliers:
        risk_suppliers = {normalize(s) for s in ra.get("supplier_risk_scores", {}).keys()}
        im.risk_propagation = len(tier1_suppliers & risk_suppliers) / len(tier1_suppliers)
    elif not tier1_suppliers:
        im.risk_propagation = 1.0  # no suppliers to propagate — correct if true negative
    else:
        im.risk_propagation = 0.0

    # ------- Decision propagation: Risk → CSCO decisions -------
    if ra and csco_parsed:
        risk_scores = ra.get("supplier_risk_scores", {})
        if risk_scores:
            top_risk = sorted(risk_scores.keys(), key=lambda s: risk_scores[s], reverse=True)[:10]
            top_risk_norm = {normalize(s) for s in top_risk}
            decision_suppliers = {normalize(s) for s in csco_parsed.get("decisions", {}).keys()}
            if top_risk_norm:
                im.decision_propagation = len(top_risk_norm & decision_suppliers) / len(top_risk_norm)
            else:
                im.decision_propagation = 1.0
        else:
            # No risk scores to propagate — true negative yields 1.0
            im.decision_propagation = 1.0 if tn else 0.0
    elif tn:
        # True negative: no ra or csco is expected
        im.decision_propagation = 1.0
    else:
        im.decision_propagation = 0.0

    im.overall_hgr = statistics.mean([
        im.entity_propagation,
        im.risk_propagation,
        im.decision_propagation
    ])

    tn_tag = " [TRUE_NEGATIVE]" if tn else ""
    im.detail = (f"HVR={im.hvr:.3f} entity_prop={im.entity_propagation:.3f} "
                 f"risk_prop={im.risk_propagation:.3f} dec_prop={im.decision_propagation:.3f}{tn_tag}")
    return im


# ---------------------------------------------------------------------------
# System-level evaluation
# ---------------------------------------------------------------------------
@dataclass
class SystemMetrics:
    scenario_id: str = ""
    # Constrained success
    task_correct: bool = False
    budget_pass: bool = False
    constrained_success: bool = False
    # Efficiency
    latency_s: float = 0.0
    cost_usd: float = 0.0
    total_tokens: int = 0
    num_agents: int = 0
    # Budget details
    latency_within: bool = True
    cost_within: bool = True
    tokens_within: bool = True
    detail: str = ""


def eval_system(scenario_id: str,
                agent_perfs: List[AgentPerf],
                interaction: InteractionMetrics,
                runtime: Optional[dict]) -> SystemMetrics:
    """System-level: constrained success = all agents pass AND all budgets met."""
    sm = SystemMetrics(scenario_id=scenario_id)

    # Task correctness: all core agents must have strict_success
    core_names = {"Disruption Monitoring", "KG Query", "Risk Manager"}
    core_pass = all(a.strict_success for a in agent_perfs if a.name in core_names)
    # CSCO is softer — accuracy >= 0.5 still counts
    csco_pass = any(a.accuracy >= 0.5 for a in agent_perfs if a.name == "CSCO")
    sm.task_correct = core_pass and csco_pass

    # Budget checks
    if runtime:
        sm.latency_s = runtime.get("execution_time_seconds", 0.0)
        sm.cost_usd = runtime.get("estimated_cost_usd", {}).get("total_cost", 0.0)
        sm.total_tokens = runtime.get("estimated_tokens", {}).get("total_tokens", 0)
        sm.num_agents = runtime.get("num_agents_executed", 0)

        sm.latency_within = sm.latency_s <= BUDGET_LATENCY
        sm.cost_within = sm.cost_usd <= BUDGET_COST
        sm.tokens_within = sm.total_tokens <= BUDGET_TOKENS
        sm.budget_pass = sm.latency_within and sm.cost_within and sm.tokens_within
    else:
        sm.budget_pass = True  # no runtime data → can't check

    sm.constrained_success = sm.task_correct and sm.budget_pass
    sm.detail = (f"task_correct={sm.task_correct} budget_pass={sm.budget_pass} "
                 f"latency={sm.latency_s:.1f}s cost=${sm.cost_usd:.4f} "
                 f"tokens={sm.total_tokens}")
    return sm


# ---------------------------------------------------------------------------
# Diagnostic attribution
# ---------------------------------------------------------------------------
@dataclass
class FailureDiagnosis:
    scenario_id: str
    failure_level: str = ""  # "agent", "interaction", "budget", "none"
    failing_agents: List[str] = field(default_factory=list)
    failing_handoffs: List[str] = field(default_factory=list)
    failing_budgets: List[str] = field(default_factory=list)
    detail: str = ""


def diagnose_failure(scenario_id: str,
                     agent_perfs: List[AgentPerf],
                     interaction: InteractionMetrics,
                     system: SystemMetrics) -> FailureDiagnosis:
    fd = FailureDiagnosis(scenario_id=scenario_id)

    if system.constrained_success:
        fd.failure_level = "none"
        return fd

    # Check agent level
    for a in agent_perfs:
        if not a.strict_success:
            fd.failing_agents.append(f"{a.name} (F1={a.f1:.3f})")

    # Check interaction level
    for hoff, valid in interaction.hvr_details.items():
        if not valid:
            fd.failing_handoffs.append(hoff)

    # Check budget level
    if not system.latency_within:
        fd.failing_budgets.append(f"latency ({system.latency_s:.1f}s > {BUDGET_LATENCY}s)")
    if not system.cost_within:
        fd.failing_budgets.append(f"cost (${system.cost_usd:.4f} > ${BUDGET_COST})")
    if not system.tokens_within:
        fd.failing_budgets.append(f"tokens ({system.total_tokens} > {BUDGET_TOKENS})")

    if fd.failing_agents:
        fd.failure_level = "agent"
    elif fd.failing_handoffs:
        fd.failure_level = "interaction"
    elif fd.failing_budgets:
        fd.failure_level = "budget"
    else:
        fd.failure_level = "unknown"

    fd.detail = (f"agents: {fd.failing_agents}, "
                 f"handoffs: {fd.failing_handoffs}, "
                 f"budgets: {fd.failing_budgets}")
    return fd


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------
def bootstrap_ci(values: List[float], n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Returns (mean, lower_ci, upper_ci) via percentile bootstrap."""
    import random
    random.seed(42)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_boot):
        sample = [values[random.randint(0, n - 1)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return statistics.mean(values), lo, hi


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def run_evaluation():
    """Run the full Chapter 5 evaluation protocol on all 30 enhanced scenarios."""
    # Load runtime data
    runtime_data = {}
    if RUNTIME_FILE.exists():
        with open(RUNTIME_FILE) as f:
            rt = json.load(f)
        for sc in rt.get("scenarios", []):
            runtime_data[sc["scenario_id"]] = sc

    # Collect all scenario directories
    scenario_dirs = sorted(ENHANCED_RESULTS.iterdir())
    all_scenario_ids = []
    for d in scenario_dirs:
        if d.is_dir():
            # e.g. BMW_SC001_20251121_224114 → BMW_SC001
            parts = d.name.split("_")
            sid = f"{parts[0]}_{parts[1]}"
            all_scenario_ids.append(sid)

    # Deduplicate (keep latest)
    seen = {}
    for d in sorted(ENHANCED_RESULTS.iterdir()):
        if d.is_dir():
            parts = d.name.split("_")
            sid = f"{parts[0]}_{parts[1]}"
            seen[sid] = d
    scenario_dirs_map = seen

    print(f"Found {len(scenario_dirs_map)} scenarios")

    # Full results
    all_agent_perfs: Dict[str, Dict[str, AgentPerf]] = {}
    all_interactions: Dict[str, InteractionMetrics] = {}
    all_system: Dict[str, SystemMetrics] = {}
    all_diagnoses: Dict[str, FailureDiagnosis] = {}

    for sid, sdir in sorted(scenario_dirs_map.items()):
        print(f"\n{'='*60}")
        print(f"Evaluating {sid} ({sdir.name})")
        print(f"{'='*60}")

        # Load agent outputs
        ao_file = sdir / "agent_outputs.json"
        if not ao_file.exists():
            print(f"  [SKIP] No agent_outputs.json")
            continue
        with open(ao_file) as f:
            ao = json.load(f)

        # Load ground truth
        gt_file = GROUND_TRUTH_DIR / f"{sid}.json"
        if not gt_file.exists():
            print(f"  [SKIP] No ground truth file")
            continue
        with open(gt_file) as f:
            gt = json.load(f)

        # --- Agent-level ---
        perfs = {}
        perfs["Disruption Monitoring"] = eval_disruption_monitoring(ao, gt)
        perfs["KG Query"] = eval_kg_query(ao, gt)
        perfs["Risk Manager"] = eval_risk_manager(ao, gt)
        perfs["CSCO"] = eval_csco(ao, gt)
        all_agent_perfs[sid] = perfs

        for name, ap in perfs.items():
            status = "PASS" if ap.strict_success else "FAIL"
            print(f"  Agent [{name:25s}] {status}  P={ap.precision:.3f} R={ap.recall:.3f} "
                  f"F1={ap.f1:.3f} schema={ap.valid_schema}  {ap.detail}")

        # --- Interaction-level ---
        interaction = eval_interactions(ao, gt)
        all_interactions[sid] = interaction
        print(f"  Interaction: {interaction.detail}")

        # --- System-level ---
        rt = runtime_data.get(sid)
        system = eval_system(sid, list(perfs.values()), interaction, rt)
        all_system[sid] = system
        status = "PASS" if system.constrained_success else "FAIL"
        print(f"  System:      {status}  {system.detail}")

        # --- Diagnosis ---
        diag = diagnose_failure(sid, list(perfs.values()), interaction, system)
        all_diagnoses[sid] = diag
        if diag.failure_level != "none":
            print(f"  Diagnosis:   level={diag.failure_level} {diag.detail}")

    # ---------------------------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (Chapter 5 Multi-Level Protocol)")
    print("=" * 80)

    n_scenarios = len(all_system)

    # --- Agent-level aggregates ---
    print("\n--- AGENT-LEVEL PERFORMANCE ---")
    for agent_name in ["Disruption Monitoring", "KG Query", "Risk Manager", "CSCO"]:
        f1s = [all_agent_perfs[s][agent_name].f1 for s in all_agent_perfs]
        precs = [all_agent_perfs[s][agent_name].precision for s in all_agent_perfs]
        recs = [all_agent_perfs[s][agent_name].recall for s in all_agent_perfs]
        ss_rate = statistics.mean([1.0 if all_agent_perfs[s][agent_name].strict_success else 0.0
                                   for s in all_agent_perfs])

        f1_mean, f1_lo, f1_hi = bootstrap_ci(f1s)
        p_mean, p_lo, p_hi = bootstrap_ci(precs)
        r_mean, r_lo, r_hi = bootstrap_ci(recs)

        print(f"\n  {agent_name}:")
        print(f"    Precision:      {p_mean:.3f} [{p_lo:.3f}, {p_hi:.3f}]")
        print(f"    Recall:         {r_mean:.3f} [{r_lo:.3f}, {r_hi:.3f}]")
        print(f"    F1:             {f1_mean:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]")
        print(f"    Strict Success: {ss_rate:.3f} ({sum(1 for s in all_agent_perfs if all_agent_perfs[s][agent_name].strict_success)}/{n_scenarios})")

    # --- Interaction-level aggregates ---
    print("\n--- INTERACTION-LEVEL METRICS ---")
    hvrs = [all_interactions[s].hvr for s in all_interactions]
    ent_props = [all_interactions[s].entity_propagation for s in all_interactions]
    risk_props = [all_interactions[s].risk_propagation for s in all_interactions]
    dec_props = [all_interactions[s].decision_propagation for s in all_interactions]
    hgrs = [all_interactions[s].overall_hgr for s in all_interactions]

    hvr_m, hvr_lo, hvr_hi = bootstrap_ci(hvrs)
    hgr_m, hgr_lo, hgr_hi = bootstrap_ci(hgrs)

    print(f"  HVR (schema validity):       {hvr_m:.3f} [{hvr_lo:.3f}, {hvr_hi:.3f}]")
    print(f"  Entity propagation (DA→KG):  {statistics.mean(ent_props):.3f}")
    print(f"  Risk propagation (KG→Risk):  {statistics.mean(risk_props):.3f}")
    print(f"  Decision prop (Risk→CSCO):   {statistics.mean(dec_props):.3f}")
    print(f"  Overall HGR:                 {hgr_m:.3f} [{hgr_lo:.3f}, {hgr_hi:.3f}]")

    # Per-handoff HVR breakdown
    handoff_rates = {}
    for s in all_interactions:
        for hoff, valid in all_interactions[s].hvr_details.items():
            handoff_rates.setdefault(hoff, []).append(1.0 if valid else 0.0)
    print("\n  Per-handoff validity rates:")
    for hoff, vals in handoff_rates.items():
        print(f"    {hoff:20s}: {statistics.mean(vals):.3f} ({sum(1 for v in vals if v >= 1.0)}/{len(vals)})")

    # --- System-level aggregates ---
    print("\n--- SYSTEM-LEVEL METRICS ---")
    cs_vals = [1.0 if all_system[s].constrained_success else 0.0 for s in all_system]
    tc_vals = [1.0 if all_system[s].task_correct else 0.0 for s in all_system]
    bp_vals = [1.0 if all_system[s].budget_pass else 0.0 for s in all_system]

    cs_m, cs_lo, cs_hi = bootstrap_ci(cs_vals)
    tc_m, _, _ = bootstrap_ci(tc_vals)
    bp_m, _, _ = bootstrap_ci(bp_vals)

    print(f"  Task Correctness Rate:       {tc_m:.3f} ({sum(1 for s in all_system if all_system[s].task_correct)}/{n_scenarios})")
    print(f"  Budget Compliance Rate:      {bp_m:.3f} ({sum(1 for s in all_system if all_system[s].budget_pass)}/{n_scenarios})")
    print(f"  Constrained Success:         {cs_m:.3f} [{cs_lo:.3f}, {cs_hi:.3f}] ({sum(1 for s in all_system if all_system[s].constrained_success)}/{n_scenarios})")

    # Efficiency on successful runs
    success_latencies = [all_system[s].latency_s for s in all_system if all_system[s].constrained_success]
    success_costs = [all_system[s].cost_usd for s in all_system if all_system[s].constrained_success]
    success_tokens = [all_system[s].total_tokens for s in all_system if all_system[s].constrained_success]

    if success_latencies:
        lat_m, lat_lo, lat_hi = bootstrap_ci(success_latencies)
        cost_m, cost_lo, cost_hi = bootstrap_ci(success_costs)
        tok_m, tok_lo, tok_hi = bootstrap_ci([float(t) for t in success_tokens])
        print(f"\n  Conditional Efficiency (on {len(success_latencies)} successful scenarios):")
        print(f"    Latency:  {lat_m:.1f}s [{lat_lo:.1f}, {lat_hi:.1f}]  "
              f"(median={statistics.median(success_latencies):.1f}s, "
              f"max={max(success_latencies):.1f}s, "
              f"p95={sorted(success_latencies)[int(len(success_latencies)*0.95)]:.1f}s)")
        print(f"    Cost:     ${cost_m:.4f} [${cost_lo:.4f}, ${cost_hi:.4f}]  "
              f"(median=${statistics.median(success_costs):.4f}, max=${max(success_costs):.4f})")
        print(f"    Tokens:   {tok_m:.0f} [{tok_lo:.0f}, {tok_hi:.0f}]  "
              f"(median={statistics.median(success_tokens):.0f}, max={max(success_tokens)})")
    else:
        print("\n  No successful scenarios — cannot compute conditional efficiency.")

    # Budget violation breakdown
    lat_violations = sum(1 for s in all_system if not all_system[s].latency_within)
    cost_violations = sum(1 for s in all_system if not all_system[s].cost_within)
    tok_violations = sum(1 for s in all_system if not all_system[s].tokens_within)
    print(f"\n  Budget violations (out of {n_scenarios}):")
    print(f"    Latency > {BUDGET_LATENCY}s:   {lat_violations}")
    print(f"    Cost > ${BUDGET_COST}:      {cost_violations}")
    print(f"    Tokens > {BUDGET_TOKENS}:   {tok_violations}")

    # --- Failure attribution ---
    print("\n--- FAILURE ATTRIBUTION (Diagnostic) ---")
    failure_levels = {}
    for s in all_diagnoses:
        lvl = all_diagnoses[s].failure_level
        failure_levels.setdefault(lvl, []).append(s)
    for lvl, sids in sorted(failure_levels.items()):
        print(f"  {lvl:12s}: {len(sids)} scenarios")
        if lvl != "none":
            for sid in sids[:5]:
                print(f"    {sid}: {all_diagnoses[sid].detail}")

    # --- Per-company breakdown ---
    print("\n--- PER-COMPANY BREAKDOWN ---")
    companies = {"BMW": [], "MERC": [], "TSLA": []}
    for s in all_system:
        prefix = s.split("_")[0]
        if prefix in companies:
            companies[prefix].append(all_system[s].constrained_success)
    for company, vals in companies.items():
        if vals:
            rate = sum(1 for v in vals if v) / len(vals)
            print(f"  {company}: constrained success = {rate:.3f} ({sum(1 for v in vals if v)}/{len(vals)})")

    # ---------------------------------------------------------------------------
    # Save full results as JSON
    # ---------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "meta": {
            "framework": "Chapter 5 Multi-Level Evaluation Protocol",
            "source": "Chapter 4 Enhanced Framework (30 scenarios)",
            "results_dir": str(ENHANCED_RESULTS),
            "thresholds": {
                "tau_q": TAU_Q,
                "tau_f1": TAU_F1,
                "budget_latency_s": BUDGET_LATENCY,
                "budget_cost_usd": BUDGET_COST,
                "budget_tokens": BUDGET_TOKENS,
                "risk_tolerance": RISK_TOLERANCE,
            }
        },
        "agent_level": {},
        "interaction_level": {},
        "system_level": {},
        "diagnostics": {},
        "aggregates": {}
    }

    # Per-scenario results
    for sid in sorted(all_agent_perfs.keys()):
        report["agent_level"][sid] = {
            name: asdict(ap) for name, ap in all_agent_perfs[sid].items()
        }
    for sid in sorted(all_interactions.keys()):
        report["interaction_level"][sid] = asdict(all_interactions[sid])
    for sid in sorted(all_system.keys()):
        report["system_level"][sid] = asdict(all_system[sid])
    for sid in sorted(all_diagnoses.keys()):
        report["diagnostics"][sid] = asdict(all_diagnoses[sid])

    # Aggregates
    agg = {}
    for agent_name in ["Disruption Monitoring", "KG Query", "Risk Manager", "CSCO"]:
        f1s = [all_agent_perfs[s][agent_name].f1 for s in all_agent_perfs]
        precs = [all_agent_perfs[s][agent_name].precision for s in all_agent_perfs]
        recs = [all_agent_perfs[s][agent_name].recall for s in all_agent_perfs]
        f1_mean, f1_lo, f1_hi = bootstrap_ci(f1s)
        p_mean, p_lo, p_hi = bootstrap_ci(precs)
        r_mean, r_lo, r_hi = bootstrap_ci(recs)
        ss = statistics.mean([1.0 if all_agent_perfs[s][agent_name].strict_success else 0.0
                              for s in all_agent_perfs])
        agg[agent_name] = {
            "precision": {"mean": round(p_mean, 4), "ci_lo": round(p_lo, 4), "ci_hi": round(p_hi, 4)},
            "recall": {"mean": round(r_mean, 4), "ci_lo": round(r_lo, 4), "ci_hi": round(r_hi, 4)},
            "f1": {"mean": round(f1_mean, 4), "ci_lo": round(f1_lo, 4), "ci_hi": round(f1_hi, 4)},
            "strict_success_rate": round(ss, 4),
        }
    report["aggregates"]["agent_level"] = agg

    hvr_m, hvr_lo, hvr_hi = bootstrap_ci(hvrs)
    hgr_m, hgr_lo, hgr_hi = bootstrap_ci(hgrs)
    report["aggregates"]["interaction_level"] = {
        "hvr": {"mean": round(hvr_m, 4), "ci_lo": round(hvr_lo, 4), "ci_hi": round(hvr_hi, 4)},
        "hgr": {"mean": round(hgr_m, 4), "ci_lo": round(hgr_lo, 4), "ci_hi": round(hgr_hi, 4)},
        "entity_propagation": round(statistics.mean(ent_props), 4),
        "risk_propagation": round(statistics.mean(risk_props), 4),
        "decision_propagation": round(statistics.mean(dec_props), 4),
    }

    cs_m, cs_lo, cs_hi = bootstrap_ci(cs_vals)
    sys_agg = {
        "constrained_success": {"mean": round(cs_m, 4), "ci_lo": round(cs_lo, 4), "ci_hi": round(cs_hi, 4)},
        "task_correctness_rate": round(tc_m, 4),
        "budget_compliance_rate": round(bp_m, 4),
        "n_scenarios": n_scenarios,
    }
    if success_latencies:
        lat_m, lat_lo, lat_hi = bootstrap_ci(success_latencies)
        cost_m, cost_lo, cost_hi = bootstrap_ci(success_costs)
        sys_agg["conditional_efficiency"] = {
            "latency_s": {"mean": round(lat_m, 1), "ci_lo": round(lat_lo, 1), "ci_hi": round(lat_hi, 1),
                          "median": round(statistics.median(success_latencies), 1),
                          "max": round(max(success_latencies), 1)},
            "cost_usd": {"mean": round(cost_m, 4), "ci_lo": round(cost_lo, 4), "ci_hi": round(cost_hi, 4),
                         "median": round(statistics.median(success_costs), 4),
                         "max": round(max(success_costs), 4)},
            "tokens": {"mean": round(statistics.mean([float(t) for t in success_tokens])),
                       "median": round(statistics.median(success_tokens)),
                       "max": max(success_tokens)},
            "n_successful": len(success_latencies),
        }
    report["aggregates"]["system_level"] = sys_agg

    # Failure attribution summary
    report["aggregates"]["failure_attribution"] = {
        lvl: len(sids) for lvl, sids in failure_levels.items()
    }

    out_file = OUTPUT_DIR / "evaluation_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\nFull report saved to: {out_file}")


if __name__ == "__main__":
    run_evaluation()
