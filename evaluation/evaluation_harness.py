"""Per-run measures of the Chapter 5 evaluation that scripts/score_v1_2.py takes from the agent outputs: the knowledge-
graph paths against the reference (eval_kg_query) and the validity of each hand-off between agents (eval_interactions).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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



@dataclass
@dataclass
class AgentPerf:
    name: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    valid_schema: bool = True
    detail: str = ""


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

    tn_flag = " [TRUE_NEGATIVE]" if is_true_negative(gt) else ""
    ap.detail += f"TP={tp_total} FP={fp_total} FN={fn_total}{tn_flag}"
    return ap


@dataclass
class InteractionMetrics:
    hvr: float = 0.0  # share of valid hand-offs
    hvr_details: Dict[str, bool] = field(default_factory=dict)


def eval_interactions(ao: dict, gt: dict) -> InteractionMetrics:
    """Validity of the four hand-offs.

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
    return im
