"""Shared run machinery of the pipeline evaluations: call log, snapshot check, pipeline runner and per-agent checks.

Used by scripts/run_evaluation.py (Chapter 5), evaluation/benchmark_v2/run_pipeline.py (Chapter 4 benchmark) and
scripts/run_case_study_v1_2.py (Chapter 4 case study). Every OpenAI call made by the pipeline is logged with the
caller, requested model, served snapshot, temperature and tokens, and a run stops if a served snapshot differs from
ENVIRONMENT.md. compare() checks each agent on its own answer: the Disruption Monitor against the reference; the KG
Query Agent's tool arguments against the Disruption Monitor's output and its paths against the offline paths for
those arguments; the Risk Manager's answer against the risk tool's file and the offline risk function; the CSCO's
actions against the decision rule applied to the scores it received.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from analysis.metrics import entity_keys, normalize, set_prf  # noqa: E402


SNAPSHOTS = {  # ENVIRONMENT.md, alias resolution on 11 October 2026
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-5-mini": "gpt-5-mini-2025-08-07",
}
# OpenAI list prices for gpt-4o-2024-08-06, USD per million tokens (checked 12 October 2026).
PRICE_IN = 2.50
PRICE_OUT = 10.00
TOOL_SYSTEM_PROMPT = "You are a supply chain risk expert in the automotive industry."
TEMP_KG = ROOT / "output" / "temp" / "kg_payloads"

CALLS: list = []


def caller(messages, instructions=None):
    texts = [instructions] if instructions else []
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") in ("system", "developer"):
            content = m.get("content")
            texts.append(content if isinstance(content, str) else json.dumps(content))
    system = " ".join(t for t in texts if t)
    if system.startswith(TOOL_SYSTEM_PROMPT):
        return "DisruptionAnalysisTool"
    match = re.search(r"You are (.+?)\.", system)
    return match.group(1).strip() if match else "unidentified"


def record(api, kwargs, response):
    served = getattr(response, "model", None)
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None) if usage else None
    completion = getattr(usage, "completion_tokens", None) if usage else None
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    cached = getattr(details, "cached_tokens", 0) if details else 0
    choices = getattr(response, "choices", None)
    if choices:
        finish = getattr(choices[0], "finish_reason", None)
    else:
        incomplete = getattr(response, "incomplete_details", None)
        finish = getattr(incomplete, "reason", None) or getattr(response, "status", None)
    entry = {
        "api": api,
        "caller": caller(kwargs.get("messages") or kwargs.get("input"), kwargs.get("instructions")),
        "requested_model": kwargs.get("model"),
        "temperature": kwargs.get("temperature", "not sent (API default)"),
        "served_model": served,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached or 0,
        "finish_reason": finish,
        "truncated": finish in ("length", "max_output_tokens"),
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    CALLS.append(entry)
    expected = SNAPSHOTS.get(kwargs.get("model"))
    if served and expected and served != expected:
        raise SystemExit(
            f"Served snapshot {served} differs from ENVIRONMENT.md ({expected}) for {kwargs.get('model')}"
        )


def wrap(cls, api, is_async):
    original = cls.create
    if is_async:
        async def create(self, *args, **kwargs):
            response = await original(self, *args, **kwargs)
            record(api, kwargs, response)
            return response
    else:
        def create(self, *args, **kwargs):
            response = original(self, *args, **kwargs)
            record(api, kwargs, response)
            return response
    cls.create = create


def install_wrappers():
    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.responses import AsyncResponses, Responses
    wrap(Completions, "chat.completions", False)
    wrap(AsyncCompletions, "chat.completions", True)
    wrap(Responses, "responses", False)
    wrap(AsyncResponses, "responses", True)


def extract_dict(output):
    if isinstance(output, dict):
        return output
    if hasattr(output, "raw"):
        raw = output.raw
        if isinstance(raw, dict):
            return raw
        output = raw
    if not isinstance(output, str):
        output = str(output)
    for candidate in (output, re.sub(r"```(?:json)?", "", output, flags=re.IGNORECASE).replace("```", "").strip()):
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            lit = ast.literal_eval(candidate)
            if isinstance(lit, dict):
                return lit
        except Exception:
            pass
    match = re.search(r"\{.*\}", str(output), re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


def chain_names(path):
    return tuple(n.get("company") for n in path if isinstance(n, dict) and n.get("company"))


def path_sets(kg):
    return {
        f"tier_{t}": {chain_names(p) for p in (kg.get(f"tier_{t}") or [])}
        for t in range(1, 5)
    }


def revisits(kg):
    n = 0
    for t in range(1, 5):
        for path in kg.get(f"tier_{t}") or []:
            names = chain_names(path)
            if len(names) != len(set(names)):
                n += 1
    return n


def cost_of(calls):
    pin = sum(c.get("prompt_tokens") or 0 for c in calls)
    pout = sum(c.get("completion_tokens") or 0 for c in calls)
    return {
        "prompt_tokens": pin,
        "completion_tokens": pout,
        "total_tokens": pin + pout,
        "usd": round(pin * PRICE_IN / 1_000_000 + pout * PRICE_OUT / 1_000_000, 4),
        "price_in_per_m": PRICE_IN,
        "price_out_per_m": PRICE_OUT,
    }


def as_json(risk):
    """The tool's answer arrives as Python text (tuples, integer keys); its file is JSON."""
    risk = json.loads(json.dumps(risk or {}, default=str))
    return {k: risk.get(k) for k in ("supplier_risk_scores", "tier1_risk_profiles")}


def run_one(scenario):
    sid = scenario["scenario_id"]
    CALLS.clear()
    for stale in TEMP_KG.glob(f"{sid}_*"):
        stale.unlink()
    start = time.time()
    from crew import SupplyChainCrew
    crew = SupplyChainCrew(company_name=scenario["monitored_company"]).crew()
    crew.kickoff(inputs={
        "extracted_text": scenario["news_article"],
        "company_name": scenario["monitored_company"],
        "scenario_id": sid,
    })
    mapped = {}
    key_map = {
        "Supply Chain Disruption Radar": "disruption_analysis",
        "Supply Chain Pathfinder": "kg_results",
        "Quantitative Risk Modeler": "risk_assessment",
        "Chief Supply Chain Officer": "chief_supply_chain_output",
    }
    answers = {}
    for task in crew.tasks:
        if not (task.agent and getattr(task, "output", None) is not None):
            continue
        name = task.agent.role.split(" for ")[0].strip()
        key = key_map.get(name)
        if key:
            raw = getattr(task.output, "raw", task.output)
            parsed = extract_dict(task.output)
            mapped[key] = parsed
            answers[key] = {"chars": len(str(raw)), "parsed": parsed is not None,
                            "tail": str(raw)[-120:]}

    kg = mapped.get("kg_results") or {}
    if isinstance(kg, dict) and kg.get("kg_payload_path") and Path(kg["kg_payload_path"]).exists():
        kg = json.loads(Path(kg["kg_payload_path"]).read_text())
    risk_relay = mapped.get("risk_assessment") or {}
    if isinstance(risk_relay.get("risk_assessment"), dict):
        risk_relay = risk_relay["risk_assessment"]
    risk_path = TEMP_KG / f"{sid}_risk.json"
    risk_file = json.loads(risk_path.read_text()) if risk_path.exists() else {}

    return {
        "scenario_id": sid,
        "runtime_seconds": round(time.time() - start, 1),
        "calls": list(CALLS),
        "truncated_calls": [c for c in CALLS if c.get("truncated")],
        "cost": cost_of(CALLS),
        "agent_answers": answers,
        "agent_outputs": {
            "disruption_analysis": mapped.get("disruption_analysis"),
            "kg_results": kg,
            "risk_assessment": risk_relay,
            "risk_tool_file": risk_file,
            "chief_supply_chain_output": mapped.get("chief_supply_chain_output"),
        },
    }


TIERS = [f"tier_{t}" for t in range(1, 5)]


LEAK_KEYS = ("expected_action", "expected_decisions")


TOL = 1e-4


def counts(kg):
    return [len((kg or {}).get(t) or []) for t in TIERS]


def compare_paths(kg, ref):
    a, b = path_sets(kg), path_sets(ref)
    return {"pipeline": counts(kg), "reference": counts(ref), "identical": all(a[t] == b[t] for t in TIERS)}


def compare_scores(risk, ref_risk):
    pred = {normalize(k): v for k, v in (risk.get("supplier_risk_scores") or {}).items()}
    gold = {normalize(k): v for k, v in (ref_risk.get("supplier_risk_scores") or {}).items()}
    plev = {normalize(p["supplier"]): p.get("risk_level") for p in risk.get("tier1_risk_profiles") or []}
    glev = {normalize(p["supplier"]): p.get("risk_level") for p in ref_risk.get("tier1_risk_profiles") or []}
    diffs = [{"supplier": k, "reference": gold.get(k), "pipeline": pred.get(k)} for k in sorted(set(pred) | set(gold))
             if k not in pred or k not in gold or abs(pred[k] - gold[k]) > TOL]
    level_diffs = [k for k in sorted(set(plev) & set(glev)) if plev[k] != glev[k]]
    return {"n_pipeline": len(pred), "n_reference": len(gold), "score_mismatches": diffs,
            "level_mismatches": level_diffs, "identical": not diffs and not level_diffs}


def compare_csco(csco, gt):
    gold = {normalize(k): v for k, v in gt["expected_decisions"].items()}
    scores = {normalize(k): v for k, v in (gt["risk_assessment"].get("supplier_risk_scores") or {}).items()}
    decisions = (csco or {}).get("decisions") or {}
    listed = {normalize(k): (k, v or {}) for k, v in decisions.items()}
    unknown = sorted(k for k, _ in listed.values() if normalize(k) not in gold)
    wrong, quoted = [], []
    for key, action in gold.items():
        name, dec = listed.get(key, (None, {}))
        got = dec.get("action") if name else "MAINTAIN_STANDARD"
        if got != action:
            wrong.append({"supplier": name or key, "expected": action, "csco": got, "score": scores.get(key),
                          "listed": bool(name)})
        raw = dec.get("risk_score_raw") if name else None
        if name and (not isinstance(raw, (int, float)) or abs(raw - scores[key]) > TOL):
            quoted.append({"supplier": name, "assessment": scores[key], "quoted": raw})
    actionable = sum(a != "MAINTAIN_STANDARD" for a in gold.values())
    return {"listed": len(decisions), "actionable": actionable, "action_errors": wrong,
            "score_quote_errors": quoted, "suppliers_not_in_assessment": unknown,
            "correct": not wrong and not quoted and not unknown}


def compare(sid, run, gt, cond, meta, resolver):
    ao = run["agent_outputs"]
    dm = ao.get("disruption_analysis") or {}
    inv, gdm = dm.get("involved") or {}, gt["disruption_analysis"]
    entities = {}
    for kind in ("countries", "industries", "companies"):
        p = entity_keys(kind, inv.get(kind) or [], resolver)
        g = entity_keys(kind, gdm["involved"][kind], resolver)
        entities[kind] = {"pipeline": inv.get(kind) or [], "ground_truth": gdm["involved"][kind],
                          "f1": set_prf(p, g)[2], "equal": p == g}
    ongoing_ok = dm.get("ongoing_disruption") is gdm["ongoing_disruption"]
    dm_out = {
        "ongoing": dm.get("ongoing_disruption"), "ongoing_ok": ongoing_ok,
        "type": dm.get("type"), "type_ok": normalize(dm.get("type")) == normalize(gdm["type"]),
        "scope": dm.get("scope"),
        "scope_ok": (dm.get("scope") == gdm["scope"]) if gdm["ongoing_disruption"] else True,
        "entities": entities, "reasoning": dm.get("reasoning"),
    }
    dm_out["pass"] = all([ongoing_ok, dm_out["type_ok"], dm_out["scope_ok"], *(e["equal"] for e in entities.values())])

    req = (meta or {}).get("requested") or {}
    relay = {
        "ongoing": (meta or {}).get("ongoing_disruption") is dm.get("ongoing_disruption"),
        "scope": req.get("scope") == dm.get("scope"),
        **{k: sorted(req.get(k) or []) == sorted(inv.get(k) or []) for k in ("countries", "industries", "companies")},
    }
    kg = ao.get("kg_results") or {}
    kg_out = {
        "relay": relay, "unresolved": {k: v for k, v in ((meta or {}).get("unresolved") or {}).items() if v},
        "vs_offline_on_pipeline_input": compare_paths(kg, cond["kg_results"]),
        "vs_ground_truth": compare_paths(kg, gt["kg_results"]),
        "revisits": revisits(kg),
    }
    kg_out["pass"] = all(relay.values()) and kg_out["vs_offline_on_pipeline_input"]["identical"] and not kg_out["revisits"]

    risk, risk_file = ao.get("risk_assessment") or {}, ao.get("risk_tool_file") or {}
    leak = [k for k in LEAK_KEYS if k in json.dumps(risk, default=str) or k in json.dumps(risk_file, default=str)]
    risk_out = {
        "answer_equals_tool_file": as_json(risk) == as_json(risk_file),
        "vs_offline_on_pipeline_paths": compare_scores(risk, cond["risk_assessment"]),
        "vs_ground_truth": compare_scores(risk, gt["risk_assessment"]),
        "expected_action_in_output": leak,
    }
    risk_out["pass"] = (risk_out["answer_equals_tool_file"] and risk_out["vs_offline_on_pipeline_paths"]["identical"]
                        and not leak)

    csco_out = {"vs_ground_truth": compare_csco(ao.get("chief_supply_chain_output"), gt),
                "vs_assessment_received": compare_csco(ao.get("chief_supply_chain_output"), cond)}
    csco_out["pass"] = csco_out["vs_assessment_received"]["correct"]

    system = {
        "end_to_end": all(x["pass"] for x in (dm_out, kg_out, risk_out, csco_out))
                      and kg_out["vs_ground_truth"]["identical"] and risk_out["vs_ground_truth"]["identical"]
                      and csco_out["vs_ground_truth"]["correct"],
        "truncated_calls": len(run.get("truncated_calls") or []),
        "served_models": sorted({c.get("served_model") or "none" for c in run.get("calls") or []}),
        "temperatures": sorted({str(c.get("temperature")) for c in run.get("calls") or []}),
    }
    return {"disruption_monitor": dm_out, "kg_query": kg_out, "risk_manager": risk_out, "csco": csco_out,
            "system": system}


def slim(run):
    out = json.loads(json.dumps(run, default=str))
    ao = out.get("agent_outputs") or {}
    if isinstance(ao.get("kg_results"), dict):
        ao["kg_results"] = {k: (len(v) if k in TIERS else v) for k, v in ao["kg_results"].items()}
    ao.pop("risk_tool_file", None)
    return out
