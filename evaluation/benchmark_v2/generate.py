"""Articles of the relevance benchmark: generation, fidelity check and near-duplicate screen (Appendix C.6).

    python -m evaluation.benchmark_v2.generate --max-usd 40 [--workers 6] [--redo TYPE_OR_ID ... --audit-round K]

Run after evaluation.benchmark_v2.ground_truth. For each item the generator (gpt-4.1-2025-04-14, temperature 0.7)
writes an article from the facts of its specification; it is not told the item type or the label. An article that
names a monitored firm, calls anything in it fictional or hypothetical, or, for an event confined to an industry, says
whether other industries were affected, fails without further reading. The checker (gpt-5-mini-2025-08-07) reads every
other article alone and extracts ongoing_disruption, type, scope, countries, industries and companies; the
ground-truth function is run on those fields. The item is accepted when that label equals the specification's label
and, for a disruption under way, the extracted type equals the specified type and the extracted countries include
every specified country. An N4 item must also pass the confinement check (prompts/confinement.md). A failing item is
regenerated at most twice; one that still fails goes to the audit.

The near-duplicate screen then runs over the accepted articles in item order, with word TF-IDF. An article above 0.90
cosine to an earlier kept article is regenerated at most twice; each new article must pass the checks and the screen
again, or the item is dropped.

--redo generates again, after audit round K, each named item or every item of a named type (a stratum with two or
more wrong labels in its audit sample); its earlier record is kept under items/test/superseded/. A regenerated article
is screened after all others, against every kept article, with the inverse document frequencies already fixed.

Each item is saved to items/test/articles/<item>.json; an item already saved is not repeated. Every call is logged to
items/test/calls.jsonl with the served model, tokens and cost. Needs an OpenAI API key in .env.
"""
import argparse
import hashlib
import json
import math
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import dotenv_values
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from evaluation.benchmark_v2.ground_truth import compute
from evaluation.benchmark_v2.specify import OUT, OUTSIDE_INDUSTRIES
from evaluation.ground_truth_generator import GroundTruthGenerator
from offline import kg, names

ROOT = OUT.parents[2]
HERE = Path(__file__).resolve().parent
GENERATOR = "gpt-4.1-2025-04-14"
CHECKER = "gpt-5-mini-2025-08-07"
PRICES = {GENERATOR: (2.00, 0.50, 8.00), CHECKER: (0.25, 0.025, 2.00)}  # USD per 1M tokens: in, cached, out
GEN_PROMPT = (HERE / "prompts" / "generator.md").read_text()
CHECK_PROMPT = (HERE / "prompts" / "checker.md").read_text()
CONFINE_PROMPT = (HERE / "prompts" / "confinement.md").read_text()
FIRM_NAMES = re.compile(r"\b(BMW|Bayerische|Mercedes|Daimler|Tesla)\b", re.I)
FICTION = re.compile(r"\b(fictional|fictitious|hypothetical|imaginary)\b", re.I)
OTHER_INDUSTRIES = re.compile(r"\b(other|another)\s+(industr\w*|sectors?)\b", re.I)
EXTRA_GENERATIONS = 2
SIMILARITY = 0.90
EVENT = {  # (type, scope): what the generator is asked to report
    ("Geopolitical", "country"): "a geopolitical crisis, such as armed conflict, political unrest or broad sanctions",
    ("Geopolitical", "industries"): "a geopolitical measure or conflict aimed at one industry, such as sanctions on "
                                    "its firms or the seizure or blockade of its sites",
    ("Trade Policy", "country"): "a trade-policy measure across many goods, such as broad tariffs, import "
                                 "restrictions or an embargo",
    ("Trade Policy", "industries"): "a trade-policy measure on one industry's products, such as tariffs, import or "
                                    "export restrictions or an embargo",
    ("Natural Disaster", "country"): "a natural disaster, such as an earthquake, flood, hurricane or wildfire",
    ("Natural Disaster", "industries"): "a natural event that strikes one industry's own sites, such as a drought, "
                                        "flood, landslide or storm damaging its plants, mines, farms or other premises",
    ("Labour Strike", "country"): "a general strike across sectors",
    ("Labour Strike", "industries"): "a strike or walkout by one industry's workers",
    ("Economic Crisis", "country"): "an economic crisis, such as a recession, currency collapse, hyperinflation or "
                                    "financial crisis",
    ("Economic Crisis", "industries"): "a financial crisis within one industry, such as a debt crisis, a credit "
                                       "freeze or a collapse in prices or demand that hits its firms",
    ("Cyber Attack", "country"): "a cyber attack on national infrastructure, such as power, telecommunications, "
                                 "payments or ports",
    ("Cyber Attack", "industries"): "a cyber attack on one industry's firms, such as ransomware or an attack on "
                                    "their IT systems",
    ("Pandemic", "country"): "an epidemic with national health measures, such as lockdowns or closures",
    ("Pandemic", "industries"): "a disease outbreak among one industry's workforce or at its sites",
}
EU = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia", "Finland",
      "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta",
      "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"}
EU_EVENT = {  # replaces EVENT in an EU member state, where trade rules and sanctions are set by the EU
    ("Geopolitical", "country"): "a geopolitical crisis, such as political unrest, a government crisis or broad "
                                 "sanctions imposed by a country outside the European Union",
    ("Geopolitical", "industries"): "a geopolitical conflict aimed at one industry, such as sabotage of its sites "
                                    "attributed to a foreign state, or the occupation or blockade of its sites by "
                                    "protesters in a political dispute",
    ("Trade Policy", "country"): "a trade-policy measure across many goods imposed by a country outside the European "
                                 "Union on goods from {country}, such as broad tariffs, import restrictions or an embargo",
    ("Trade Policy", "industries"): "a trade-policy measure imposed by a country outside the European Union on one "
                                    "industry's products from {country}, such as tariffs, import restrictions or an "
                                    "embargo",
}
CONFINED = ("Describe no effect on any business outside the industry, including its suppliers, contractors, service "
            "and technology providers, logistics firms and other manufacturers; on public services such as hospitals "
            "or public transport; or on shared infrastructure such as public roads, railways, ports, airports, power, "
            "water or telecommunications, unless the industry itself operates it.")
INDUSTRIES = sorted(set(kg.load().industry.values()) | set(OUTSIDE_INDUSTRIES))


def confined(ind):
    others = ", ".join(i for i in INDUSTRIES if i != ind)
    return (f"{CONFINED} The {ind} industry is distinct from each of these industries, whose firms are outside it: "
            f"{others}. Describe no disruption of any of their firms.")
LOCK = threading.Lock()
SPENT = [0.0]


# -- prompts --------------------------------------------------------------------------------------------------------
def facts(spec):
    country = spec["display"]["countries"][0]
    company = spec["companies"][0] if spec["companies"] else None
    ind = spec["industries"][0] if spec["industries"] else None
    if spec["status"] == "none":
        who = f"{company}, a company in the {ind} industry in {country}" if company else \
            f"the {ind} industry in {country}"
        return [f"The article reports {spec['topic']} concerning {who}.",
                "Nothing in the article is a disruption: no strike, shortage, outage, closure, sanction, tariff, "
                "disaster, insolvency, disease outbreak or cyber incident, and no warning or forecast of one."]
    if spec["type"] == "Company Bankruptcy":
        subject = f"a company in the {ind} industry in {country}, which you name" if spec.get("invented_company") \
            else f"{company}, a company in the {ind} industry in {country}"
        out = [f"The insolvency of {subject} (bankruptcy, administration or liquidation). The article reports its "
               f"effects on the company's own operations. Mention no other industry. {confined(ind)}"]
    else:
        events = EU_EVENT if country in EU and (spec["type"], spec["scope"]) in EU_EVENT else EVENT
        event = events[spec["type"], spec["scope"]].replace("one industry", f"the {ind} industry", 1)
        event = event.format(country=country) if "{country}" in event else f"{event}, in {country}"
        if spec["scope"] == "industries":
            out = [f"{event[0].upper()}{event[1:]}. Choose a specific, plausible event of this kind. Report its "
                   f"effects on that industry's own firms, sites and workers. Mention no other industry, and do not "
                   f"say whether other industries or sectors were affected. {confined(ind)}"]
        else:
            out = [f"{event[0].upper()}{event[1:]}. Choose a specific, plausible event of this kind that affects "
                   f"economic activity across {country}, not one industry alone."]
    if company and spec["type"] != "Company Bankruptcy":
        out.append(f"{company} is among the companies affected.")
    if spec["status"] == "ongoing":
        out.append("The event is still under way at the time of writing; the article reports its current effects.")
    elif spec["status"] == "over":
        out.append("The event ended in recent days; the article reports how it was resolved and that the affected "
                   "operations have resumed. Nothing remains disrupted: report no remaining shortage, delay, backlog, "
                   "closure or restriction.")
    else:
        year = historical_year(spec)
        out.append(f"The article looks back on the event, which happened in {year}, and reports nothing disrupted "
                   f"now. The event may be real or invented but must be plausible for {country} in {year}.")
    return out


def historical_year(spec):
    """2021 when a real company is named (five years before generation, and the year most likely to postdate its
    founding); otherwise 2016 to 2021, fixed by the item id."""
    if spec["companies"]:
        return 2021
    return 2016 + int(hashlib.sha256(spec["item_id"].encode()).hexdigest(), 16) % 6


def generator_prompt(spec):
    if spec.get("invented_company"):
        rule = "The insolvent company is fictional: give it a realistic name. Name no other real company."
    elif spec["companies"]:
        rule = (f"Write the name {spec['companies'][0]} exactly as given at least once. Name no other real company; "
                f"any other company you need must be fictional.")
    else:
        rule = "Name no real company; any company you need must be fictional."
    return GEN_PROMPT.format(length_words=spec["length_words"], style=spec["style"],
                             facts="\n".join(f"- {f}" for f in facts(spec)), company_rule=rule)


# -- API ------------------------------------------------------------------------------------------------------------
def cost(model, usage):
    cin, ccached, cout = PRICES[model]
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return ((usage.prompt_tokens - cached) * cin + cached * ccached + usage.completion_tokens * cout) / 1e6


def call(client, log, max_usd, item_id, role, model, prompt, **kw):
    delay = 30
    for attempt in range(1, 10):
        with LOCK:
            if SPENT[0] >= max_usd:
                raise SystemExit(f"stopped at ${SPENT[0]:.2f}: the budget of ${max_usd} is reached")
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], **kw)
        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and e.status_code < 500 and not isinstance(e, RateLimitError):
                raise
            with LOCK, log.open("a") as f:
                f.write(json.dumps({"item_id": item_id, "role": role, "model": model, "api_error": repr(e)[:300],
                                    "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            time.sleep(delay)
            delay = min(delay * 2, 600)
            continue
        usd = cost(model, r.usage)
        rec = {"item_id": item_id, "role": role, "model": model, "served_model": r.model,
               "finish_reason": r.choices[0].finish_reason, "usage": r.usage.model_dump(), "usd": round(usd, 6),
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with LOCK:
            SPENT[0] += usd
            with log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return r.choices[0].message.content or "", rec
    raise RuntimeError(f"{item_id} {role}: nine API errors in a row")


# -- fidelity -------------------------------------------------------------------------------------------------------
def extracted_spec(parsed):
    lists = {k: [str(x) for x in parsed.get(k) or [] if str(x).strip()] for k in ("countries", "industries",
                                                                                   "companies")}
    return {"ongoing_disruption": parsed.get("ongoing_disruption") is True, "type": str(parsed.get("type")),
            "scope": parsed.get("scope") if parsed.get("scope") in ("country", "industries") else "country", **lists}


def json_call(ctx, spec, role, prompt, key):
    client, log, max_usd, *_ = ctx
    for _ in range(3):
        text, rec = call(client, log, max_usd, spec["item_id"], role, CHECKER, prompt,
                         response_format={"type": "json_object"}, reasoning_effort="medium")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and key in parsed:
                return parsed, rec
        except json.JSONDecodeError:
            pass
    return None, rec


def confinement(ctx, spec, article):
    """Effects outside the industry that the checker finds in an article on an industry-confined event."""
    prompt = CONFINE_PROMPT.replace("{industries}", ", ".join(INDUSTRIES)).replace("{industry}", spec["industries"][0]) \
        .replace("{country}", spec["display"]["countries"][0]).replace("{article}", article)
    parsed, rec = json_call(ctx, spec, "confinement", prompt, "effects_outside")
    effects = None if parsed is None else [e for e in parsed["effects_outside"] if isinstance(e, dict)]
    return effects, rec


def fidelity(ctx, spec, label, article):
    """The checker's reading of the article, and whether it agrees with the specification."""
    if FIRM_NAMES.search(article):
        return {"accepted": False, "reason": "names a monitored firm"}
    if FICTION.search(article):
        return {"accepted": False, "reason": "calls something fictional or hypothetical"}
    if spec["status"] != "none" and spec["scope"] == "industries" and OTHER_INDUSTRIES.search(article):
        return {"accepted": False, "reason": "says whether other industries were affected"}
    client, log, max_usd, net, resolver = ctx
    for tries in range(1, 4):
        text, rec = call(client, log, max_usd, spec["item_id"], "checker", CHECKER,
                         CHECK_PROMPT.replace("{article}", article), response_format={"type": "json_object"},
                         reasoning_effort="medium")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "ongoing_disruption" in parsed:
                break
        except json.JSONDecodeError:
            pass
    else:
        return {"accepted": False, "reason": "checker returned no valid JSON in three calls", "raw": text}
    ext = extracted_spec(parsed)
    gt, _ = compute(net, resolver, dict(spec, **ext))
    got = bool(gt["scenario_metadata"]["label"])
    reasons = []
    if got != label:
        reasons.append(f"label {got} from the extracted fields, {label} specified")
    if spec["ongoing_disruption"]:
        if not ext["ongoing_disruption"]:
            reasons.append("not read as ongoing")
        if ext["type"] != spec["type"]:
            reasons.append(f"type {ext['type']!r}, {spec['type']!r} specified")
        found = {resolver.country(c) for c in ext["countries"]}
        if missing := [c for c in spec["countries"] if c not in found]:
            reasons.append(f"countries miss {missing}")
    out = {"extracted": parsed, "extracted_label": got, "checker_call": rec}
    if spec["ongoing_disruption"] and spec["scope"] == "industries" and not reasons:
        effects, out["confinement_call"] = confinement(ctx, spec, article)
        out["effects_outside"] = effects
        if effects is None:
            reasons.append("confinement check returned no valid JSON in three calls")
        elif effects:
            reasons.append(f"{len(effects)} effects outside the industry")
    return {"accepted": not reasons, "reason": "; ".join(reasons) or None, **out}


def generate(ctx, spec):
    client, log, max_usd, *_ = ctx
    prompt = generator_prompt(spec)
    text, rec = call(client, log, max_usd, spec["item_id"], "generator", GENERATOR, prompt, temperature=0.7,
                     max_tokens=1500)
    return {"prompt": prompt, "article": text.strip(), "generator_call": rec}


def first_pass(ctx, folder, spec, label):
    path = folder / "articles" / f"{spec['item_id']}.json"
    if path.exists():
        return
    attempts = []
    for _ in range(1 + EXTRA_GENERATIONS):
        a = generate(ctx, spec)
        a["fidelity"] = fidelity(ctx, spec, label, a["article"])
        attempts.append(a)
        if a["fidelity"]["accepted"]:
            break
    ok = attempts[-1]["fidelity"]["accepted"]
    old = latest_superseded(folder, path.name)
    path.write_text(json.dumps({"item_id": spec["item_id"], "item_type": spec["item_type"], "label": label,
                                "status": "accepted" if ok else "audit",
                                "article": attempts[-1]["article"] if ok else None,
                                "fidelity_regenerations": len(attempts) - 1, "attempts": attempts,
                                **({"supersedes": str(old.relative_to(folder))} if old else {})},
                               indent=1, ensure_ascii=False))


# -- near duplicates ------------------------------------------------------------------------------------------------
def tokens(text):
    return re.findall(r"[^\W_]{2,}", text.lower())


def fit_idf(texts):
    n = len(texts)
    df = Counter(t for x in texts for t in set(tokens(x)))
    return {t: math.log((1 + n) / (1 + k)) + 1 for t, k in df.items()}, math.log(1 + n) + 1


def vector(text, idf):
    weights, unseen = idf
    v = {t: c * weights.get(t, unseen) for t, c in Counter(tokens(text)).items()}
    norm = math.sqrt(sum(w * w for w in v.values()))
    return {t: w / norm for t, w in v.items()} if norm else {}


def cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


def superseded_dir(folder, audit_round):
    """Where the records replaced after audit round K are kept: superseded/ for round 1, superseded/round<K>/ after."""
    return folder / "superseded" if audit_round == 1 else folder / "superseded" / f"round{audit_round}"


def latest_superseded(folder, name):
    """The most recently replaced record of an item, or None."""
    rounds = sorted((int(p.name[5:]) for p in (folder / "superseded").glob("round*")), reverse=True)
    for path in [folder / "superseded" / f"round{k}" / name for k in rounds] + [folder / "superseded" / name]:
        if path.exists():
            return path
    return None


def supersede(folder, item_ids, audit_round):
    """Move each item's record, once per audit round, so that the item is generated again."""
    dest = superseded_dir(folder, audit_round)
    dest.mkdir(parents=True, exist_ok=True)
    for item_id in item_ids:
        old, new = dest / f"{item_id}.json", folder / "articles" / f"{item_id}.json"
        if not old.exists():
            new.rename(old)


def first_article(folder, item_id, rec):
    """The article that passed the fidelity check before the screen, from the superseded record if there is one."""
    old = folder / "superseded" / f"{item_id}.json"
    rec = json.loads(old.read_text()) if old.exists() else rec
    if "screen" in rec:
        return next(t["article"] for t in rec["screen"]["tries"] if t.get("article"))
    return rec["article"]


def screen(ctx, folder, labels, specs):
    """Items generated again after the audit are screened last, against every kept article."""
    records = {p.stem: json.loads(p.read_text()) for p in (folder / "articles").glob("*.json")}
    fitted = folder / "screen_idf_items.json"
    if not fitted.exists():
        fitted.write_text(json.dumps(sorted(i for i, r in records.items() if r["status"] == "accepted")))
    idf_items = json.loads(fitted.read_text())
    idf = fit_idf([first_article(folder, i, records[i]) for i in idf_items])
    redone = {p.stem for p in (folder / "superseded").glob("*.json")}
    order = [i for i in idf_items if i not in redone] + \
            [i for i in sorted(redone) if records[i]["status"] == "accepted"]
    kept = []
    for item_id in order:
        rec = records[item_id]
        if "screen" in rec:
            if rec["status"] == "accepted":
                kept.append(vector(rec["article"], idf))
            continue
        tries, regenerations, article = [], 0, rec["article"]
        while True:
            if article is not None:
                top = max((cosine(vector(article, idf), u) for u in kept), default=0.0)
                tries.append({"article": article, "max_similarity": round(top, 4)})
                if top <= SIMILARITY:
                    break
            if regenerations == EXTRA_GENERATIONS:
                article = None
                break
            regenerations += 1
            a = generate(ctx, specs[item_id])
            a["fidelity"] = fidelity(ctx, specs[item_id], labels[item_id], a["article"])
            tries.append({"replacement": a})
            article = a["article"] if a["fidelity"]["accepted"] else None
        rec["screen"] = {"tries": tries, "similarity_regenerations": regenerations, "idf_articles": len(idf_items)}
        if article is None:
            rec["status"], rec["article"] = "dropped_duplicate", None
        else:
            rec["article"] = article
            kept.append(vector(article, idf))
        (folder / "articles" / f"{item_id}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--redo", nargs="*", default=[], metavar="TYPE_OR_ID",
                    help="item types or item ids to generate again after an audit round")
    ap.add_argument("--audit-round", type=int, default=1, help="the audit round whose rule --redo applies")
    args = ap.parse_args()
    folder = OUT / "test"
    (folder / "articles").mkdir(exist_ok=True)
    specs = {s["item_id"]: s for s in json.loads((folder / "specs.json").read_text())["items"]}
    if args.redo:
        supersede(folder, sorted(i for i, s in specs.items() if s["item_type"] in args.redo or i in args.redo),
                  args.audit_round)
    labels = {i: json.loads((folder / "ground_truth_final" / f"{i}.json").read_text())["scenario_metadata"]["label"]
              for i in specs}
    net = kg.load()
    GroundTruthGenerator._resolver = resolver = names.Resolver(net)
    client = OpenAI(api_key=dotenv_values(ROOT / ".env")["OPENAI_API_KEY"], max_retries=0, timeout=600)
    ctx = (client, folder / "calls.jsonl", args.max_usd, net, resolver)
    with ThreadPoolExecutor(args.workers) as ex:
        for f in [ex.submit(first_pass, ctx, folder, specs[i], labels[i]) for i in specs]:
            f.result()
    screen(ctx, folder, labels, specs)
    records = [json.loads((folder / "articles" / f"{i}.json").read_text()) for i in sorted(specs)]
    by_type = {}
    for r in records:
        t = by_type.setdefault(r["item_type"], Counter())
        t[r["status"]] += 1
        t["fidelity_regenerations"] += r["fidelity_regenerations"]
        t["similarity_regenerations"] += r.get("screen", {}).get("similarity_regenerations", 0)
        t["generated_again_after_audit"] += "supersedes" in r
    calls = [json.loads(x) for x in (folder / "calls.jsonl").read_text().splitlines()]
    summary = {"items": len(records), "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
               "served_models": dict(Counter(c.get("served_model") for c in calls if "served_model" in c)),
               "api_errors": sum("api_error" in c for c in calls),
               "usd": round(sum(c.get("usd", 0) for c in calls), 4)}
    (folder / "generation_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
