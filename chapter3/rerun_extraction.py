"""Chapter 3 extraction reruns: prompt variants and other models on the evaluation documents, three runs each.

    python chapter3/rerun_extraction.py --max-usd 40 [--arms as_used gpt-5.5 ...] [--workers 4]

Arms
  as_used         the prompt of the reported runs on gpt-4o-2024-08-06: one user message, temperature 0.7
  demonstrations  the same prompt with two annotated documents of the first-stage annotation added as examples
                  (prompts/extraction_prompt_demonstrations.txt); the two documents are drawn with seed 20261014 and
                  are excluded from every comparison with this arm
  paraphrased     the same prompt with its instructions reworded; ontology, output format and examples unchanged
  gpt-5.5         the prompt as used on gpt-5.5-2026-04-23 (no temperature: the model accepts only its default)
  gpt-4           the prompt as used on gpt-4-0613, temperature 0.7
  o3-mini         the prompt as used on o3-mini-2025-01-31 (no temperature: the model accepts only its default)

Inputs (read-only): the documents in chapter3/documents and the prompts in chapter3/prompts. Outputs: chapter3/results/reruns/<arm>/run_<k>/
Nodes_Rels_<document>.txt, in the format of the stored runs, and calls.jsonl with the served model, tokens, cost and
finish reason of every call. A saved output is not requested again. API errors are retried with backoff.
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[1]

DOCUMENTS = REPO / "chapter3" / "documents"
PROMPTS = REPO / "chapter3" / "prompts"
OUT = REPO / "chapter3" / "results" / "reruns"
RUNS, SEED = 3, 20261014
ARMS = {
    "as_used": ("gpt-4o-2024-08-06", "as_used", 0.7),
    "demonstrations": ("gpt-4o-2024-08-06", "demonstrations", 0.7),
    "paraphrased": ("gpt-4o-2024-08-06", "paraphrased", 0.7),
    "gpt-5.5": ("gpt-5.5-2026-04-23", "as_used", None),
    "gpt-4": ("gpt-4-0613", "as_used", 0.7),
    "o3-mini": ("o3-mini-2025-01-31", "as_used", None),
}
PRICES = {"gpt-4o-2024-08-06": (2.50, 1.25, 10.00), "gpt-5.5-2026-04-23": (5.00, 0.50, 30.00),
          "gpt-4-0613": (30.00, 30.00, 60.00), "o3-mini-2025-01-31": (1.10, 0.55, 4.40)}  # USD per 1M: in, cached, out
LOCK = threading.Lock()
SPENT = [0.0]


def documents():
    return {f.name[:-len("_InputTexts.txt")]: f.read_text(encoding="utf-8")
            for f in sorted(DOCUMENTS.glob("*_InputTexts.txt"))}


def demonstration_docs(docs):
    return sorted(random.Random(SEED).sample(sorted(docs), 2))


def prompts(docs):
    return {name: (PROMPTS / f"extraction_prompt_{name}.txt").read_text()
            for name in ("as_used", "demonstrations", "paraphrased")}


def cost(model, usage):
    cin, ccached, cout = PRICES[model]
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return ((usage.prompt_tokens - cached) * cin + cached * ccached + usage.completion_tokens * cout) / 1e6


def call(client, arm, run, doc, text, prompt, max_usd):
    from openai import APIConnectionError, APIStatusError, RateLimitError

    model, _, temperature = ARMS[arm]
    path = OUT / arm / f"run_{run}" / f"Nodes_Rels_{doc}.txt"
    if path.exists():
        return
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt.replace("{text}", text)}]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    errors, delay = [], 30
    while True:
        with LOCK:
            if SPENT[0] >= max_usd:
                raise SystemExit(f"stopped at ${SPENT[0]:.2f}: the budget of ${max_usd} is reached")
        try:
            r = client.chat.completions.create(**kwargs)
            break
        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and e.status_code < 500 and not isinstance(e, RateLimitError):
                raise
            errors.append(repr(e)[:300])
            if len(errors) > 8:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 600)
    usd = cost(model, r.usage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((r.choices[0].message.content or "").strip(), encoding="utf-8")
    with LOCK:
        SPENT[0] += usd
        with (OUT / "calls.jsonl").open("a") as f:
            f.write(json.dumps({"arm": arm, "run": run, "document": doc, "model": model, "served_model": r.model,
                                "temperature": temperature, "finish_reason": r.choices[0].finish_reason,
                                "usage": r.usage.model_dump(), "usd": round(usd, 6), "retried_errors": errors,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    docs = documents()
    reference = json.loads((REPO / "chapter3" / "reference_final" / "summary.json").read_text())["documents"]
    assert sorted(docs) == sorted(reference), "the documents differ from those of the reference annotation"
    texts = prompts(docs)
    from openai import OpenAI

    client = OpenAI(api_key=dotenv_values(REPO / ".env")["OPENAI_API_KEY"], max_retries=0, timeout=900)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "demonstration_documents.json").write_text(json.dumps(demonstration_docs(docs)))
    work = [(a, k, d) for a in args.arms for k in range(1, RUNS + 1) for d in docs]
    with ThreadPoolExecutor(args.workers) as ex:
        for f in [ex.submit(call, client, a, k, d, docs[d], texts[ARMS[a][1]], args.max_usd) for a, k, d in work]:
            f.result()
    saved = sum(1 for a, k, d in work if (OUT / a / f"run_{k}" / f"Nodes_Rels_{d}.txt").exists())
    print(f"{saved} of {len(work)} outputs saved; ${SPENT[0]:.2f} spent in this session")


if __name__ == "__main__":
    main()
