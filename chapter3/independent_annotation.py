"""Second annotation of the Chapter 3 documents by gpt-5.5-2026-04-23, for the consistency check of the ground truth.

    python chapter3/independent_annotation.py

The model receives chapter3/prompts/annotation_prompt.txt: the ontology block and output format of the extraction
prompt, verbatim, with annotation instructions and the document; it receives none of the prompt's examples or extraction
guidance and nothing of the ground truth. Reasoning effort high; no temperature (the model accepts only its default).
One annotation per document, saved in chapter3/results/independent_annotation/ in the format of the ground truth, with
the served model, tokens, cost and finish reason in calls.jsonl. A saved annotation is not requested again.
"""
import json
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerun_extraction import REPO, cost, documents  # noqa: E402

MODEL = "gpt-5.5-2026-04-23"
OUT = REPO / "chapter3" / "results" / "independent_annotation"
PROMPT = (REPO / "chapter3" / "prompts" / "annotation_prompt.txt").read_text()


def main():
    client = OpenAI(api_key=dotenv_values(REPO / ".env")["OPENAI_API_KEY"], max_retries=5, timeout=1800)
    OUT.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for doc, text in documents().items():
        path = OUT / f"GT_Nodes_Rels_{doc}.txt"
        if path.exists():
            continue
        r = client.chat.completions.create(model=MODEL, reasoning_effort="high",
                                           messages=[{"role": "user", "content": PROMPT.replace("{text}", text)}])
        usd = cost(MODEL, r.usage)
        spent += usd
        path.write_text((r.choices[0].message.content or "").strip(), encoding="utf-8")
        with (OUT / "calls.jsonl").open("a") as f:
            f.write(json.dumps({"document": doc, "model": MODEL, "served_model": r.model, "reasoning_effort": "high",
                                "finish_reason": r.choices[0].finish_reason, "usage": r.usage.model_dump(),
                                "usd": round(usd, 6), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        print(f"{doc}: {r.choices[0].finish_reason}, ${usd:.3f}")
    print(f"${spent:.2f} spent")


if __name__ == "__main__":
    main()
