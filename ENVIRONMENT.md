# Software environment and model versions

## Python packages

| File | Purpose |
|---|---|
| `requirements.txt` | Recomputing every reported result from the stored outputs: no API key, no Neo4j |
| `requirements-lock.txt` | Running the pipeline: the exact environment of every stored pipeline run (Python 3.12.2) |

Each pipeline run record stores, under `provenance`, the Python version, the platform and the versions of crewai
(1.9.3), openai (1.83.0) and neo4j (6.1.0).

Recomputed values can differ from the stored files in the last floating-point digit when another Python or NumPy
version sums in a different order (for example 1.6918719999999998 for 1.691872). Every reported value is unaffected.

## Language models

Every API call is logged with the requested model, the snapshot that served it, the temperature sent and the tokens.
The pipeline runners stop if a served snapshot differs from the table below (`SNAPSHOTS` in
`scripts/pipeline_runner.py`).

| Use | Model (served snapshot) | Temperature |
|---|---|---|
| Chapter 3: the ten stored extraction runs | GPT-4o through LangChain | 0.7 (LangChain default) |
| Chapter 3: prompt and model comparison (`chapter3/rerun_extraction.py`) | `gpt-4o-2024-08-06`, `gpt-4-0613` | 0.7 |
| | `gpt-5.5-2026-04-23`, `o3-mini-2025-01-31` | model default |
| Chapter 3: second annotation (`chapter3/independent_annotation.py`) | `gpt-5.5-2026-04-23` | model default |
| Chapter 4: benchmark articles (`evaluation/benchmark_v2/generate.py`) | generator `gpt-4.1-2025-04-14` | 0.7 |
| | fidelity check `gpt-5-mini-2025-08-07` | model default |
| Chapters 4 and 5: the pipeline | `gpt-4o` → `gpt-4o-2024-08-06`, `gpt-4.1` → `gpt-4.1-2025-04-14`, `gpt-5-mini` → `gpt-5-mini-2025-08-07` | not sent |
| Case study: company lookup of the Product Search Agent | `text-embedding-3-small` | – |

No agent is given a model in the code: CrewAI takes it from the environment variable `OPENAI_MODEL_NAME`, which the
runners set for each run. The served snapshot of every pipeline call is in the run records (`calls`).

## Runs

Each pipeline run builds a new crew. A run that stops with a system error is scored as wrong. A
run interrupted by the API or the graph database before the system answered has produced no output and is run again;
its record keeps the earlier attempt (`earlier_attempts`).
