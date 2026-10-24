# Supply chain disruption monitoring with agentic AI

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sara-almahri/supply-chain-disruption-monitoring/blob/main/notebooks/reproduce_results.ipynb)

Code, evaluation data and stored results of the PhD thesis *Large language models and agentic AI for supply chain
visibility and disruption response* (Sara AlMahri, Department of Engineering, University of Cambridge).

Every reported result can be recomputed from the stored outputs without an API key or a Neo4j instance, in the
browser with the [Colab notebook](notebooks/reproduce_results.ipynb) or locally (below).

## The framework

A pipeline of language-model agents, built on [CrewAI](https://github.com/crewAIInc/crewAI), goes from a news article
to an action for each affected Tier-1 supplier of a monitored firm. The four evaluated agents are:

1. **Disruption Monitor**: decides whether the article reports a disruption under way and records its type, its scope
   and the countries, industries and companies it affects.
2. **Knowledge Graph Query Agent**: calls a tool that traces, in the Neo4j supply network, every Tier-1 to Tier-4 path
   of the monitored firm whose last company is disrupted.
3. **Risk Manager**: calls a tool that computes the risk score of each affected Tier-1 supplier (Equation 4.1).
4. **Chief Supply Chain Officer (CSCO)**: assigns REPLACE_SUPPLIER (score 0.60 or above), INCREASE_MONITORING (0.45 to
   below 0.60) or MAINTAIN_STANDARD and justifies each decision.

The Product Search, Network Visualizer and Alternative Sourcing Agents complete the pipeline of the case study
(Section 4.5). The prompts are in `config/agents.yaml` and `config/tasks.yaml`.

## Contents

| Thesis | Folder | What it holds |
|---|---|---|
| Chapter 3 | `chapter3/` | Evaluation documents, reference annotation, prompts, stored extraction runs, reruns and scorers |
| Chapters 4 and 5 | `agents/`, `tools/`, `config/`, `crew.py`, `main.py` | The pipeline |
| Chapters 4 and 5 | `dataset/` | The supply network and its cleaning |
| Chapters 4 and 5 | `offline/` | The network, path tracing and risk function in Python, without Neo4j |
| Chapter 4 | `evaluation/benchmark_v2/` | The relevance benchmark: specifications, references, articles and scripts |
| Chapter 5 | `evaluation/scenarios/`, `evaluation/ground_truth_final/` | The 30 scenarios and their references |
| Chapters 4 and 5 | `evaluation/results/` | Every stored run and the computed scores |
| All | `analysis/` | The metrics module and the metric definitions (`METRICS.md`) |
| All | `scripts/`, `thesis_figures/` | Run, scoring and figure scripts |
| All | `notebooks/` | The reproduction notebook |

## Key results

| Chapter | Result |
|---|---|
| 3 | Entity extraction F1 0.458, MCC 0.453; relation extraction F1 0.343, MCC 0.354 (GPT-4o, ten runs) |
| 4 | Relevance benchmark: precision 0.992, recall 1.000, specificity 0.999, MCC 0.995 (95% interval 0.986 to 1.000); 0.439 without the ongoing and industry restrictions |
| 5 | Decisions correct in 0.833 (GPT-4o), 0.847 (gpt-4.1) and 0.753 (gpt-5-mini) of runs; within budget 0.640, 0.653, 0.547; the Knowledge Graph Query Agent and the Risk Manager are correct on their input in every run |

## Reproducing the results

No model is called, and no API key or Neo4j instance is needed.

```bash
git clone https://github.com/sara-almahri/supply-chain-disruption-monitoring.git
cd supply-chain-disruption-monitoring
pip install -r requirements.txt

python chapter3/score_ch3.py                       # Chapter 3, Table 3.1
python chapter3/score_reruns.py                    # Chapter 3, Section 3.4.2 and Table 3.4
python -m evaluation.benchmark_v2.score            # Chapter 4, Tables 4.5 to 4.7
python scripts/score_v1_2.py                       # Chapter 5, per-run scores and Table 5.4
python scripts/ch5_analysis_v1_2.py                # Chapter 5, Tables 5.3 to 5.8
python -m offline.check_references --set all       # references of Chapters 4 and 5 from the network
python -m unittest discover -s analysis/tests -t .
python -m unittest discover -s offline/tests -t .
```

Each scorer rewrites its file in `chapter3/results/` or `evaluation/results/`, and section 4 of the notebook compares
the rewritten files with the released ones. The figures are redrawn with `chapter3/figures_ch3.py`, `scripts/figures_ch5_v1_2.py` and
`thesis_figures/figures_ch4_case_study.py`. `ENVIRONMENT.md` lists the package versions and the model snapshots.

## Running the pipeline

Running the agents needs Python 3.12, an OpenAI API key and a Neo4j instance (5.11 or later).

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements-lock.txt
cp .env.example .env                               # Neo4j URI, user and password; OpenAI API key

python scripts/kg_ingestion.py                     # load dataset/supplychainKG_clean.csv into Neo4j
python scripts/generate_company_embeddings.py      # company embeddings for the Product Search Agent
python scripts/verify_kg.py                        # check Neo4j against the CSV

python main.py                                     # one run: company in config/company_config.yaml
DISABLE_PRODUCT_AGENTS=1 python main.py            # the four evaluated agents only
```

The evaluation runners record every API call and stop if a served model snapshot differs from `ENVIRONMENT.md`:

| Script | Runs |
|---|---|
| `scripts/run_evaluation.py` | The 630 runs of Chapter 5 |
| `evaluation/benchmark_v2/run_pipeline.py`, `run_comparison.py` | The benchmark runs of Chapter 4 (`evaluation/benchmark_v2/README.md`) |
| `scripts/run_case_study_v1_2.py` | The case study of Section 4.5 |
| `chapter3/rerun_extraction.py`, `chapter3/independent_annotation.py` | The Chapter 3 reruns and the second annotation |

## Data

- **Supply network** (`dataset/`): 6,222 companies in 96 countries and 24 industries, with 16,640 supply links.
  `supplychainKG.csv` is the source file and `supplychainKG_clean.csv` the cleaned network used by every result
  (`scripts/clean_kg.py`, with the reviewed corrections in `dataset/cleaning/`).
- **Chapter 3 documents** (`chapter3/documents/`): English Wikipedia articles, with the reference annotation in
  `chapter3/reference_final/`.
- **Relevance benchmark** (`evaluation/benchmark_v2/`): synthesised articles; each item is specified first and its
  label, paths, risk scores and actions are computed from the specification.
- **Scenarios** (`evaluation/scenarios/`): ten synthesised articles, each evaluated for BMW, Mercedes-Benz and Tesla,
  with their paraphrases and noisy versions in `evaluation/results/`.
- **Case-study article**: the Encyclopaedia Britannica entry "Russia-Ukraine War" is not redistributed;
  `python -m scripts.run_case_study_v1_2 article` fetches the archived capture and checks it against the hashes in
  `evaluation/results/case_study_v1_2/input/article_meta.json`.

The synthesised articles describe invented events and are not news reports.

## Citation

```bibtex
@misc{almahri2025scframework,
  author       = {AlMahri, Sara},
  title        = {Supply Chain Disruption Monitoring: Framework, Evaluation Data, and Results},
  year         = {2025},
  howpublished = {\url{https://github.com/sara-almahri/supply-chain-disruption-monitoring}}
}
```

Related publications:

- AlMahri, S., Xu, L. and Brintrup, A. (2026). Enhancing supply chain visibility with knowledge graphs and large
  language models. *International Journal of Production Research*, 64(6), 2178–2209.
  [doi:10.1080/00207543.2025.2575841](https://doi.org/10.1080/00207543.2025.2575841)
- AlMahri, S., Xu, L. and Brintrup, A. (2025). Automating supply chain disruption monitoring via an agentic AI
  approach. Available at SSRN 5317578.

## Licence

Code and data: [Creative Commons Attribution-NonCommercial 4.0 International](LICENSE). The Wikipedia text in
`chapter3/documents/` and the files derived from it (`chapter3/reference_final/`, `chapter3/stored_runs/` and
`chapter3/results/reruns/`) are licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
