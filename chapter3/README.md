# Chapter 3: knowledge-graph construction

Documents, ground truth, prompts, stored extraction outputs and scoring scripts of the extraction study of Chapter 3.

| Path | Contents |
| --- | --- |
| `documents/` | The evaluation documents (English Wikipedia, CC BY-SA 4.0) |
| `reference_final/` | The ground truth, one file per document |
| `prompts/` | The extraction prompt as used, its two variants, and the prompt of the second annotation |
| `stored_runs/runs/run_1` ... `run_10` | The ten reported GPT-4o runs |
| `results/reruns/<arm>/run_<k>/` | The reruns of Section 3.4.2: prompt variants and other models, with `calls.jsonl` |
| `results/ch3_scores.json` | Scores of the reported runs (Table 3.1) |
| `results/reruns_scores.json` | Scores of the reruns with paired intervals (Table 3.4) |
| `figures/` | The count figures of Section 3.4 |

## Reproducing the results

No API key is needed to score the stored outputs:

```bash
python chapter3/score_ch3.py      # Table 3.1  -> results/ch3_scores.json
python chapter3/score_reruns.py   # Table 3.4  -> results/reruns_scores.json
python chapter3/figures_ch3.py    # figures/
```

`rerun_extraction.py` and `independent_annotation.py` call the OpenAI API (key in `.env`) and do not repeat a call
whose output is already saved.
