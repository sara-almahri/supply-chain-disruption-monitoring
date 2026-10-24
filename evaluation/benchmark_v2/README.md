# Relevance benchmark (Chapter 4)

The benchmark of Section 4.4.1, specified in Appendix C.6: news articles evaluated for one monitored firm each (BMW,
Mercedes-Benz or Tesla), with labels computed from each item's specification on the supply network, not read from the
article. The test set has 1,117 items: 120 positives and 997 negatives of four types.

| Type | Items | What it tests |
| --- | ---: | --- |
| P | 120 | A disruption under way that reaches the firm's Tier-1 to Tier-4 suppliers |
| N1 | 300 | Business news with no disruption, naming parts of the firm's network |
| N2 | 250 | A disruption under way in a country in none of the three networks |
| N3 | 150 | A disruption that is over, or a historical account, where a disruption under way would be positive |
| N4 | 297 | A disruption confined to an industry from which the firm has no supplier in that country |

## Files

| Path | Contents |
| --- | --- |
| `items/test/specs.json` | The specification of every item (seed 20261014) |
| `items/test/ground_truth_final/<item>.json` | Label, disrupted paths, Tier-1 risk scores and expected actions of every item |
| `items/test/articles/<item>.json` | The article of every item, its label and its status |
| `items/test/test_set.json` | The 1,117 items of the test set |
| `items/test/repeat_sample.json` | The 112 items of the five further runs |
| `prompts/` | The prompts of the generator, the fidelity check and the confinement check |
| `../results/benchmark_v2/test/` | The runs of the primary system and of the comparison, and `scores.json` |

The status of an article is `accepted` (it passed the checks), `audited` (it failed a check after regeneration and the
audit confirmed its label) or `dropped` (the audit did not confirm its label; N4-029, N4-127 and N4-186, which are not
in the test set).

## Scripts

Run from the repository root, in this order. Only the scripts marked with an asterisk call a model or need Neo4j.

```bash
python -m evaluation.benchmark_v2.specify          # items/test/specs.json
python -m evaluation.benchmark_v2.ground_truth     # items/test/ground_truth_final/
python -m evaluation.benchmark_v2.generate --max-usd 40    # * articles (OpenAI API)
python -m evaluation.benchmark_v2.repeat_sample    # items/test/repeat_sample.json
python -m evaluation.benchmark_v2.run_pipeline     # * primary system (OpenAI API and Neo4j)
python -m evaluation.benchmark_v2.run_comparison   # * comparison (Neo4j)
python -m evaluation.benchmark_v2.score            # Tables 4.5 to 4.7 -> results/benchmark_v2/test/scores.json
```

`python -m offline.check_references --set benchmark` recomputes the ground truth of every test item on the released
network and compares it with the stored files.
