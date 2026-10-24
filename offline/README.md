# offline

The knowledge graph, path tracing and risk scoring in Python (NetworkX), over `dataset/supplychainKG_clean.csv`. No
Neo4j instance and no API key are needed. The references of Chapters 4 and 5 are computed with this package.

- `kg.py`: the network, read with the same rules as `scripts/kg_ingestion.py`; the disruption rule (`is_disrupted`)
  and the disrupted paths of a firm, Tier 1 to Tier 4, with no company twice on a path.
- `names.py`: resolution of country, industry and company names to their spelling in the graph.
- `risk.py`: Tier-1 supplier risk scores (Equation 4.1) and the risk levels.
- `check_references.py`: recomputes the stored references and reports any difference.
- `tests/`: unit tests.

```bash
python -m offline.check_references --set scenarios   # the 30 scenarios of Chapter 5, under a minute
python -m offline.check_references --set benchmark   # the 1,117 benchmark items of Chapter 4, about four minutes
python -m unittest discover -s offline/tests -t .
```
