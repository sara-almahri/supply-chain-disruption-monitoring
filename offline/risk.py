"""
Tier-1 supplier risk scores (Equation 4.1), used by the pipeline's risk tool and by the references.

For each Tier-1 supplier i with at least one disrupted chain:
    exposure_i    = sum over tiers t of w_t x (number of distinct disrupted (company, country) pairs at tier t on chains
                    through i), with w = 1.0, 0.75, 0.5, 0.35 for tiers 1-4
    dependency_i  = share of i's own upstream network (i and the companies on simple paths of length 1 to 3 from
                    it) that meets the disruption rule (kg.is_disrupted)
    sub_central_i = mean centrality score of the companies after i on its disrupted chains
    sub_pagerank_i= mean PageRank of the same companies
    central_i     = centrality score of i
    depth_i       = deepest tier with a disrupted chain through i, divided by 4
    risk_i = 0.35 x mm(exposure) + 0.25 x dependency + 0.20 x max(mm(sub_central), mm(sub_pagerank))
             + 0.10 x mm(central) + 0.10 x depth,  clipped to [0, 1] and rounded to 4 decimals,
where mm is min-max scaling over the firm's affected Tier-1 suppliers (1 for all when all values are equal and positive,
0 when all are zero). The centrality score of a node is the mean of its normalised betweenness, its closeness to the
firm (1 / (d + 1), d the number of steps to the firm), its eigenvector centrality and its degree centrality (mean of
in-degree and out-degree, each divided by the maximum in the graph). The metrics graph is oriented from the firm towards
its suppliers.
"""

from typing import Dict, Iterable, Optional, Set, Tuple

import networkx as nx

WEIGHTS = {"exposure": 0.35, "dependency": 0.25, "sub_criticality": 0.20, "central": 0.10, "depth": 0.10}
TIER_WEIGHTS = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.35}


def build_graph(firm: str, edges: Iterable[Tuple[str, str]]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(firm)
    g.add_edges_from(sorted(edges))  # fixed order: floating-point sums do not depend on set iteration
    return g


def centrality_scores(g: nx.DiGraph, firm: str) -> Dict[str, float]:
    betweenness = nx.betweenness_centrality(g, normalized=True)
    steps_to_firm = nx.single_source_shortest_path_length(g, firm)
    closeness = {n: 1.0 / (steps_to_firm[n] + 1) if n in steps_to_firm else 0.0 for n in g.nodes()}
    try:
        eigen = nx.eigenvector_centrality(g, max_iter=1000, tol=1e-06)
    except Exception:
        deg = dict(g.degree())
        top = max(deg.values()) if deg else 1.0
        eigen = {k: v / top for k, v in deg.items()}
    ind, outd = dict(g.in_degree()), dict(g.out_degree())
    mi, mo = max(ind.values()), max(outd.values())
    score = {}
    for n in g.nodes():
        degree = ((ind[n] / mi if mi else 0.0) + (outd[n] / mo if mo else 0.0)) / 2
        parts = [betweenness.get(n, 0.0), closeness[n], eigen.get(n, 0.0), degree]
        score[n] = min(max(sum(min(max(p, 0.0), 1.0) for p in parts) / 4, 0.0), 1.0)
    return score


def min_max(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: 1.0 if v > 0 else 0.0 for k, v in values.items()}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def exposures(kg_results: Dict) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for t in range(1, 5):
        for chain in kg_results.get(f"tier_{t}", []):
            if len(chain) < 2:
                continue
            e = out.setdefault(chain[1]["company"], {"finals": {k: set() for k in range(1, 5)}, "after": set()})
            e["after"].update(n["company"] for n in chain[1:])
            e["finals"][len(chain) - 1].add((chain[-1]["company"], chain[-1]["country"]))
    return out


def own_network(net, supplier: str, mode: str = "simple", extra_tiers: int = 3) -> Set[str]:
    """The Tier-1 supplier and every company on a simple path of length 1 to extra_tiers starting from it
    (its own upstream network)."""
    from offline import kg as _kg
    nodes = {supplier}
    for length in range(1, extra_tiers + 1):
        for p in _kg.paths(net, supplier, length, mode):
            nodes.update(p)
    return nodes


def risk_scores(kg_results: Dict, firm: str, edges: Set[Tuple[str, str]], net, countries: Iterable[str] = (),
                companies: Iterable[str] = (), industries: Optional[Iterable[str]] = None
                ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Scores and components of every affected Tier-1 supplier. countries, companies and industries are graph
    spellings; industries is None when the disruption is not confined to named industries.
    """
    exp = exposures(kg_results)
    if not exp:
        return {}, {}
    g = build_graph(firm, edges)
    central = centrality_scores(g, firm)
    pagerank = nx.pagerank(g, alpha=0.85, max_iter=100)
    countries_set = {str(c).strip() for c in countries if str(c).strip()}
    companies_set = {str(c).strip() for c in companies if str(c).strip()}
    industries_set = None if industries is None else {str(i).strip() for i in industries if str(i).strip()}
    from offline import kg as _kg

    raw = {k: {} for k in ("exposure", "dependency", "sub_central", "sub_pagerank", "central", "depth")}
    for i, e in exp.items():
        counts = {t: len(e["finals"][t]) for t in range(1, 5)}
        raw["exposure"][i] = sum(counts[t] * TIER_WEIGHTS[t] for t in range(1, 5))
        nodes = own_network(net, i, "simple", extra_tiers=3)
        hit = {n for n in nodes if _kg.is_disrupted(net, n, countries_set, companies_set, industries_set)}
        raw["dependency"][i] = (len(hit) / len(nodes)) if nodes else 0.0
        after = sorted(n for n in e["after"] if n in g)
        raw["sub_central"][i] = sum(central[n] for n in after) / len(after) if after else 0.0
        raw["sub_pagerank"][i] = sum(pagerank[n] for n in after) / len(after) if after else 0.0
        raw["central"][i] = central.get(i, 0.0)
        raw["depth"][i] = max([t for t in range(1, 5) if counts[t]], default=0)

    mm = {k: min_max(raw[k]) for k in ("exposure", "sub_central", "sub_pagerank", "central")}
    scores, parts = {}, {}
    for i in exp:
        p = {"exposure": mm["exposure"][i], "dependency": raw["dependency"][i],
             "sub_criticality": max(mm["sub_central"][i], mm["sub_pagerank"][i]),
             "central": mm["central"][i], "depth": raw["depth"][i] / 4}
        scores[i] = round(min(max(sum(WEIGHTS[k] * p[k] for k in WEIGHTS), 0.0), 1.0), 4)
        parts[i] = {**p, "raw": {k: raw[k][i] for k in raw}}
    return scores, parts


def risk_level(score: float) -> str:
    """HIGH for a score of 0.60 or above, MEDIUM from 0.45 to below 0.60, LOW otherwise, on the unrounded score."""
    return "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.45 else "LOW"
