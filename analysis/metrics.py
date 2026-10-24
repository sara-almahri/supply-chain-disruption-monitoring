"""
Shared metrics for every result reported in the thesis (definitions in analysis/METRICS.md).

Every precision, recall, F1, MCC, interval, baseline and prevalence projection is computed with this module and with
nothing else. The module has no I/O: scoring scripts load outputs and ground truth, build confusion matrices with the
functions below, and report the values these functions return.

Undefined values are returned as math.nan and must be reported as "undefined", never as 0 or 1:
  - precision when nothing is predicted positive; recall when there are no positives; specificity when there are no
    negatives;
  - MCC when the ground truth contains only one class (the decisions contain nothing to discriminate).
When the ground truth contains both classes but the prediction is constant, MCC is 0 (no predictive value).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import beta

NAN = math.nan
DEFAULT_SEED = 20261011
DEFAULT_BOOT = 10_000

# Chapter 4 evaluation mapping from a Tier-1 risk score to a risk level and a CSCO action (raw score, no rounding).
HIGH_RISK = 0.60
MEDIUM_RISK = 0.45
RISK_LEVELS = ("NONE", "LOW", "MEDIUM", "HIGH")
ACTIONS = ("MAINTAIN_STANDARD", "INCREASE_MONITORING", "REPLACE_SUPPLIER")
TIERS = ("tier_1", "tier_2", "tier_3", "tier_4")


# ---------------------------------------------------------------------------------------------------------------------
# String matching and the set-based scores of the reported results
# ---------------------------------------------------------------------------------------------------------------------
def normalize(s: Optional[str]) -> str:
    """Normalisation used for every string comparison (identical to evaluation/evaluation_harness.py)."""
    if not s:
        return ""
    s = s.strip().lower()
    s = s.replace("&", " ").replace(" and ", " ")
    s = s.replace("labour", "labor")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_set(items: Iterable[Optional[str]]) -> Set[str]:
    return {normalize(x) for x in items if x and normalize(x)}


def entity_keys(kind: str, names: Iterable[Optional[str]], resolver=None) -> Set[str]:
    """Comparison keys of extracted countries, industries or companies: the knowledge-graph name when the name resolves
    (offline.names.Resolver, as the pipeline resolves it), otherwise the normalised string."""
    if resolver is None:
        from offline.names import default_resolver
        resolver = default_resolver()
    keys = set()
    for name in names or []:
        if not name or not normalize(name):
            continue
        if kind == "countries":
            resolved = resolver.country(name)
        elif kind == "industries":
            resolved = resolver.industry(name)
        elif kind == "companies":
            spellings = resolver.company(name)
            resolved = sorted(spellings)[0] if spellings else None
        else:
            raise ValueError(kind)
        keys.add(normalize(resolved or name))
    return keys


def set_prf(pred: Iterable[str], gold: Iterable[str]) -> Tuple[float, float, float]:
    """Precision, recall and F1 of one predicted set against one gold set, with the conventions of the reported
    results: both empty scores 1, 1, 1; an empty prediction against a non-empty gold set scores 1, 0, 0; a non-empty
    prediction against an empty gold set scores 0, 1, 0."""
    p, g = norm_set(pred), norm_set(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p:
        return 1.0, 0.0, 0.0
    if not g:
        return 0.0, 1.0, 0.0
    tp = len(p & g)
    prec, rec = tp / len(p), tp / len(g)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def jaccard(a: Set[Hashable], b: Set[Hashable]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def match_sets(pred: Sequence[frozenset], gold: Sequence[frozenset], threshold: float = 1.0) -> Tuple[int, int, int]:
    """One-to-one matching of predicted to gold items (each item a set, e.g. a path as a set of nodes).

    A pair matches when its Jaccard similarity is at least `threshold`; pairs are taken in decreasing order of
    similarity and each item is used at most once. threshold=1.0 is exact set equality. Duplicate items count once.
    Returns (tp, fp, fn)."""
    p, g = list(dict.fromkeys(pred)), list(dict.fromkeys(gold))
    if threshold >= 1.0:
        tp = len(set(p) & set(g))
        return tp, len(p) - tp, len(g) - tp
    pairs = []
    for i, a in enumerate(p):
        for j, b in enumerate(g):
            s = jaccard(set(a), set(b))
            if s >= threshold:
                pairs.append((s, i, j))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_p, used_g = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    return tp, len(p) - tp, len(g) - tp


# ---------------------------------------------------------------------------------------------------------------------
# Binary confusion matrix
# ---------------------------------------------------------------------------------------------------------------------
def _ratio(num: float, den: float) -> float:
    return num / den if den else NAN


@dataclass(frozen=True)
class Confusion:
    """Binary confusion matrix. Counts may be expected (non-integer) values, as in the prevalence projection."""
    tp: float = 0
    fp: float = 0
    fn: float = 0
    tn: float = 0

    def __add__(self, other: "Confusion") -> "Confusion":
        return Confusion(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)

    @classmethod
    def from_labels(cls, y_true: Iterable[bool], y_pred: Iterable[bool]) -> "Confusion":
        tp = fp = fn = tn = 0
        y_true, y_pred = list(y_true), list(y_pred)
        if len(y_true) != len(y_pred):
            raise ValueError(f"{len(y_true)} true labels but {len(y_pred)} predictions")
        for t, p in zip(y_true, y_pred):
            if t and p:
                tp += 1
            elif p:
                fp += 1
            elif t:
                fn += 1
            else:
                tn += 1
        return cls(tp, fp, fn, tn)

    @classmethod
    def from_sets(cls, pred: Set[Hashable], gold: Set[Hashable], universe: Set[Hashable]) -> "Confusion":
        """One decision per element of the universe (is the element in the set?). Predicted or gold elements outside
        the universe are added to it, so a prediction outside the vocabulary is a false positive."""
        u = set(universe) | pred | gold
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        return cls(tp, fp, fn, len(u) - tp - fp - fn)

    @classmethod
    def from_matches(cls, tp: float, fp: float, fn: float, n_decisions: float) -> "Confusion":
        """Confusion matrix from matched counts and the total number of decisions (size of the universe)."""
        tn = n_decisions - tp - fp - fn
        if tn < 0:
            raise ValueError("n_decisions is smaller than tp + fp + fn")
        return cls(tp, fp, fn, tn)

    @property
    def n(self) -> float:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def positives(self) -> float:
        return self.tp + self.fn

    @property
    def negatives(self) -> float:
        return self.tn + self.fp

    @property
    def prevalence(self) -> float:
        return _ratio(self.positives, self.n)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.positives)

    sensitivity = recall

    @property
    def specificity(self) -> float:
        return _ratio(self.tn, self.negatives)

    @property
    def npv(self) -> float:
        return _ratio(self.tn, self.tn + self.fn)

    @property
    def accuracy(self) -> float:
        return _ratio(self.tp + self.tn, self.n)

    @property
    def f1(self) -> float:
        return _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def mcc(self) -> float:
        if self.positives == 0 or self.negatives == 0:
            return NAN
        pred_pos, pred_neg = self.tp + self.fp, self.tn + self.fn
        if pred_pos == 0 or pred_neg == 0:
            return 0.0
        num = self.tp * self.tn - self.fp * self.fn
        return num / math.sqrt(pred_pos * self.positives * self.negatives * pred_neg)

    def as_dict(self) -> Dict[str, float]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn, "n": self.n,
                "prevalence": self.prevalence, "precision": self.precision, "recall": self.recall,
                "specificity": self.specificity, "f1": self.f1, "mcc": self.mcc, "accuracy": self.accuracy}


# ---------------------------------------------------------------------------------------------------------------------
# Multiclass confusion matrix (MCC as the R_K statistic of Gorodkin, 2004)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MulticlassConfusion:
    labels: Tuple[Hashable, ...]
    matrix: Tuple[Tuple[float, ...], ...]  # matrix[i][j]: true label i, predicted label j

    @classmethod
    def from_labels(cls, y_true: Iterable[Hashable], y_pred: Iterable[Hashable],
                    labels: Sequence[Hashable]) -> "MulticlassConfusion":
        index = {lab: i for i, lab in enumerate(labels)}
        m = [[0] * len(labels) for _ in labels]
        y_true, y_pred = list(y_true), list(y_pred)
        if len(y_true) != len(y_pred):
            raise ValueError(f"{len(y_true)} true labels but {len(y_pred)} predictions")
        for t, p in zip(y_true, y_pred):
            if t not in index or p not in index:
                raise ValueError(f"label outside {tuple(labels)}: true={t!r}, predicted={p!r}")
            m[index[t]][index[p]] += 1
        return cls(tuple(labels), tuple(tuple(r) for r in m))

    def __add__(self, other: "MulticlassConfusion") -> "MulticlassConfusion":
        if self.labels != other.labels:
            raise ValueError("label sets differ")
        return MulticlassConfusion(self.labels, tuple(tuple(a + b for a, b in zip(r1, r2))
                                                      for r1, r2 in zip(self.matrix, other.matrix)))

    @property
    def n(self) -> float:
        return sum(map(sum, self.matrix))

    @property
    def accuracy(self) -> float:
        return _ratio(sum(self.matrix[i][i] for i in range(len(self.labels))), self.n)

    @property
    def mcc(self) -> float:
        k = len(self.labels)
        t = [sum(self.matrix[i]) for i in range(k)]                       # true count per class
        p = [sum(self.matrix[i][j] for i in range(k)) for j in range(k)]  # predicted count per class
        s, c = self.n, sum(self.matrix[i][i] for i in range(k))
        if sum(1 for x in t if x) < 2:
            return NAN
        if sum(1 for x in p if x) < 2:
            return 0.0
        num = c * s - sum(p[i] * t[i] for i in range(k))
        return num / math.sqrt((s * s - sum(x * x for x in p)) * (s * s - sum(x * x for x in t)))

    def one_vs_rest(self, label: Hashable) -> Confusion:
        i = self.labels.index(label)
        k = len(self.labels)
        tp = self.matrix[i][i]
        fn = sum(self.matrix[i]) - tp
        fp = sum(self.matrix[r][i] for r in range(k)) - tp
        return Confusion(tp, fp, fn, self.n - tp - fp - fn)


# ---------------------------------------------------------------------------------------------------------------------
# Units of decision of Chapters 4 and 5
# ---------------------------------------------------------------------------------------------------------------------
def n_paths(kg_results: Optional[Mapping[str, Any]]) -> int:
    """Number of disrupted paths, Tier 1 to Tier 4, in a knowledge-graph result (path lists or chain counts)."""
    if not isinstance(kg_results, Mapping):
        return 0
    if any(t in kg_results for t in TIERS):
        return sum(len(kg_results.get(t) or []) for t in TIERS)
    counts = kg_results.get("chain_counts")
    if isinstance(counts, Mapping):
        return int(sum(int(counts.get(t) or 0) for t in TIERS))
    return 0


def alert(kg_results: Optional[Mapping[str, Any]]) -> bool:
    """Scenario-level decision: an alert is raised when at least one disrupted path to the monitored firm exists."""
    return n_paths(kg_results) > 0


def path_node_set(path: Iterable[Mapping[str, Any]]) -> frozenset:
    """A path as the order-insensitive set of its (company, country, industry) nodes."""
    return frozenset((normalize(n.get("company", "")), normalize(n.get("country", "")), normalize(n.get("industry", "")))
                     for n in path if isinstance(n, Mapping))


def risk_level(score: Optional[float]) -> str:
    """Risk level of a Tier-1 supplier; NONE when the supplier has no score (not exposed)."""
    if score is None:
        return "NONE"
    if score >= HIGH_RISK:
        return "HIGH"
    if score >= MEDIUM_RISK:
        return "MEDIUM"
    return "LOW"


def expected_action(score: Optional[float]) -> str:
    """CSCO action implied by a Tier-1 risk score; MAINTAIN_STANDARD when the supplier has no score."""
    return {"HIGH": "REPLACE_SUPPLIER", "MEDIUM": "INCREASE_MONITORING"}.get(risk_level(score), "MAINTAIN_STANDARD")


# ---------------------------------------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------------------------------------
def constant_baseline(n_pos: int, n_neg: int, predict_positive: bool) -> Confusion:
    """Classifier that gives the same answer to every decision (always alert / never alert)."""
    return Confusion(n_pos, n_neg, 0, 0) if predict_positive else Confusion(0, 0, n_pos, n_neg)


def random_baseline(n_pos: float, n_neg: float, positive_rate: float) -> Confusion:
    """Expected confusion matrix of a classifier that predicts positive at random with probability `positive_rate`,
    independently of the input. Its MCC is 0 and its F1 is 2*pi*q / (pi + q) for prevalence pi and rate q."""
    q = positive_rate
    return Confusion(n_pos * q, n_neg * q, n_pos * (1 - q), n_neg * (1 - q))


def majority_label(y_true: Iterable[Hashable]) -> Hashable:
    values = list(y_true)
    return max(sorted(set(values), key=str), key=values.count)


# ---------------------------------------------------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------------------------------------------------
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact two-sided (1 - alpha) interval for a binomial proportion k / n."""
    if n <= 0:
        return NAN, NAN
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def bootstrap(units: Sequence[Any], statistic: Callable[[Sequence[Any]], float], *,
              strata: Optional[Sequence[Hashable]] = None, n_boot: int = DEFAULT_BOOT, seed: int = DEFAULT_SEED,
              alpha: float = 0.05) -> Dict[str, float]:
    """Percentile bootstrap interval of `statistic` over resampled units.

    A unit is everything that belongs to one scenario (all its decisions and runs), so decisions within a scenario are
    resampled together. With `strata` (e.g. positive / negative scenario), units are resampled within each stratum and
    the stratum sizes are kept. Resamples on which the statistic is undefined are counted and excluded."""
    rng = np.random.default_rng(seed)
    units = list(units)
    groups: Dict[Hashable, List[int]] = {}
    for i, s in enumerate(strata if strata is not None else [0] * len(units)):
        groups.setdefault(s, []).append(i)
    estimates = []
    undefined = 0
    for _ in range(n_boot):
        idx = [j for members in groups.values() for j in rng.choice(members, size=len(members), replace=True)]
        v = statistic([units[j] for j in idx])
        if math.isnan(v):
            undefined += 1
        else:
            estimates.append(v)
    point = statistic(units)
    if not estimates:
        return {"estimate": point, "lo": NAN, "hi": NAN, "n_boot": n_boot, "n_undefined": undefined}
    lo, hi = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {"estimate": point, "lo": float(lo), "hi": float(hi), "n_boot": n_boot, "n_undefined": undefined}


# ---------------------------------------------------------------------------------------------------------------------
# Prevalence
# ---------------------------------------------------------------------------------------------------------------------
def expected_confusion(sensitivity: float, specificity: float, prevalence: float, n: float = 1.0) -> Confusion:
    """Expected confusion matrix of n decisions at a given prevalence, for fixed sensitivity and specificity."""
    pos, neg = prevalence * n, (1 - prevalence) * n
    return Confusion(pos * sensitivity, neg * (1 - specificity), pos * (1 - sensitivity), neg * specificity)


def mcc_at_prevalence(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Closed form of expected_confusion(...).mcc:
    (sens + spec - 1) * sqrt(pi (1 - pi)) / sqrt(q (1 - q)), with q = pi sens + (1 - pi)(1 - spec)."""
    return expected_confusion(sensitivity, specificity, prevalence).mcc


def mixed_specificity(specificity_by_type: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Specificity over a mixture of negative types with the stated weights (normalised to sum to 1)."""
    total = sum(weights.values())
    if set(weights) - set(specificity_by_type):
        raise ValueError("a weighted negative type has no specificity")
    return sum(specificity_by_type[t] * w / total for t, w in weights.items())


def project(tp: int, n_pos: int, tn_by_type: Mapping[str, int], n_by_type: Mapping[str, int],
            weights: Mapping[str, float], prevalences: Sequence[float], *, n_draws: int = DEFAULT_BOOT,
            seed: int = DEFAULT_SEED, alpha: float = 0.05) -> List[Dict[str, float]]:
    """Projected precision, F1 and MCC at prevalences below those observed.

    Sensitivity (tp / n_pos) and the specificity of each negative type (tn / n) are measured on the benchmark and
    assumed to hold at the lower prevalence; the negative types are mixed with the stated weights. Uncertainty is
    propagated by drawing each proportion from its Jeffreys posterior Beta(k + 1/2, n - k + 1/2)."""
    rng = np.random.default_rng(seed)
    sens_draws = rng.beta(tp + 0.5, n_pos - tp + 0.5, n_draws)
    spec_draws = {t: rng.beta(tn_by_type[t] + 0.5, n_by_type[t] - tn_by_type[t] + 0.5, n_draws) for t in weights}
    total_w = sum(weights.values())
    spec_mix_draws = sum(spec_draws[t] * weights[t] / total_w for t in weights)
    sens = tp / n_pos
    spec = mixed_specificity({t: tn_by_type[t] / n_by_type[t] for t in weights}, weights)
    rows = []
    for pi in prevalences:
        point = expected_confusion(sens, spec, pi)
        draws = [expected_confusion(s, c, pi) for s, c in zip(sens_draws, spec_mix_draws)]
        row = {"prevalence": pi, "sensitivity": sens, "specificity": spec}
        for name in ("precision", "f1", "mcc"):
            vals = np.array([getattr(d, name) for d in draws], dtype=float)
            vals = vals[~np.isnan(vals)]
            lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if len(vals) else (NAN, NAN)
            row.update({name: getattr(point, name), f"{name}_lo": float(lo), f"{name}_hi": float(hi)})
        rows.append(row)
    return rows


def subsample_at_prevalence(positives: Sequence[Any], negatives: Sequence[Any], prevalence: float,
                            statistic: Callable[[Sequence[Any], Sequence[Any]], float], *, n_rep: int = 1_000,
                            seed: int = DEFAULT_SEED) -> Dict[str, float]:
    """Observed metric at a lower prevalence: all negatives are kept and positives are drawn without replacement so
    that positives / (positives + negatives) is as close as possible to `prevalence`, repeated n_rep times.
    The range reflects which positives are drawn, not sampling uncertainty in the benchmark."""
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be in (0, 1)")
    k = round(prevalence * len(negatives) / (1 - prevalence))
    if not 1 <= k <= len(positives):
        raise ValueError(f"prevalence {prevalence} needs {k} positives; {len(positives)} available")
    rng = np.random.default_rng(seed)
    vals = np.array([statistic([positives[i] for i in rng.choice(len(positives), size=k, replace=False)], negatives)
                     for _ in range(n_rep)], dtype=float)
    vals = vals[~np.isnan(vals)]
    lo, med, hi = np.quantile(vals, [0.025, 0.5, 0.975]) if len(vals) else (NAN, NAN, NAN)
    return {"prevalence": k / (k + len(negatives)), "n_pos": k, "n_neg": len(negatives),
            "median": float(med), "lo": float(lo), "hi": float(hi), "n_rep": n_rep}
