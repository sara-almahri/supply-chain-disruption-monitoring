"""Tests for analysis/metrics.py.

    python -m unittest discover -s analysis/tests -t .

Reference values marked "scikit-learn" were computed with scikit-learn 1.6.1 (matthews_corrcoef, f1_score) and those
marked "statsmodels" with statsmodels proportion_confint(method="beta"), outside the locked environment.
"""

import json
import math
import unittest
from pathlib import Path

from analysis import metrics as m
from evaluation import evaluation_harness as harness

ROOT = Path(__file__).resolve().parents[2]


def labels(tp, fp, fn, tn):
    return [True] * tp + [False] * fp + [True] * fn + [False] * tn, [True] * tp + [True] * fp + [False] * fn + [False] * tn


class AlwaysPositiveExample(unittest.TestCase):
    """27 positives and 3 negatives; a classifier that always predicts positive."""

    def test_always_positive(self):
        c = m.constant_baseline(27, 3, predict_positive=True)
        self.assertEqual((c.tp, c.fp, c.fn, c.tn), (27, 3, 0, 0))
        self.assertAlmostEqual(c.precision, 0.9)
        self.assertEqual(c.recall, 1.0)
        self.assertAlmostEqual(c.f1, 0.947368421053, places=12)  # scikit-learn
        self.assertEqual(round(c.f1, 3), 0.947)
        self.assertEqual(c.mcc, 0.0)
        self.assertEqual(c.specificity, 0.0)

    def test_from_labels_gives_the_same_matrix(self):
        t, p = labels(27, 3, 0, 0)
        self.assertEqual(m.Confusion.from_labels(t, p), m.constant_baseline(27, 3, True))


class BinaryMCC(unittest.TestCase):
    CASES = {  # (tp, fp, fn, tn): (MCC, F1), scikit-learn
        (20, 7, 5, 68): (0.689242311266, 0.769230769231),
        (3, 1, 2, 994): (0.669360463429, 0.666666666667),
        (1, 0, 0, 999): (1.0, 1.0),
        (0, 0, 5, 5): (0.0, 0.0),
    }

    def test_against_reference(self):
        for counts, (mcc, f1) in self.CASES.items():
            c = m.Confusion(*counts)
            self.assertAlmostEqual(c.mcc, mcc, places=12, msg=counts)
            self.assertAlmostEqual(c.f1, f1, places=12, msg=counts)

    def test_perfect_and_inverted(self):
        self.assertEqual(m.Confusion(10, 0, 0, 90).mcc, 1.0)
        self.assertEqual(m.Confusion(0, 90, 10, 0).mcc, -1.0)

    def test_undefined_when_truth_has_one_class(self):
        only_negatives = m.Confusion(0, 2, 0, 98)
        self.assertTrue(math.isnan(only_negatives.mcc))
        self.assertTrue(math.isnan(only_negatives.recall))
        self.assertAlmostEqual(only_negatives.specificity, 0.98)
        self.assertTrue(math.isnan(m.Confusion(5, 0, 1, 0).mcc))

    def test_constant_prediction_is_zero(self):
        self.assertEqual(m.constant_baseline(10, 990, predict_positive=False).mcc, 0.0)
        self.assertTrue(math.isnan(m.constant_baseline(10, 990, predict_positive=False).precision))

    def test_pooling(self):
        a, b = m.Confusion(1, 2, 3, 4), m.Confusion(10, 20, 30, 40)
        self.assertEqual(a + b, m.Confusion(11, 22, 33, 44))

    def test_from_sets(self):
        c = m.Confusion.from_sets({"a", "b", "x"}, {"a", "c"}, universe={"a", "b", "c", "d", "e"})
        self.assertEqual((c.tp, c.fp, c.fn, c.tn), (1, 2, 1, 2))  # "x" lies outside the universe: a false positive
        with self.assertRaises(ValueError):
            m.Confusion.from_matches(5, 5, 5, 10)

    def test_mcc_increases_with_true_negatives(self):
        for tp, fp, fn in [(5, 3, 2), (1, 10, 10), (0, 4, 4), (30, 0, 1)]:
            values = [m.Confusion(tp, fp, fn, tn).mcc for tn in (1, 2, 5, 10, 100, 10_000)]
            self.assertEqual(values, sorted(values), msg=(tp, fp, fn))
            self.assertEqual(len(set(values)), len(values), msg=(tp, fp, fn))

    def test_large_negative_universe_approaches_fowlkes_mallows(self):
        c = m.Confusion(30, 10, 20, 10**9)
        self.assertAlmostEqual(c.mcc, math.sqrt(c.precision * c.recall), places=6)


class MulticlassMCC(unittest.TestCase):
    T3 = [2, 1, 2, 2, 1, 2, 2, 0, 0, 0, 0, 2, 2, 0, 1, 2, 0, 2, 0, 1, 2, 0, 1, 0, 2, 0, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1,
          1, 2, 1, 0, 2, 0, 2, 1, 0, 0, 1, 0, 0, 1, 2, 1, 2, 2, 2, 1, 1, 1]
    P3 = [2, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 1, 2, 2, 2, 0, 1, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 1,
          1, 1, 1, 0, 2, 2, 2, 1, 2, 0, 1, 0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1]
    T4 = [0, 1, 3, 0, 2, 2, 1, 0, 2, 1, 3, 2, 1, 0, 2, 1, 2, 3, 1, 3, 2, 2, 3, 3, 3, 3, 0, 1, 3, 1, 0, 2, 3, 2, 3, 2, 0, 3,
          1, 3, 3, 3, 0, 3, 2, 1, 3, 3, 2, 0, 2, 0, 0, 0, 0, 3, 1, 0, 0, 1, 1, 1, 3, 3, 2, 3, 1, 0, 1, 0, 3, 0, 1, 0, 3, 0,
          2, 3, 2, 3]
    P4 = [3, 3, 1, 2, 2, 3, 2, 0, 3, 1, 2, 1, 1, 0, 2, 1, 1, 2, 0, 0, 2, 2, 3, 1, 0, 3, 0, 2, 3, 1, 2, 3, 3, 2, 3, 2, 0, 3,
          1, 3, 3, 0, 0, 3, 2, 3, 2, 0, 2, 2, 2, 1, 3, 0, 0, 3, 1, 0, 0, 3, 1, 0, 1, 0, 1, 3, 2, 3, 3, 0, 1, 3, 1, 0, 2, 0,
          3, 3, 1, 2]

    def test_against_reference(self):
        self.assertAlmostEqual(m.MulticlassConfusion.from_labels(self.T3, self.P3, (0, 1, 2)).mcc,
                               0.654527856953, places=12)  # scikit-learn
        self.assertAlmostEqual(m.MulticlassConfusion.from_labels(self.T4, self.P4, (0, 1, 2, 3)).mcc,
                               0.346009899947, places=12)  # scikit-learn

    def test_two_classes_equal_binary(self):
        t, p = labels(20, 7, 5, 68)
        mc = m.MulticlassConfusion.from_labels(t, p, (True, False))
        self.assertAlmostEqual(mc.mcc, m.Confusion(20, 7, 5, 68).mcc, places=12)
        self.assertEqual(mc.one_vs_rest(True), m.Confusion(20, 7, 5, 68))

    def test_majority_baseline_is_zero_and_single_class_truth_undefined(self):
        truth = ["MAINTAIN_STANDARD"] * 90 + ["REPLACE_SUPPLIER"] * 10
        majority = m.majority_label(truth)
        self.assertEqual(majority, "MAINTAIN_STANDARD")
        mc = m.MulticlassConfusion.from_labels(truth, [majority] * 100, m.ACTIONS)
        self.assertEqual(mc.accuracy, 0.9)
        self.assertEqual(mc.mcc, 0.0)
        single = m.MulticlassConfusion.from_labels(["LOW"] * 5, ["LOW", "HIGH", "LOW", "LOW", "LOW"], m.RISK_LEVELS)
        self.assertTrue(math.isnan(single.mcc))

    def test_rejects_unknown_label(self):
        with self.assertRaises(ValueError):
            m.MulticlassConfusion.from_labels(["LOW"], ["CRITICAL"], m.RISK_LEVELS)


class LegacyConventions(unittest.TestCase):
    """The set-based scores of the reported results are unchanged."""

    STRINGS = ["Metals & Mining", "metals and mining", "  Labour  Unrest ", "Labor unrest", "", None, "South Africa"]
    SETS = [([], []), (["a"], []), ([], ["a"]), (["Metals & Mining", "X"], ["metals and mining"]),
            (["South Africa", "Chile"], ["south africa", "Peru", "Chile"]), (["A", "a "], ["A"])]

    def test_normalize_identical_to_harness(self):
        for s in self.STRINGS:
            self.assertEqual(m.normalize(s), harness.normalize(s), msg=repr(s))

    def test_set_prf_identical_to_harness(self):
        for pred, gold in self.SETS:
            self.assertEqual(m.set_prf(pred, gold), harness.set_prf(pred, gold), msg=(pred, gold))

    def test_empty_against_empty_scores_one(self):
        self.assertEqual(m.set_prf([], []), (1.0, 1.0, 1.0))


class Matching(unittest.TestCase):
    def test_exact(self):
        a, b, c = frozenset({1, 2}), frozenset({3}), frozenset({4, 5})
        self.assertEqual(m.match_sets([a, b, b], [a, c]), (1, 1, 1))

    def test_jaccard_threshold_one_to_one(self):
        g1 = frozenset(range(10))
        p1 = frozenset(range(9))          # Jaccard 0.9 with g1
        p2 = frozenset(range(8))          # Jaccard 0.8 with g1
        self.assertEqual(m.match_sets([p1, p2], [g1], threshold=0.9), (1, 1, 0))
        self.assertEqual(m.match_sets([p2], [g1], threshold=0.9), (0, 1, 1))

    def test_path_node_set_is_order_insensitive_and_normalised(self):
        path = [{"company": "BMW AG", "country": "Germany", "industry": "Automotive"},
                {"company": "Kumba Iron Ore", "country": "South Africa", "industry": "Metals & Mining"}]
        self.assertEqual(m.path_node_set(path), m.path_node_set(list(reversed(path))))
        self.assertIn(("kumba iron ore", "south africa", "metals mining"), m.path_node_set(path))


class UnitsOfDecision(unittest.TestCase):
    def test_alert_rule(self):
        self.assertFalse(m.alert({t: [] for t in m.TIERS}))
        self.assertTrue(m.alert({"tier_1": [], "tier_2": [], "tier_3": [[{}]], "tier_4": []}))
        self.assertTrue(m.alert({"chain_counts": {"tier_1": 0, "tier_2": 0, "tier_3": 0, "tier_4": 2}}))
        self.assertFalse(m.alert(None))

    def test_ground_truth_v1_2_has_21_alerts_and_9_non_alerts(self):
        files = sorted((ROOT / "evaluation" / "ground_truth_final").glob("*_SC*.json"))
        self.assertEqual(len(files), 30)
        alerts = {f.stem: m.alert(json.loads(f.read_text())["kg_results"]) for f in files}
        self.assertEqual(sum(alerts.values()), 21)
        self.assertEqual(sorted(k for k, v in alerts.items() if not v),
                         sorted(f"{firm}_{t}" for firm in ("BMW", "MERC", "TSLA") for t in ("SC004", "SC008", "SC010")))

    def test_risk_mapping(self):
        cases = [(None, "NONE", "MAINTAIN_STANDARD"), (0.2992, "LOW", "MAINTAIN_STANDARD"),
                 (0.45, "MEDIUM", "INCREASE_MONITORING"), (0.5999, "MEDIUM", "INCREASE_MONITORING"),
                 (0.60, "HIGH", "REPLACE_SUPPLIER"), (0.5971, "MEDIUM", "INCREASE_MONITORING")]
        for score, level, action in cases:
            self.assertEqual((m.risk_level(score), m.expected_action(score)), (level, action), msg=score)


class EntityKeys(unittest.TestCase):
    def test_names_compared_after_resolution(self):
        self.assertEqual(m.entity_keys("countries", ["United States", "Usa"]), m.entity_keys("countries", ["USA"]))
        self.assertEqual(m.entity_keys("industries", ["mine"]), m.entity_keys("industries", ["Metals & Mining"]))
        self.assertEqual(m.entity_keys("companies", ["Telstra"]), m.entity_keys("companies", ["Telstra Group Ltd"]))

    def test_unresolved_names_kept_as_strings(self):
        self.assertEqual(m.entity_keys("companies", ["Optus", ""]), {"optus"})
        self.assertEqual(m.entity_keys("countries", ["Eastern Europe"]), {"eastern europe"})


class Baselines(unittest.TestCase):
    def test_random_baseline(self):
        c = m.random_baseline(120, 1000, positive_rate=0.3)
        self.assertAlmostEqual(c.mcc, 0.0, places=12)
        pi = 120 / 1120
        self.assertAlmostEqual(c.f1, 2 * pi * 0.3 / (pi + 0.3), places=12)


class Intervals(unittest.TestCase):
    CASES = {(0, 150): (0.0, 0.0242925972), (5, 150): (0.0109101865, 0.0760723225),
             (27, 30): (0.7347115495, 0.9788828630), (150, 150): (0.9757074028, 1.0)}  # statsmodels

    def test_clopper_pearson(self):
        for (k, n), (lo, hi) in self.CASES.items():
            got = m.clopper_pearson(k, n)
            self.assertAlmostEqual(got[0], lo, places=9, msg=(k, n))
            self.assertAlmostEqual(got[1], hi, places=9, msg=(k, n))

    def test_bootstrap_is_reproducible_and_stratified(self):
        units = [m.Confusion(1, 0, 0, 0)] * 27 + [m.Confusion(0, 0, 0, 1)] * 2 + [m.Confusion(0, 1, 0, 0)]
        strata = ["pos"] * 27 + ["neg"] * 3
        stat = lambda us: sum(us, m.Confusion()).mcc  # noqa: E731
        a = m.bootstrap(units, stat, strata=strata, n_boot=2000)
        b = m.bootstrap(units, stat, strata=strata, n_boot=2000)
        self.assertEqual(a, b)
        self.assertEqual(a["n_undefined"], 0)  # every resample keeps 27 positives and 3 negatives
        self.assertAlmostEqual(a["estimate"], m.Confusion(27, 1, 0, 2).mcc)
        self.assertLessEqual(a["lo"], a["estimate"])
        self.assertLessEqual(a["estimate"], a["hi"])
        unstratified = m.bootstrap(units, stat, n_boot=2000)
        self.assertGreater(unstratified["n_undefined"], 0)  # some resamples contain no negative scenario


class Prevalence(unittest.TestCase):
    def test_closed_form(self):
        for sens, spec, pi in [(0.95, 0.95, 0.5), (0.95, 0.95, 0.001), (0.8, 0.99, 0.01), (0.6, 0.7, 0.2)]:
            q = pi * sens + (1 - pi) * (1 - spec)
            closed = (sens + spec - 1) * math.sqrt(pi * (1 - pi)) / math.sqrt(q * (1 - q))
            self.assertAlmostEqual(m.mcc_at_prevalence(sens, spec, pi), closed, places=12)
        self.assertAlmostEqual(m.mcc_at_prevalence(0.95, 0.95, 0.5), 0.9, places=12)
        self.assertAlmostEqual(m.mcc_at_prevalence(0.5, 0.5, 0.01), 0.0, places=12)

    def test_mcc_falls_with_prevalence_for_fixed_sensitivity_and_specificity(self):
        values = [m.mcc_at_prevalence(0.95, 0.95, pi) for pi in (0.5, 0.1, 0.01, 0.001)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_expected_confusion_matches_counts(self):
        c = m.expected_confusion(0.9, 0.98, 0.1, n=1000)
        for got, want in zip((c.tp, c.fp, c.fn, c.tn), (90, 18, 10, 882)):
            self.assertAlmostEqual(got, want, places=9)

    def test_mixed_specificity(self):
        self.assertAlmostEqual(m.mixed_specificity({"N1": 0.9, "N2": 1.0}, {"N1": 1, "N2": 3}), 0.975)
        with self.assertRaises(ValueError):
            m.mixed_specificity({"N1": 0.9}, {"N1": 1, "N4": 1})

    def test_projection(self):
        rows = m.project(tp=114, n_pos=120, tn_by_type={"N1": 240, "N2": 250}, n_by_type={"N1": 250, "N2": 250},
                         weights={"N1": 1, "N2": 1}, prevalences=[0.01, 0.001], n_draws=2000)
        self.assertAlmostEqual(rows[0]["sensitivity"], 0.95)
        self.assertAlmostEqual(rows[0]["specificity"], 0.98)
        self.assertAlmostEqual(rows[0]["mcc"], m.mcc_at_prevalence(0.95, 0.98, 0.01), places=12)
        for r in rows:
            self.assertLessEqual(r["mcc_lo"], r["mcc"])
            self.assertLessEqual(r["mcc"], r["mcc_hi"])
        self.assertGreater(rows[0]["mcc"], rows[1]["mcc"])

    def test_subsampling(self):
        positives = [m.Confusion(1, 0, 0, 0)] * 100 + [m.Confusion(0, 0, 1, 0)] * 20
        negatives = [m.Confusion(0, 0, 0, 1)] * 990 + [m.Confusion(0, 1, 0, 0)] * 10
        stat = lambda pos, neg: sum(list(pos) + list(neg), m.Confusion()).mcc  # noqa: E731
        r = m.subsample_at_prevalence(positives, negatives, 0.01, stat, n_rep=200)
        self.assertEqual((r["n_pos"], r["n_neg"]), (10, 1000))
        self.assertLessEqual(r["lo"], r["median"])
        self.assertLessEqual(r["median"], r["hi"])
        with self.assertRaises(ValueError):
            m.subsample_at_prevalence(positives, negatives, 0.5, stat)


if __name__ == "__main__":
    unittest.main()
