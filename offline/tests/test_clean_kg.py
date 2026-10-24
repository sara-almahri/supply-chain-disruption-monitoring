"""Tests that dataset/supplychainKG_clean.csv is the output of scripts/clean_kg.py and satisfies its rules.

    python -m unittest discover -s offline/tests -t .
"""

import csv
import unittest
from collections import defaultdict

from offline import kg, names
from scripts import clean_kg


class CleanNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(kg.CSV_PATH, encoding="utf-8", newline="") as f:
            cls.rows = list(csv.DictReader(f))
        cls.net = kg.load()

    def test_file_is_reproduced(self):
        cleaned, _ = clean_kg.clean(clean_kg.read_source())
        self.assertEqual([tuple(r[k] for k in clean_kg.COLUMNS) for r in cleaned],
                         [tuple(r[k] for k in clean_kg.COLUMNS) for r in self.rows])

    def test_one_industry_and_one_country_per_company(self):
        industry, country = defaultdict(set), defaultdict(set)
        for r in self.rows:
            for name_k, ind_k, cty_k in clean_kg.SIDES:
                industry[r[name_k]].add(r[ind_k])
                country[r[name_k]].add(r[cty_k])
        self.assertTrue(all(len(v) == 1 for v in industry.values()))
        self.assertTrue(all(len(v) == 1 for v in country.values()))

    def test_links_unique_and_no_self_links(self):
        pairs = [(r["Supplier"], r["Customer"]) for r in self.rows]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertFalse(any(s == c for s, c in pairs))

    def test_names_and_vocabularies(self):
        with open(clean_kg.MERGES, encoding="utf-8", newline="") as f:
            merges = list(csv.DictReader(f))
        companies = set(self.net.industry)
        self.assertFalse({m["variant"] for m in merges} & companies)
        self.assertLessEqual({m["canonical"] for m in merges}, companies)
        keys = [names._spelling_key(c) for c in companies]
        self.assertEqual(len(keys), len(set(keys)))
        countries = {c for cs in self.net.countries.values() for c in cs}
        variants = {s for v in names.COUNTRY_SPELLINGS.values() for s in v} - set(names.COUNTRY_SPELLINGS)
        self.assertFalse(countries & (variants | names.NOT_COUNTRIES))
        self.assertNotIn("Mine", set(self.net.industry.values()))


if __name__ == "__main__":
    unittest.main()
