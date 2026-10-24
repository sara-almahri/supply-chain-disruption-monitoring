"""Tests for offline/names.py and the disruption rule (offline.kg.is_disrupted, offline.kg.disrupted_chains).

    python -m unittest discover -s offline/tests -t .
"""

import unittest

from offline import kg, names


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = kg.load()
        cls.resolver = names.Resolver(cls.net)


class Countries(Base):
    def test_aliases(self):
        cases = {
            "USA": "USA", "Usa": "USA", "United States": "USA", "u.s.": "USA", "UK": "United Kingdom",
            "Britain": "United Kingdom", "uae": "United Arab Emirates", "Philipines": "Philippines",
            "Czechia": "Czech Republic", "Libya": "Libyan Arab Jamahiriya", "Türkiye": "Turkey",
            "Russian Federation": "Russia", "Viet Nam": "Vietnam", "japan": "Japan",
            "the United States": "USA", "The  United Kingdom": "United Kingdom", "England": "United Kingdom",
            "Scotland": "United Kingdom", "the Netherlands": "Netherlands", "the Philippines": "Philippines",
        }
        for raw, canonical in cases.items():
            self.assertEqual(self.resolver.country(raw), canonical, raw)

    def test_unknown_unresolved(self):
        for raw in ("Steel", "Eastern Europe", "North Korea", "Atlantis"):
            self.assertIsNone(self.resolver.country(raw), raw)

    def test_spellings_matched_together(self):
        match = self.resolver.disruption(countries=["United States"])["match"]
        self.assertEqual(set(match["countries"]), self.resolver.countries["USA"])

    def test_label_canonical(self):
        self.assertEqual(names.country_label(["Usa", "USA"], {"USA", "Usa"}), "USA")
        self.assertEqual(names.country_label(["Germany", "Usa"], set()), "Germany")
        self.assertEqual(names.country_label([], set()), "Unknown")


class Industries(Base):
    def test_resolution(self):
        self.assertEqual(self.resolver.industry("semiconductors"), "Semiconductors")
        self.assertEqual(self.resolver.industry("Mine"), "Metals & Mining")
        self.assertIsNone(self.resolver.industry("Aerospace"))

    def test_metals_and_mining_includes_mine(self):
        match = self.resolver.disruption(countries=["Chile"], industries=["Mine"], scope="industries")["match"]
        self.assertEqual(set(match["industries"]), self.resolver.industries["Metals & Mining"])

    def test_country_scope_ignores_industries(self):
        match = self.resolver.disruption(countries=["Chile"], industries=["Steel"], scope="country")["match"]
        self.assertIsNone(match["industries"])


class Companies(Base):
    def test_resolution(self):
        cases = {
            "Tesla": {"Tesla Inc"}, "POSCO Holdings": {"POSCO Holdings Inc"}, "POSCO": {"POSCO"},
            "BHP": {"BHP Group Ltd"}, "Glencore": {"Glencore PLC"},
            "Mercedes-Benz": {"Mercedes-Benz Group AG"}, "Mercedes Benz Group AG": {"Mercedes-Benz Group AG"},
            "komatsu": {"Komatsu Ltd"},
        }
        for raw, expected in cases.items():
            self.assertEqual(self.resolver.company(raw), expected, raw)

    def test_ambiguous_or_unknown_unresolved(self):
        for raw in ("APA", "BYD", "Steyr Motors", "Ltd", "TSMC"):
            self.assertIsNone(self.resolver.company(raw), raw)

    def test_every_graph_name_resolves_to_itself(self):
        self.assertTrue(all(c in (self.resolver.company(c) or ()) for c in self.net.industry))

    def test_unresolved_names_reported(self):
        out = self.resolver.disruption(countries=["Japan", "Eastern Europe"], industries=["Aerospace"],
                                       companies=["APA", "Tesla"], scope="industries")
        self.assertEqual(out["unresolved"],
                         {"countries": ["Eastern Europe"], "industries": ["Aerospace"], "companies": ["APA"]})
        self.assertEqual(out["countries"], ["Japan"])
        self.assertEqual(out["companies"], ["Tesla Inc"])


class SourceSpellings(unittest.TestCase):
    """The source file spells some countries, industries and companies in several ways; all are matched together."""

    @classmethod
    def setUpClass(cls):
        cls.resolver = names.Resolver(kg.load(kg.SOURCE_CSV_PATH))

    def test_matched_together(self):
        out = self.resolver.disruption(countries=["United States"], industries=["Metals & Mining"],
                                       companies=["Mercedes-Benz"], scope="industries")["match"]
        self.assertEqual(set(out["countries"]), {"USA", "Usa"})
        self.assertEqual(set(out["industries"]), {"Metals & Mining", "Mine"})
        self.assertEqual(set(out["companies"]), {"Mercedes-Benz Group AG", "Mercedes Benz Group AG"})


class Rule(Base):
    def test_is_disrupted_scope(self):
        net = self.net
        company = next(c for c, i in net.industry.items() if i == "Semiconductors" and "Taiwan" in net.located_in(c))
        self.assertTrue(kg.is_disrupted(net, company, {"Taiwan"}, set()))
        self.assertTrue(kg.is_disrupted(net, company, {"Taiwan"}, set(), {"Semiconductors"}))
        self.assertFalse(kg.is_disrupted(net, company, {"Taiwan"}, set(), {"Steel"}))
        self.assertTrue(kg.is_disrupted(net, company, set(), {company}, {"Steel"}))

    def test_industry_scope_selects_subset(self):
        firm = "Bayerische Motoren Werke AG"
        wide = self.resolver.disruption(countries=["Japan"])["match"]
        narrow = self.resolver.disruption(countries=["Japan"], industries=["Automotive"], scope="industries")["match"]
        a = kg.disrupted_chains(self.net, firm, wide["countries"], wide["companies"], mode="simple")
        b = kg.disrupted_chains(self.net, firm, narrow["countries"], narrow["companies"], mode="simple",
                                industries=narrow["industries"])
        for t in range(1, kg.MAX_TIER + 1):
            for chain in b[f"tier_{t}"]:
                self.assertEqual((chain[-1]["industry"], chain[-1]["country"]), ("Automotive", "Japan"))
            self.assertLessEqual(len(b[f"tier_{t}"]), len(a[f"tier_{t}"]))


if __name__ == "__main__":
    unittest.main()
