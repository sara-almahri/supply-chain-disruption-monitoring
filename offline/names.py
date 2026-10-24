"""
Resolution of extracted country, industry and company names to the names used in the knowledge graph.

Resolution is deterministic and never guesses:
    countries   the graph's own spellings and a fixed table of common alternative names, compared without case;
    industries  the graph's industry names, compared without case; 'Mine' is the same sector as 'Metals & Mining';
    companies   names that differ only in case, punctuation or spacing are the same company; otherwise a name
                matches a company whose name is the same after removing trailing legal forms (Ltd, Inc, AG, ...)
                and the words 'Group', 'Holding(s)' and 'Company', when every such word in the name also appears in
                the company's name and exactly one company matches. Any other name is unresolved and reported.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set

# Canonical country -> every spelling used for it in the source file (the cleaned file uses the canonical name).
COUNTRY_SPELLINGS: Dict[str, Set[str]] = {
    "USA": {"USA", "Usa"},
    "United Kingdom": {"United Kingdom", "Uk"},
    "United Arab Emirates": {"United Arab Emirates", "Uae"},
    "Philippines": {"Philippines", "Philipines"},
}
# Alternative names (lower case) -> canonical country.
COUNTRY_ALIASES: Dict[str, str] = {
    "united states": "USA", "united states of america": "USA", "us": "USA", "u.s.": "USA", "u.s.a.": "USA",
    "america": "USA",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "britain": "United Kingdom", "great britain": "United Kingdom",
    "england": "United Kingdom", "scotland": "United Kingdom", "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "uae": "United Arab Emirates", "u.a.e.": "United Arab Emirates",
    "czechia": "Czech Republic",
    "libya": "Libyan Arab Jamahiriya",
    "dr congo": "Democratic Republic Of The Congo", "drc": "Democratic Republic Of The Congo",
    "democratic republic of congo": "Democratic Republic Of The Congo", "congo-kinshasa": "Democratic Republic Of The Congo",
    "british virgin islands": "Virgin Islands, British",
    "turkiye": "Turkey", "türkiye": "Turkey",
    "republic of korea": "South Korea", "korea, republic of": "South Korea",
    "russian federation": "Russia",
    "viet nam": "Vietnam",
    "holland": "Netherlands", "the netherlands": "Netherlands",
    "hong kong sar": "Hong Kong",
}
# Values in the source file's country field that are not countries.
NOT_COUNTRIES = {"Steel"}

INDUSTRY_SPELLINGS: Dict[str, Set[str]] = {"Metals & Mining": {"Metals & Mining", "Mine"}}

_CANONICAL_SPELLING = {s: c for c, spellings in COUNTRY_SPELLINGS.items() for s in spellings}


def country_label(countries, disrupted: Set[str]) -> str:
    """The country reported for a company: a disrupted country where it is located, otherwise the first of its
    countries; in both cases under its canonical name."""
    if not countries:
        return "Unknown"
    ordered = sorted(countries)
    chosen = next((c for c in ordered if c in disrupted), ordered[0])
    return _CANONICAL_SPELLING.get(chosen, chosen)

LEGAL_FORMS = {
    "ab", "ag", "as", "asa", "bhd", "bv", "co", "corp", "corporation", "cv", "de", "gmbh", "inc", "incorporated",
    "jsc", "kk", "limited", "llc", "lp", "ltd", "ltda", "nv", "oao", "oyj", "pjsc", "plc", "pt", "pte", "sa", "sab",
    "se", "spa", "tbk",
}
GROUP_WORDS = {"company", "group", "holding", "holdings"}


def _spelling_key(name: str) -> str:
    """Same company when only case, punctuation, spacing or '&'/'and' differ."""
    text = str(name).lower().replace("&", " and ")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _group_words(key: str) -> Set[str]:
    core = _core_key(key, LEGAL_FORMS | GROUP_WORDS).split()
    return set(key.split()[len(core):]) & GROUP_WORDS


def _core_key(name: str, words: Set[str] = LEGAL_FORMS) -> str:
    tokens = _spelling_key(name).split()
    while tokens and tokens[-1] in words:
        tokens.pop()
    while tokens and tokens[0] in ("the", "pt"):
        tokens.pop(0)
    return " ".join(tokens)


class Resolver:
    """Name resolution against one network (offline.kg.Network)."""

    def __init__(self, net):
        graph_countries = {c for cs in net.countries.values() for c in cs} - NOT_COUNTRIES
        self.countries: Dict[str, Set[str]] = {}
        for country in graph_countries:
            canonical = next((k for k, v in COUNTRY_SPELLINGS.items() if country in v), country)
            self.countries.setdefault(canonical, set()).add(country)
        self._country_lookup: Dict[str, str] = {c.lower(): c for c in self.countries}
        for canonical, spellings in COUNTRY_SPELLINGS.items():
            if canonical in self.countries:
                for s in spellings:
                    self._country_lookup.setdefault(s.lower(), canonical)
        for alias, canonical in COUNTRY_ALIASES.items():
            if canonical in self.countries:
                self._country_lookup.setdefault(alias, canonical)

        graph_industries = set(net.industry.values())
        self.industries: Dict[str, Set[str]] = {}
        for industry in graph_industries:
            canonical = next((k for k, v in INDUSTRY_SPELLINGS.items() if industry in v), industry)
            self.industries.setdefault(canonical, set()).add(industry)
        self._industry_lookup: Dict[str, str] = {}
        for canonical, spellings in self.industries.items():
            for s in spellings | {canonical} | INDUSTRY_SPELLINGS.get(canonical, set()):
                self._industry_lookup[s.lower()] = canonical

        self.company_names: Set[str] = set(net.industry)
        self._by_spelling: Dict[str, Set[str]] = defaultdict(set)
        self._by_core: Dict[str, Set[str]] = defaultdict(set)
        for company in net.industry:
            self._by_spelling[_spelling_key(company)].add(company)
        for key in self._by_spelling:
            core = _core_key(key, LEGAL_FORMS | GROUP_WORDS)
            if core:
                self._by_core[core].add(key)

    # -- single names --------------------------------------------------------------------------------------------
    def country(self, name: str) -> Optional[str]:
        key = " ".join(str(name).strip().lower().split())
        return self._country_lookup.get(key) or (self._country_lookup.get(key[4:]) if key.startswith("the ") else None)

    def industry(self, name: str) -> Optional[str]:
        return self._industry_lookup.get(str(name).strip().lower())

    def company(self, name: str) -> Optional[Set[str]]:
        """Every graph spelling of the company, or None when unresolved or ambiguous."""
        key = _spelling_key(name)
        if not key:
            return None
        if key in self._by_spelling:
            return set(self._by_spelling[key])
        wanted = _group_words(key)
        keys = {k for k in self._by_core.get(_core_key(key, LEGAL_FORMS | GROUP_WORDS), set())
                if wanted <= _group_words(k)}
        if len(keys) == 1:
            return set(self._by_spelling[next(iter(keys))])
        return None

    # -- a disruption --------------------------------------------------------------------------------------------
    def disruption(self, countries: Iterable[str] = (), industries: Iterable[str] = (),
                   companies: Iterable[str] = (), scope: str = "country") -> Dict[str, object]:
        """Resolved names, the graph spellings to match, and the names that could not be resolved."""
        out = {"countries": [], "industries": [], "companies": [],
               "unresolved": {"countries": [], "industries": [], "companies": []},
               "scope": scope if scope in ("country", "industries") else "country"}
        match_countries: Set[str] = set()
        match_industries: Set[str] = set()
        match_companies: Set[str] = set()
        for raw in _clean(countries):
            c = self.country(raw)
            if c:
                out["countries"].append(c)
                match_countries |= self.countries[c]
            else:
                out["unresolved"]["countries"].append(raw)
        for raw in _clean(industries):
            i = self.industry(raw)
            if i:
                out["industries"].append(i)
                match_industries |= self.industries[i]
            else:
                out["unresolved"]["industries"].append(raw)
        for raw in _clean(companies):
            names = self.company(raw)
            if names:
                out["companies"].append(sorted(names)[0])
                match_companies |= names
            else:
                out["unresolved"]["companies"].append(raw)
        for key in ("countries", "industries", "companies"):
            out[key] = sorted(set(out[key]))
        out["match"] = {
            "countries": sorted(match_countries),
            "industries": None if out["scope"] == "country" else sorted(match_industries),
            "companies": sorted(match_companies),
        }
        return out


def _clean(values: Iterable[str]) -> List[str]:
    return [str(v).strip() for v in (values or []) if str(v).strip()]


@lru_cache(maxsize=1)
def default_resolver() -> Resolver:
    from offline import kg
    return Resolver(kg.load())
