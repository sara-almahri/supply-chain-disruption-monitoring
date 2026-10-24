#!/usr/bin/env python3
"""Builds dataset/supplychainKG_clean.csv, the supply network of Chapters 4 and 5, from the source file
dataset/supplychainKG.csv.

    python -m scripts.clean_kg

The source file is left unchanged. The rules, applied in this order:

1. Company names. Each name listed in dataset/cleaning/company_merges.csv is replaced by its canonical name; the table
   was reviewed pair by pair and holds transcription errors only (misread or stray characters, garbled legal forms).
   Names that differ only in case, punctuation, spacing or "&"/"and" are then merged into their most frequent
   spelling. Names that differ in legal form or in a group word (POSCO / POSCO Holdings Inc, SoftBank Corp / SoftBank
   Group Corp) denote different legal entities and are kept apart.
2. Countries take their canonical spelling (USA for Usa, United Kingdom for Uk). A value that is not a country is
   discarded and the company's country is taken from its other rows.
3. Industries: "Mine" (the 30 mine and refinery sites) is Metals & Mining.
4. Each company has one industry and one country: the value on most of the rows in which the company appears, the
   earliest row deciding a tie. dataset/cleaning/company_attributes.csv lists the reviewed exceptions: a company whose
   recorded values disagree and whose home country, on the public record, is a less frequent recorded value, and a
   company whose source rows record no country.
5. Links: a link from a company to itself is dropped; a supplier-customer pair is kept once.

The 30 rows with Relationship1 "owns" link a mine or refinery to the company that owns it. They are kept as supply
links: the site's output goes to its owner, and the site's country is where that supply is exposed.

Writes the cleaned file and dataset/cleaning/cleaning_report.json.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from offline.names import COUNTRY_SPELLINGS, INDUSTRY_SPELLINGS, NOT_COUNTRIES, _spelling_key  # noqa: E402

SOURCE = ROOT / "dataset" / "supplychainKG.csv"
CLEAN = ROOT / "dataset" / "supplychainKG_clean.csv"
CLEANING = ROOT / "dataset" / "cleaning"
MERGES = CLEANING / "company_merges.csv"
ATTRIBUTES = CLEANING / "company_attributes.csv"
REPORT = CLEANING / "cleaning_report.json"

COLUMNS = ("Supplier", "Supplier Industry", "Supplier Country", "Customer", "Customer Industry", "Customer Country",
           "Relationship1", "Relationship2", "Relationship3")
SIDES = (("Supplier", "Supplier Industry", "Supplier Country"), ("Customer", "Customer Industry", "Customer Country"))

COUNTRY = {s: c for c, spellings in COUNTRY_SPELLINGS.items() for s in spellings}
INDUSTRY = {s: c for c, spellings in INDUSTRY_SPELLINGS.items() for s in spellings}


def read_source(path: Path = SOURCE) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = [{k: " ".join((row.get(k) or "").split()) for k in COLUMNS} for row in csv.DictReader(f)]
    return [r for r in rows if all(r[k] for side in SIDES for k in side)]


def read_merges(names: set[str]) -> dict[str, tuple[str, str]]:
    with open(MERGES, encoding="utf-8", newline="") as f:
        table = {r["variant"]: (r["canonical"], r["reason"]) for r in csv.DictReader(f)}
    missing = sorted(v for v in table if v not in names)
    chained = sorted(v for v, (c, _) in table.items() if c in table)
    if missing or chained:
        raise SystemExit(f"company_merges.csv: variants not in the source {missing}; canonical names that are "
                         f"themselves variants {chained}")
    return table


def read_attributes() -> dict[tuple[str, str], tuple[str, str]]:
    if not ATTRIBUTES.exists():
        return {}
    with open(ATTRIBUTES, encoding="utf-8", newline="") as f:
        return {(r["company"], r["field"]): (r["value"], r["reason"]) for r in csv.DictReader(f)}


def name_map(rows: list[dict]) -> dict[str, tuple[str, str]]:
    """Source name -> (cleaned name, rule) for every name that changes."""
    count = Counter(r[side[0]] for r in rows for side in SIDES)
    table = read_merges(set(count))
    mapping = {v: (c, reason) for v, (c, reason) in table.items()}
    renamed = Counter()
    for name, n in count.items():
        renamed[mapping.get(name, (name,))[0]] += n
    groups = defaultdict(list)
    for name in renamed:
        groups[_spelling_key(name)].append(name)
    for spellings in groups.values():
        if len(spellings) < 2:
            continue
        keep = min(spellings, key=lambda s: (-renamed[s], s))
        for s in spellings:
            if s != keep:
                mapping[s] = (keep, "spelling variant")
    for v, (c, reason) in list(mapping.items()):
        while c in mapping and mapping[c][0] != c:
            c = mapping[c][0]
        mapping[v] = (c, reason)
    return mapping


def modal(counter: Counter, first_seen: dict) -> tuple[str, bool]:
    top = max(counter.values())
    tied = [v for v, n in counter.items() if n == top]
    return min(tied, key=lambda v: first_seen[v]), len(tied) > 1


def clean(rows: list[dict]) -> tuple[list[dict], dict]:
    names = name_map(rows)
    attributes = read_attributes()
    report: dict = {"source": str(SOURCE.relative_to(ROOT)), "source_rows": len(rows)}

    industries, countries = defaultdict(Counter), defaultdict(Counter)
    first_industry, first_country = defaultdict(dict), defaultdict(dict)
    discarded_countries = Counter()
    staged = []
    for i, r in enumerate(rows):
        out = dict(r)
        for name_k, ind_k, cty_k in SIDES:
            company = names.get(r[name_k], (r[name_k],))[0]
            industry = INDUSTRY.get(r[ind_k], r[ind_k])
            country = COUNTRY.get(r[cty_k], r[cty_k])
            out[name_k], out[ind_k] = company, industry
            industries[company][industry] += 1
            first_industry[company].setdefault(industry, i)
            if country in NOT_COUNTRIES:
                discarded_countries[(r[name_k], country)] += 1
            else:
                countries[company][country] += 1
                first_country[company].setdefault(country, i)
        staged.append(out)

    industry_of, country_of, ties, changed_industry, changed_country = {}, {}, [], {}, {}
    for company in industries:
        for fld, counter, first, target, changed in (
                ("industry", industries[company], first_industry[company], industry_of, changed_industry),
                ("country", countries[company], first_country[company], country_of, changed_country)):
            if (company, fld) in attributes:
                value, tied = attributes[(company, fld)][0], False
                if counter and value not in counter:
                    raise SystemExit(f"company_attributes.csv: {company} {fld} {value} is not among {dict(counter)}")
            elif not counter:
                raise SystemExit(f"{company}: no valid {fld}; add it to company_attributes.csv")
            else:
                value, tied = modal(counter, first)
            if tied:
                ties.append({"company": company, "field": fld, "values": dict(counter), "chosen": value})
            target[company] = value
            if len(counter) > 1:
                changed[company] = {"recorded": dict(counter), "assigned": value}

    cleaned, seen, self_links, duplicates = [], set(), [], 0
    for out in staged:
        s, c = out["Supplier"], out["Customer"]
        if s == c:
            self_links.append(s)
            continue
        if (s, c) in seen:
            duplicates += 1
            continue
        seen.add((s, c))
        for name_k, ind_k, cty_k in SIDES:
            out[ind_k], out[cty_k] = industry_of[out[name_k]], country_of[out[name_k]]
        cleaned.append(out)

    rows_per_name = Counter(r[side[0]] for r in rows for side in SIDES)
    report.update({
        "rules": {
            "company_names_merged": len(names),
            "company_names_merged_by_rule": dict(Counter(rule for _, rule in names.values())),
            "country_values_discarded": [{"company": n, "value": v, "rows": k}
                                         for (n, v), k in sorted(discarded_countries.items())],
            "companies_with_several_industries": len(changed_industry),
            "companies_with_several_countries": len(changed_country),
            "ties_resolved_by_first_row": ties,
            "ties_resolved_by_review": [{"company": c, "field": f, "value": v, "reason": why}
                                        for (c, f), (v, why) in sorted(attributes.items())],
            "self_links_dropped": sorted(self_links),
            "duplicate_links_dropped": duplicates,
        },
        "clean_rows": len(cleaned),
        "companies_before": len(rows_per_name),
        "companies_after": len(industry_of),
        "countries_after": len(set(country_of.values())),
        "industries_after": dict(Counter(industry_of.values()).most_common()),
        "name_changes": [{"source": v, "clean": c, "rule": rule, "rows": rows_per_name[v]}
                         for v, (c, rule) in sorted(names.items())],
        "industry_assignments": changed_industry,
        "country_assignments": changed_country,
    })
    return cleaned, report


def main():
    cleaned, report = clean(read_source())
    with open(CLEAN, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(cleaned)
    REPORT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for key in ("source_rows", "clean_rows", "companies_before", "companies_after", "countries_after"):
        print(f"{key}: {report[key]}")
    for key, value in report["rules"].items():
        print(f"{key}: {len(value) if isinstance(value, list) else value}")
    print("wrote", CLEAN.relative_to(ROOT), "and", REPORT.relative_to(ROOT))


if __name__ == "__main__":
    main()
