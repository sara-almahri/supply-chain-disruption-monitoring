#!/usr/bin/env python3
"""Data-quality audit of dataset/supplychainKG.csv, the supply network of Chapters 4 and 5.

    python -m scripts.audit_kg [csv_path]

Prints a summary and writes dataset/audit/<csv stem>_audit.json with every finding.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from offline.names import COUNTRY_SPELLINGS, GROUP_WORDS, LEGAL_FORMS, _spelling_key  # noqa: E402

REQUIRED = ("Supplier", "Supplier Industry", "Supplier Country", "Customer", "Customer Industry", "Customer Country")
FIRMS = ("Bayerische Motoren Werke AG", "Mercedes-Benz Group AG", "Tesla Inc")


def audit(csv_path: Path) -> dict:
    raw = csv_path.read_bytes()
    findings: dict = {"file": str(csv_path.relative_to(ROOT)), "bytes": len(raw)}
    try:
        raw.decode("utf-8")
        findings["encoding"] = "valid UTF-8"
    except UnicodeDecodeError as exc:
        findings["encoding"] = f"invalid UTF-8 at byte {exc.start}"
    text = raw.decode("utf-8", errors="replace")
    findings["non_ascii_characters"] = sorted({ch for ch in text if ord(ch) > 127})

    rows, skipped, padded = [], [], []
    extra_columns = Counter()
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        findings["columns"] = reader.fieldnames
        for line, row in enumerate(reader, start=2):
            for k in reader.fieldnames[6:]:
                extra_columns[(k, (row.get(k) or "").strip())] += 1
            if any(not (row.get(k) or "").strip() for k in REQUIRED):
                skipped.append({"line": line, "row": {k: row.get(k) for k in REQUIRED}})
                continue
            if any(row[k] != row[k].strip() or "  " in row[k] for k in REQUIRED):
                padded.append(line)
            rows.append((line, tuple(" ".join(row[k].split()) for k in REQUIRED)))
    findings["rows"] = len(rows) + len(skipped)
    findings["rows_with_empty_required_field"] = skipped
    findings["rows_with_extra_whitespace"] = len(padded)
    findings["other_column_values"] = {f"{k}={v}": n for (k, v), n in sorted(extra_columns.items())}

    full = Counter(r for _, r in rows)
    findings["exact_duplicate_rows"] = sum(n - 1 for n in full.values() if n > 1)

    industries, countries, pair_rows = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    for line, (s, si, sc, c, ci, cc) in rows:
        industries[s][si] += 1
        industries[c][ci] += 1
        countries[s][sc] += 1
        countries[c][cc] += 1
        pair_rows[(s, c)].append(line)
    companies = set(industries)
    findings["companies"] = len(companies)
    findings["links"] = len(pair_rows)
    findings["self_links"] = sorted(f"{s}" for (s, c) in pair_rows if s == c)
    findings["two_way_links"] = sum(1 for (s, c) in pair_rows if s < c and (c, s) in pair_rows)

    groups = defaultdict(set)
    for name in companies:
        groups[_spelling_key(name)].add(name)
    findings["company_spelling_variants"] = sorted(sorted(v) for v in groups.values() if len(v) > 1)
    words = LEGAL_FORMS | GROUP_WORDS
    findings["repeated_legal_form"] = sorted(
        n for n in companies if re.search(r"\b(\w+)\s+\1\s*$", n, re.IGNORECASE) and n.split()[-1].lower() in words)
    findings["names_with_unusual_characters"] = sorted(n for n in companies if re.search(r"[^\w\s&.,'()/+-]", n)
                                                       or n.rstrip()[-1:] in ":;,")
    findings["names_of_legal_forms_only"] = sorted(n for n in companies
                                                   if all(t.lower().strip(".,") in words for t in n.split()))

    industry_vocab = Counter(i for cnt in industries.values() for i, n in cnt.items() for _ in range(n))
    findings["industries"] = dict(industry_vocab.most_common())
    findings["companies_with_several_industries"] = {
        n: dict(cnt) for n, cnt in sorted(industries.items()) if len(cnt) > 1}

    country_vocab = Counter(c for cnt in countries.values() for c, n in cnt.items() for _ in range(n))
    findings["countries"] = dict(sorted(country_vocab.items()))
    lower = defaultdict(set)
    for c in country_vocab:
        lower[c.lower()].add(c)
    known = {s for v in COUNTRY_SPELLINGS.values() for s in v}
    findings["country_spelling_variants"] = sorted(
        sorted(v) for v in list(lower.values()) + [sorted(s) for s in COUNTRY_SPELLINGS.values()]
        if len(v) > 1 and set(v) <= set(country_vocab))
    findings["country_values_to_check"] = sorted(c for c in country_vocab if c in ("Steel",) or c not in known and
                                                 any(ch.isdigit() for ch in c))
    multi = {n: dict(cnt) for n, cnt in sorted(countries.items()) if len(cnt) > 1}
    findings["companies_in_several_countries"] = len(multi)
    findings["companies_in_several_countries_examples"] = dict(list(multi.items())[:40])

    suppliers = defaultdict(set)
    for (s, c) in pair_rows:
        suppliers[c].add(s)
    findings["firms"] = {f: {"in_graph": f in companies, "tier1_suppliers": len(suppliers.get(f, ()))} for f in FIRMS}
    return findings


def main():
    csv_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "dataset" / "supplychainKG.csv"
    findings = audit(csv_path)
    out = ROOT / "dataset" / "audit" / f"{csv_path.stem}_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=1, ensure_ascii=False))
    for key, value in findings.items():
        if isinstance(value, (list, dict)) and len(value) > 12:
            print(f"{key}: {len(value)} entries")
        else:
            print(f"{key}: {value}")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
