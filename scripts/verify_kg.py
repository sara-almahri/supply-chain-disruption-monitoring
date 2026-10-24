"""
Verify that the Neo4j knowledge graph is identical to dataset/supplychainKG_clean.csv.

Checks node and relationship counts, compares the full suppliesTo and
locatedIn edge sets and the company industry attribute with the CSV, and
confirms that every company has an embedding and the vector index is online.
Writes a JSON record to logs/.

Usage:
    python scripts/verify_kg.py
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "dataset" / "supplychainKG_clean.csv"
FIELDS = ["Supplier", "Supplier Industry", "Supplier Country",
          "Customer", "Customer Industry", "Customer Country"]


def csv_graph():
    companies, countries, supplies, located, industry = set(), set(), set(), set(), {}
    with open(CSV_FILE, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            if any(not (row.get(k) or "").strip() for k in FIELDS):
                continue
            s, si, sc, c, ci, cc = (row[k].strip() for k in FIELDS)
            companies |= {s, c}
            countries |= {sc, cc}
            supplies.add((s, c))
            located |= {(s, sc), (c, cc)}
            industry.setdefault(s, si)
            industry.setdefault(c, ci)
    return companies, countries, supplies, located, industry


def neo4j_graph(session):
    q = lambda cypher: session.run(cypher).data()
    companies = {r["n"] for r in q("MATCH (c:Company) RETURN c.name AS n")}
    countries = {r["n"] for r in q("MATCH (c:Country) RETURN c.name AS n")}
    supplies = {(r["s"], r["c"]) for r in
                q("MATCH (s:Company)-[:suppliesTo]->(c:Company) RETURN s.name AS s, c.name AS c")}
    located = {(r["s"], r["c"]) for r in
               q("MATCH (s:Company)-[:locatedIn]->(c:Country) RETURN s.name AS s, c.name AS c")}
    industry = {r["n"]: r["i"] for r in q("MATCH (c:Company) RETURN c.name AS n, c.industry AS i")}
    counts = q("MATCH (n) WITH count(n) AS nodes "
               "MATCH ()-[r]->() RETURN nodes, count(r) AS relationships")[0]
    embedded = q("MATCH (c:Company) WHERE c.embedding IS NOT NULL RETURN count(c) AS n")[0]["n"]
    index = q("SHOW INDEXES YIELD name, type, state WHERE name = 'company_embedding_index' "
              "RETURN type, state")
    return companies, countries, supplies, located, industry, counts, embedded, index


def main():
    load_dotenv(ROOT / ".env")
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                  auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]))
    with driver.session(database=os.environ.get("NEO4J_DATABASE") or None) as session:
        kg = neo4j_graph(session)
    driver.close()

    ref = csv_graph()
    checks = {
        "companies_equal": ref[0] == kg[0],
        "countries_equal": ref[1] == kg[1],
        "suppliesTo_equal": ref[2] == kg[2],
        "locatedIn_equal": ref[3] == kg[3],
        "industry_equal": ref[4] == kg[4],
        "all_companies_embedded": kg[6] == len(kg[0]),
        "vector_index_online": bool(kg[7]) and kg[7][0]["state"] == "ONLINE",
    }
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "neo4j_uri": os.environ["NEO4J_URI"],
        "csv": {"companies": len(ref[0]), "countries": len(ref[1]),
                "suppliesTo": len(ref[2]), "locatedIn": len(ref[3]),
                "nodes": len(ref[0]) + len(ref[1]),
                "relationships": len(ref[2]) + len(ref[3])},
        "neo4j": {"companies": len(kg[0]), "countries": len(kg[1]),
                  "suppliesTo": len(kg[2]), "locatedIn": len(kg[3]),
                  "nodes": kg[5]["nodes"], "relationships": kg[5]["relationships"],
                  "companies_with_embedding": kg[6], "vector_index": kg[7]},
        "checks": checks,
        "passed": all(checks.values()),
    }
    out = ROOT / "logs" / f"kg_verification_{CSV_FILE.stem}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    sys.exit(0 if record["passed"] else 1)


if __name__ == "__main__":
    main()
