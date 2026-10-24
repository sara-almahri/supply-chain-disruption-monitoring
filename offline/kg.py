"""
The supply network with the ingestion rules of scripts/kg_ingestion.py, and the disrupted supply chains of each tier.

CSV_PATH is the cleaned network, dataset/supplychainKG_clean.csv (scripts/clean_kg.py), on which every result of
Chapters 4 and 5 is computed. SOURCE_CSV_PATH is the network before cleaning, dataset/supplychainKG.csv.

Ingestion rules: a row with an empty required field is skipped; names are stripped; a company keeps the industry of the
first row in which it appears and is located in every country it appears with; one supplier-customer pair is one link.
In the cleaned file each company has one industry and one country and each link appears once.

A path is an upstream chain from the firm, (firm, Tier-1 supplier, ..., Tier-k supplier), that contains no company
twice ("simple"), as the path tool of the pipeline returns.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

CSV_PATH = Path(__file__).resolve().parent.parent / "dataset" / "supplychainKG_clean.csv"
SOURCE_CSV_PATH = Path(__file__).resolve().parent.parent / "dataset" / "supplychainKG.csv"
REQUIRED = ("Supplier", "Supplier Industry", "Supplier Country", "Customer", "Customer Industry", "Customer Country")
MODES = ("simple",)
MAX_TIER = 4


@dataclass
class Network:
    industry: Dict[str, str] = field(default_factory=dict)
    countries: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    suppliers: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    link_rows: Dict[Tuple[str, str], List[int]] = field(default_factory=lambda: defaultdict(list))
    skipped_rows: List[int] = field(default_factory=list)

    def located_in(self, company: str) -> List[str]:
        return self.countries.get(company, [])


def load(csv_path: Path = CSV_PATH) -> Network:
    net = Network()
    with open(csv_path, encoding="utf-8", errors="ignore", newline="") as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            if any(not (row.get(k) or "").strip() for k in REQUIRED):
                net.skipped_rows.append(line)
                continue
            s, si, sc = (row[k].strip() for k in REQUIRED[:3])
            c, ci, cc = (row[k].strip() for k in REQUIRED[3:])
            for company, ind, country in ((s, si, sc), (c, ci, cc)):
                net.industry.setdefault(company, ind)
                if country not in net.countries[company]:
                    net.countries[company].append(country)
            if (s, c) not in net.link_rows:
                net.suppliers[c].append(s)
            net.link_rows[(s, c)].append(line)
    return net


def paths(net: Network, firm: str, length: int, mode: str) -> Iterator[Tuple[str, ...]]:
    """Every upstream path of the given length from the firm: (firm, Tier-1 supplier, ..., Tier-length supplier)."""
    if mode not in MODES:
        raise ValueError(mode)

    def extend(path: List[str]):
        if len(path) == length + 1:
            yield tuple(path)
            return
        for supplier in net.suppliers.get(path[-1], ()):
            if supplier in path:
                continue
            path.append(supplier)
            yield from extend(path)
            path.pop()

    yield from extend([firm])


def select_country(countries: Sequence[str], disrupted: Set[str]) -> str:
    from offline.names import country_label
    return country_label(countries, disrupted)


def is_disrupted(net: Network, company: str, countries: Set[str], companies: Set[str],
                 industries: Optional[Set[str]] = None) -> bool:
    """A named disrupted company, or a company located in a disrupted country; when the disruption is confined to
    named industries (industries given), the company must also operate in one of them. Arguments are graph spellings
    (offline.names.Resolver.disruption()["match"])."""
    if company in companies:
        return True
    in_country = any(c in countries for c in net.located_in(company))
    if industries is None:
        return in_country
    return in_country and net.industry.get(company) in industries


def disrupted_chains(net: Network, firm: str, countries: Sequence[str], companies: Sequence[str] = (),
                     mode: str = "simple", industries: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """The tier_1 ... tier_4 lists in the format of tools/full_supply_chain_path_tool.build_disrupted_supply_chains."""
    countries = sorted({str(c).strip() for c in countries or [] if str(c).strip()})
    companies = sorted({str(c).strip() for c in companies or [] if str(c).strip()})
    ind = None if industries is None else set(industries)
    out: Dict[str, object] = {f"tier_{t}": [] for t in range(1, MAX_TIER + 1)}
    if countries or companies:
        cset, kset = set(countries), set(companies)
        for t in range(1, MAX_TIER + 1):
            chains = []
            for p in paths(net, firm, t, mode):
                if not is_disrupted(net, p[-1], cset, kset, ind):
                    continue
                chains.append([{"company": n, "country": select_country(net.located_in(n), cset),
                                "industry": net.industry.get(n) or "Unknown"} for n in p])
            chains.sort(key=lambda ch: (ch[-1]["company"], ch[-1]["country"], len(ch),
                                        tuple((n["company"], n["country"], n["industry"]) for n in ch)))
            out[f"tier_{t}"] = chains
    out["monitored_company"] = firm
    out["disrupted_countries"] = countries
    out["summary"] = {"total_disrupted_chains": sum(len(out[f"tier_{t}"]) for t in range(1, MAX_TIER + 1)),
                      **{f"tier_{t}_count": len(out[f"tier_{t}"]) for t in range(1, MAX_TIER + 1)}}
    return out


def metrics_graph_edges(net: Network, firm: str, mode: str, max_depth: int = MAX_TIER) -> Set[Tuple[str, str]]:
    """Links on the firm's upstream paths of length 1 to max_depth, each oriented from the firm towards its suppliers
    (customer -> supplier), as in the graph on which the risk tools compute centrality, PageRank and dependency."""
    if mode not in MODES:
        raise ValueError(mode)
    edges: Set[Tuple[str, str]] = set()
    for length in range(0, max_depth):
        for p in paths(net, firm, length, "simple"):
            for supplier in net.suppliers.get(p[-1], ()):
                if supplier not in p:
                    edges.add((p[-1], supplier))
    return edges
