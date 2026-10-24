"""Draw the specification of every item of the relevance benchmark (Appendix C.6).

    python -m evaluation.benchmark_v2.specify

Seed 20261014. Writes evaluation/benchmark_v2/items/test/specs.json.

Frame. Tiers are shortest-path tiers on the cleaned network. A positive of tier k is drawn
from the countries (scope "country") or (country, industry) pairs (scope "industries") whose shallowest supplier of
the firm is at tier k, so that the lowest disrupted tier is k. A Company Bankruptcy item names one supplier of the firm
at tier k, with its country and industry and scope "industries", as the Disruption Monitor is instructed to record a
company's insolvency. N1 names a country, an industry and usually a company of the firm's network in business news. N2
is set in one of the countries in none of the three networks. N3 is drawn as a positive and reported as over or as
history. N4 is set in a country of the firm's network, confined to an industry from which the firm has no supplier
there: half an industry of the knowledge graph, half one outside it. Names are the network's; display names are the
English names the articles use, each of which the resolver maps back to the network's name.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from offline import kg, names

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation" / "benchmark_v2" / "items"
SEED = 20261014
SIZES = {"P": 120, "N1": 300, "N2": 250, "N3": 150, "N4": 300}
FIRMS = ("Bayerische Motoren Werke AG", "Mercedes-Benz Group AG", "Tesla Inc")
TYPES = ("Geopolitical", "Trade Policy", "Natural Disaster", "Labour Strike", "Economic Crisis", "Company Bankruptcy",
         "Cyber Attack", "Pandemic")
TOPICS = ("quarterly or annual results", "an investment or expansion", "a senior appointment", "a product launch",
          "routine trade figures")
OUTSIDE_INDUSTRIES = ("Agriculture", "Tourism", "Retail", "Hospitality", "Fisheries", "Banking", "Real Estate",
                      "Media & Publishing")
STYLES = ("wire-service report", "business newspaper article", "trade-press report", "regional news report",
          "financial news brief")
REGIONS = {
    "Europe": ["Austria", "Belarus", "Belgium", "Bosnia And Herzegovina", "Bulgaria", "Cyprus", "Czech Republic",
               "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Guernsey", "Hungary", "Iceland",
               "Ireland", "Italy", "Latvia", "Luxembourg", "Netherlands", "Norway", "Poland", "Romania", "Russia",
               "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom"],
    "Asia-Pacific": ["Australia", "Bangladesh", "Bhutan", "Cambodia", "China", "Hong Kong", "India", "Indonesia",
                     "Japan", "Kazakhstan", "Kyrgyzstan", "Malaysia", "Myanmar", "New Zealand", "Pakistan",
                     "Philippines", "Singapore", "South Korea", "Sri Lanka", "Taiwan", "Thailand", "Uzbekistan",
                     "Vietnam"],
    "Americas": ["Argentina", "Bermuda", "Brazil", "Canada", "Cayman Islands", "Chile", "Colombia", "Costa Rica",
                 "Curacao", "Ecuador", "El Salvador", "Mexico", "Panama", "Peru", "USA", "Uruguay", "Venezuela",
                 "Virgin Islands, British"],
    "Africa and Middle East": ["Algeria", "Azerbaijan", "Bahrain", "Democratic Republic Of The Congo", "Egypt",
                               "Ghana", "Israel", "Jordan", "Kuwait", "Lebanon", "Libyan Arab Jamahiriya",
                               "Madagascar", "Mauritius", "Nigeria", "Qatar", "Saudi Arabia", "South Africa",
                               "Tunisia", "Turkey", "Uganda", "United Arab Emirates", "Zambia", "Zimbabwe"],
}
DISPLAY = {"USA": "the United States", "Bosnia And Herzegovina": "Bosnia and Herzegovina",
           "Democratic Republic Of The Congo": "the Democratic Republic of the Congo", "Libyan Arab Jamahiriya": "Libya",
           "Virgin Islands, British": "the British Virgin Islands", "United Kingdom": "the United Kingdom",
           "Netherlands": "the Netherlands", "Philippines": "the Philippines",
           "United Arab Emirates": "the United Arab Emirates", "Czech Republic": "the Czech Republic",
           "Cayman Islands": "the Cayman Islands"}


def region_of():
    return {c: region for region, countries in REGIONS.items() for c in countries}


class Frame:
    """Shortest-path tiers of each firm's suppliers and the countries and pairs by their shallowest tier."""

    def __init__(self, net):
        self.net = net
        self.tier, self.cmin, self.pmin = {}, {}, {}
        self.by_pair = {}
        for firm in FIRMS:
            dist, front = {firm: 0}, [firm]
            for t in range(1, kg.MAX_TIER + 1):
                nxt = []
                for x in front:
                    for s in net.suppliers.get(x, ()):
                        if s not in dist:
                            dist[s] = t
                            nxt.append(s)
                front = nxt
            sup = {s: d for s, d in dist.items() if d >= 1}
            cmin, pmin, by_pair = defaultdict(lambda: 9), defaultdict(lambda: 9), defaultdict(list)
            for s, d in sup.items():
                for c in net.located_in(s):
                    cmin[c] = min(cmin[c], d)
                    pmin[c, net.industry[s]] = min(pmin[c, net.industry[s]], d)
                    by_pair[c, net.industry[s]].append(s)
            self.tier[firm], self.cmin[firm], self.pmin[firm] = sup, dict(cmin), dict(pmin)
            self.by_pair[firm] = {k: sorted(v) for k, v in by_pair.items()}
        self.all_countries = sorted({c for v in net.countries.values() for c in v})
        in_any = set().union(*(self.cmin[f] for f in FIRMS))
        self.outside = [c for c in self.all_countries if c not in in_any]
        self.located = defaultdict(list)
        for company, countries in net.countries.items():
            for c in countries:
                self.located[c, net.industry.get(company)].append(company)


def display(country):
    return DISPLAY.get(country, country)


def pick(rng, candidates, used_countries, used_regions, regions, country_of=lambda x: x):
    """Prefer an unused region, then an unused country."""
    def rank(x):
        c = country_of(x)
        return (regions.get(c) in used_regions, c in used_countries)
    best = min(rank(x) for x in candidates)
    choice = rng.choice(sorted(x for x in candidates if rank(x) == best))
    used_countries.add(country_of(choice))
    used_regions.add(regions.get(country_of(choice)))
    return choice


def positive_spec(rng, fr, firm, dtype, tier, used_c, used_r, regions):
    """Countries, industries, companies and scope of an ongoing disruption whose lowest disrupted tier is `tier`."""
    scope = "industries" if dtype == "Company Bankruptcy" else rng.choice(("country", "industries"))
    if scope == "country":
        cands = [c for c, t in fr.cmin[firm].items() if t == tier]
        if not cands:
            scope = "industries"
    if scope == "country":
        c = pick(rng, cands, used_c, used_r, regions)
        at_or_below = sorted(s for s, t in fr.tier[firm].items() if t >= tier and c in fr.net.located_in(s))
        companies = [rng.choice(at_or_below)] if at_or_below and rng.random() < 0.5 else []
        return {"scope": "country", "countries": [c], "industries": [], "companies": companies}
    pairs = [p for p, t in fr.pmin[firm].items() if t == tier]
    c, ind = pick(rng, pairs, used_c, used_r, regions, country_of=lambda p: p[0])
    at_tier = sorted(s for s in fr.by_pair[firm][c, ind] if fr.tier[firm][s] == tier)
    named = dtype == "Company Bankruptcy" or rng.random() < 0.5
    return {"scope": "industries", "countries": [c], "industries": [ind],
            "companies": [rng.choice(at_tier)] if named else []}


def cycle(rng, values, n):
    """n values in equal proportions (as near as n allows), in random order."""
    out = []
    while len(out) < n:
        block = list(values)
        rng.shuffle(block)
        out += block
    out = out[:n]
    rng.shuffle(out)
    return out


def positive_cells(rng, n):
    """(type, tier) of each positive: 15 per type and 30 per tier for 120; for a smaller set, distinct types."""
    if n == 120:
        types = list(TYPES)
        rng.shuffle(types)
        cells = []
        for i, t in enumerate(types):
            for tier in range(1, 5):
                cells += [(t, tier)] * (3 if tier == i % 4 + 1 else 4)
        return cells
    types = rng.sample(TYPES, n)
    tiers = [(i % 4) + 1 for i in range(n)]
    return list(zip(types, tiers))


def draw(set_name="test"):
    rng = random.Random(SEED)
    net = kg.load()
    fr = Frame(net)
    regions = region_of()
    missing = [c for c in fr.all_countries if c not in regions]
    assert not missing, missing
    resolver = names.Resolver(net)
    for c in fr.all_countries:
        assert resolver.country(display(c).removeprefix("the ")) == c, c
    sizes = SIZES
    items = []

    def base(kind, i, firm):
        return {"item_id": f"{kind}-{i:03d}", "set": set_name, "item_type": kind, "firm": firm,
                "length_words": rng.randint(250, 450), "style": rng.choice(STYLES)}

    # P
    cells = positive_cells(rng, sizes["P"])
    firms = cycle(rng, FIRMS, len(cells))
    used = defaultdict(lambda: (set(), set()))
    for i, ((dtype, tier), firm) in enumerate(zip(cells, firms), 1):
        uc, ur = used[dtype, tier]
        s = base("P", i, firm)
        s.update(status="ongoing", ongoing_disruption=True, type=dtype, intended_tier=tier,
                 **positive_spec(rng, fr, firm, dtype, tier, uc, ur, regions))
        items.append(s)
    # N1
    firms = cycle(rng, FIRMS, sizes["N1"])
    topics = cycle(rng, TOPICS, sizes["N1"])
    for i, (firm, topic) in enumerate(zip(firms, topics), 1):
        s = base("N1", i, firm)
        c = rng.choice(sorted(fr.cmin[firm]))
        in_c = sorted(x for x in fr.tier[firm] if c in net.located_in(x))
        company = rng.choice(in_c)
        s.update(status="none", ongoing_disruption=False, type="None", topic=topic, scope="country",
                 countries=[c], industries=[net.industry[company]],
                 companies=[company] if rng.random() < 0.7 else [])
        items.append(s)
    # N2
    firms = cycle(rng, FIRMS, sizes["N2"])
    dtypes = cycle(rng, TYPES, sizes["N2"])
    for i, (firm, dtype) in enumerate(zip(firms, dtypes), 1):
        s = base("N2", i, firm)
        c = rng.choice(fr.outside)
        here = sorted({ind for (cc, ind) in fr.located if cc == c})
        scope = "industries" if dtype == "Company Bankruptcy" else rng.choice(("country", "industries"))
        ind = rng.choice(here) if here else rng.choice(OUTSIDE_INDUSTRIES)
        companies = sorted(fr.located.get((c, ind), []))
        named = companies and (dtype == "Company Bankruptcy" or rng.random() < 0.5)
        s.update(status="ongoing", ongoing_disruption=True, type=dtype, scope=scope, countries=[c],
                 industries=[ind] if scope == "industries" else [],
                 companies=[rng.choice(companies)] if named else [],
                 invented_company=dtype == "Company Bankruptcy" and not companies)
        items.append(s)
    # N3
    firms = cycle(rng, FIRMS, sizes["N3"])
    dtypes = cycle(rng, TYPES, sizes["N3"])
    statuses = cycle(rng, ("over", "historical"), sizes["N3"])
    tiers = cycle(rng, (1, 2, 3, 4), sizes["N3"])
    used = defaultdict(lambda: (set(), set()))
    for i, (firm, dtype, status, tier) in enumerate(zip(firms, dtypes, statuses, tiers), 1):
        uc, ur = used[dtype, tier]
        s = base("N3", i, firm)
        s.update(status=status, ongoing_disruption=False, type=dtype, intended_tier=tier,
                 **positive_spec(rng, fr, firm, dtype, tier, uc, ur, regions))
        items.append(s)
    # N4
    firms = cycle(rng, FIRMS, sizes["N4"])
    dtypes = cycle(rng, TYPES, sizes["N4"])
    inside = cycle(rng, (True, False), sizes["N4"])
    kg_industries = sorted(set(net.industry.values()))
    for i, (firm, dtype, in_kg) in enumerate(zip(firms, dtypes, inside), 1):
        s = base("N4", i, firm)
        countries = sorted(fr.cmin[firm])
        if in_kg:
            options = [(c, ind) for c in countries for ind in kg_industries if (c, ind) not in fr.pmin[firm]]
            c, ind = rng.choice(options)
            pool = sorted(x for x in fr.located.get((c, ind), []) if x not in fr.tier[firm])
        else:
            c, ind, pool = rng.choice(countries), rng.choice(OUTSIDE_INDUSTRIES), []
        named = pool and (dtype == "Company Bankruptcy" or rng.random() < 0.5)
        s.update(status="ongoing", ongoing_disruption=True, type=dtype, scope="industries", countries=[c],
                 industries=[ind], industry_in_graph=in_kg, companies=[rng.choice(pool)] if named else [],
                 invented_company=dtype == "Company Bankruptcy" and not pool)
        items.append(s)
    for s in items:
        s["display"] = {"countries": [display(c) for c in s["countries"]]}
    return items, fr


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    items, fr = draw()
    out = OUT / "test"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"seed": SEED, "items": len(items),
               "by_type": dict(Counter(s["item_type"] for s in items)),
               "by_firm": {t: dict(Counter(s["firm"] for s in items if s["item_type"] == t)) for t in SIZES},
               "positive_type_by_tier": dict(Counter(f"{s['type']}|{s['intended_tier']}" for s in items
                                                     if s["item_type"] == "P")),
               "negative_disruption_types": {t: dict(Counter(s["type"] for s in items if s["item_type"] == t))
                                             for t in ("N2", "N3", "N4")}}
    (out / "specs.json").write_text(json.dumps({"summary": summary, "items": items}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
