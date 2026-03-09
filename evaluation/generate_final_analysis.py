#!/usr/bin/env python3
"""
Chapter 5 — final analysis and publication-quality figures.

Design:  3 figures only.  Each tells one visual story.
  Fig 5.3  Outcome stability       (horizontal bar)
  Fig 5.4  Noise resilience        (line chart)
  Fig 5.5  Multi-model comparison  (grouped bar)

Style:  Arial / Helvetica · vibrant blue-green palette · zero grids
        no titles · left+bottom spines only · 600 dpi PDF
"""

import json, statistics
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "ch5_evaluation" / "comprehensive" / "results"
GT_DIR  = ROOT / "evaluation" / "ground_truth_multi_company"
WS67    = RESULTS / "ws6_ws7_report.json"
OUT     = RESULTS / "final_analysis"
OUT.mkdir(exist_ok=True)

# ── vibrant blue-green palette ────────────────────────────────────────────────
B1 = '#003EAA'   # deep royal blue
B2 = '#0066E6'   # vibrant blue
B3 = '#2D8CFF'   # bright blue
B4 = '#77B8FF'   # light blue
G1 = '#006B3C'   # deep green
G2 = '#00994C'   # vivid green
G3 = '#2DBE73'   # bright green
G4 = '#7BE0A8'   # light green
GY = '#5A5A5A'   # neutral grey
LG = '#C7C7C7'   # light grey (axis only)

def style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'Helvetica',
                            'DejaVu Sans', 'sans-serif'],
        'font.size': 10,
        'axes.labelsize': 10.5,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.7,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#222222',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'axes.grid': False,
        'legend.frameon': False,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
    })

W = 5.8   # column width
H = 3.5   # default height

def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=600,
                    facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  ✓ {name}')

# ── data helpers ──────────────────────────────────────────────────────────────

def safe_dict(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except: pass
    return {}

def normalize_type(t):
    return t.lower().strip().replace('labour', 'labor')

def load_gt():
    gts = {}
    for f in GT_DIR.glob("*.json"):
        if f.stem == 'index': continue
        gts[f.stem] = json.loads(f.read_text())
    return gts

def is_true_negative(gt):
    kg = gt.get('kg_results', {})
    chains = sum(len(kg.get(f'tier_{t}', [])) for t in range(1, 5))
    risk = gt.get('risk_assessment', {}).get('supplier_risk_scores', {})
    return chains == 0 and len(risk) == 0

def eval_dm_type(dm, gt):
    gt_t = normalize_type(gt.get('disruption_analysis', {}).get('type', ''))
    pr_t = normalize_type(dm.get('type', dm.get('disruption_type', '')))
    return gt_t == pr_t

def eval_dm_entity_f1(dm, gt):
    gi = gt.get('disruption_analysis', {}).get('involved', {})
    pi = dm.get('involved', {})
    ge, pe = set(), set()
    for c in ['countries', 'industries', 'companies']:
        for e in gi.get(c, []): ge.add(e.lower().strip())
        for e in pi.get(c, dm.get(f'affected_{c}', [])):
            if isinstance(e, str): pe.add(e.lower().strip())
    if not ge and not pe: return 1.0
    if not pe: return 0.0
    tp = len(ge & pe)
    p = tp / len(pe); r = tp / len(ge) if ge else 0
    return 2*p*r/(p+r) if (p+r) > 0 else 0


def eval_dm_composite(dm, gt):
    """Equal-weight F1 across the four critical DM output fields."""
    gt_da = gt.get('disruption_analysis', {})
    gt_inv = gt_da.get('involved', {})
    inv = dm.get('involved', {})

    type_f1 = 1.0 if eval_dm_type(dm, gt) else 0.0

    def set_f1(pred, gold):
        g = set((x or '').lower().strip() for x in gold if isinstance(x, str))
        p = set((x or '').lower().strip() for x in pred if isinstance(x, str))
        if not g and not p:
            return 1.0
        if not p:
            return 0.0
        tp = len(g & p)
        prec = tp / len(p)
        rec = tp / len(g) if g else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    f_c = set_f1(inv.get('countries', []), gt_inv.get('countries', []))
    f_i = set_f1(inv.get('industries', []), gt_inv.get('industries', []))
    f_co = set_f1(inv.get('companies', []), gt_inv.get('companies', []))
    return statistics.mean([type_f1, f_c, f_i, f_co])


def eval_dm_strict(dm, gt, tau=0.8):
    has_schema = isinstance(dm, dict) and ('type' in dm) and ('involved' in dm)
    if not has_schema:
        return False
    return eval_dm_composite(dm, gt) >= tau

def eval_kg_strict(kg, gt, tn):
    gc = sum(len(gt.get('kg_results',{}).get(f'tier_{t}',[]))for t in range(1,5))
    pc = 0
    for t in range(1,5):
        for k in [f'tier_{t}', f'tier_{t}_chains']:
            v = kg.get(k,[])
            if isinstance(v,list): pc += len(v)
    cc = kg.get('chain_counts',{})
    if isinstance(cc,dict):
        for v in cc.values():
            if isinstance(v,(int,float)): pc += int(v)
    if tn: return pc == 0
    if gc == 0: return pc == 0
    return pc > 0

def eval_risk_strict(risk, gt, tn):
    gr = gt.get('risk_assessment',{}).get('supplier_risk_scores',{})
    pr = risk.get('supplier_risk_scores', risk.get('tier1_supplier_risks',{}))
    if isinstance(pr,str):
        try: pr = json.loads(pr)
        except: pr = {}
    if not isinstance(pr,dict): pr = {}
    if tn: return len(pr)==0
    if not gr: return len(pr)==0
    ov = len(set(gr)&set(pr))
    return (ov/len(gr) if gr else 0) >= 0.8

def eval_csco_strict(csco, gt, tn):
    pd = csco.get('decisions', csco.get('supplier_decisions',[]))
    if tn: return not pd or len(pd)==0
    return isinstance(pd,(list,dict)) and len(pd)>0

def eval_scenario(fpath, gt):
    d = json.loads(Path(fpath).read_text())
    a = d.get('agent_outputs',{})
    dm=safe_dict(a.get('disruption_analysis',{}))
    kg=safe_dict(a.get('kg_results',{}))
    ri=safe_dict(a.get('risk_assessment',{}))
    cs=safe_dict(a.get('chief_supply_chain_output',{}))
    tn=is_true_negative(gt)
    r={'dm_type':eval_dm_type(dm,gt),
       'dm_f1_entity':eval_dm_entity_f1(dm,gt),
       'dm_f1_composite':eval_dm_composite(dm,gt),
       'dm_strict':eval_dm_strict(dm,gt),
       'kg_strict':eval_kg_strict(kg,gt,tn),'risk_strict':eval_risk_strict(ri,gt,tn),
       'csco_strict':eval_csco_strict(cs,gt,tn),'latency':d.get('runtime_seconds',0),'tn':tn}
    r['constrained']=all([r['dm_strict'],r['kg_strict'],r['risk_strict'],r['csco_strict']])
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_ws1(gts):
    """Compute pass/fail stability: a scenario is stable for a given agent
    if all 5 runs produce the same pass/fail outcome at tau_q = 0.80.

    Uses ws5_final/gpt-4o data and the AUTHORITATIVE ch5_evaluation_harness
    functions (the same functions that produce the baseline Table 5.1 numbers).
    This ensures stability is measured against the same metric the thesis
    reports for baseline performance."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "ch5_evaluation"))
    from ch5_evaluation_harness import (
        eval_disruption_monitoring as _eval_dm,
        eval_kg_query as _eval_kg,
        eval_risk_manager as _eval_risk,
        eval_csco as _eval_csco,
    )
    d = RESULTS / "ws5_final" / "gpt-4o"
    if not d.exists():
        d = RESULTS / "ws1_stability"  # fallback
    sr = defaultdict(list)
    for f in sorted(d.glob("*.json")):
        sr[f.stem.rsplit('_run',1)[0]].append(f)
    comp = {s: fs for s, fs in sr.items() if len(fs) >= 5 and s in gts}
    n = len(comp)
    st = {'dm':0,'kg':0,'risk':0,'csco':0,'full':0}
    for sc, rfs in comp.items():
        gt = gts[sc]
        dm_pfs, kg_pfs, ri_pfs, cs_pfs = [], [], [], []
        for fp in rfs:
            dd = json.loads(fp.read_text())
            ao = dd.get('agent_outputs',{})
            dm_pfs.append(_eval_dm(ao, gt).strict_success)
            kg_pfs.append(_eval_kg(ao, gt).strict_success)
            ri_pfs.append(_eval_risk(ao, gt).strict_success)
            cs_pfs.append(_eval_csco(ao, gt).strict_success)
        dm_s = len(set(dm_pfs)) == 1
        kg_s = len(set(kg_pfs)) == 1
        ri_s = len(set(ri_pfs)) == 1
        cs_s = len(set(cs_pfs)) == 1
        if dm_s: st['dm'] += 1
        if kg_s: st['kg'] += 1
        if ri_s: st['risk'] += 1
        if cs_s: st['csco'] += 1
        if dm_s and kg_s and ri_s and cs_s: st['full'] += 1
    return {'n': n, 'stability': {k: v/n if n else 0 for k, v in st.items()},
            'counts': st}

def analyse_ws2(gts):
    d = RESULTS/"ws2_prompt_sensitivity"/"runs"
    sr = defaultdict(list)
    for f in sorted(d.glob("*.json")):
        sr[f.stem.rsplit('_para',1)[0]].append(f)
    comp = {s: fs for s, fs in sr.items() if len(fs) >= 3}
    tc = ec = 0
    for sc, rfs in comp.items():
        ts, es = [], []
        for fp in rfs:
            dd = json.loads(fp.read_text())
            dm = safe_dict(dd.get('agent_outputs',{}).get('disruption_analysis',{}))
            ts.append(normalize_type(dm.get('type',dm.get('disruption_type',''))))
            inv = dm.get('involved',{})
            ent = set()
            for c in ['countries','industries','companies']:
                for e in inv.get(c,[]): ent.add(e.lower().strip())
            es.append(frozenset(ent))
        if len(set(ts)) == 1: tc += 1
        if len(set(es)) == 1: ec += 1
    n = len(comp)
    return {'n': n, 'type_rate': tc/n if n else 0, 'entity_rate': ec/n if n else 0,
            'type_count': tc, 'entity_count': ec}

def analyse_ws3(gts):
    d = RESULTS/"ws3_noise_sensitivity"/"runs"
    nr = {'mild':[],'moderate':[],'high':[]}
    for f in sorted(d.glob("*.json")):
        b = f.stem
        for lv in ['mild','moderate','high']:
            if b.endswith(f'_{lv}'):
                sc = b[:-len(f'_{lv}')]; break
        else: continue
        if sc not in gts: continue
        dd = json.loads(f.read_text())
        dm = safe_dict(dd.get('agent_outputs',{}).get('disruption_analysis',{}))
        nr[lv].append({'scenario':sc,
                       'type_correct':eval_dm_type(dm, gts[sc]),
                       'entity_f1':eval_dm_entity_f1(dm, gts[sc])})
    allsc = set()
    for lv in nr:
        for r in nr[lv]: allsc.add(r['scenario'])
    # matched baseline
    imp = ROOT / "evaluation_results" / "ch5_improved_results"
    bl = []
    if imp.exists():
        for dd in imp.iterdir():
            if not dd.is_dir(): continue
            sid = dd.name.split('_2025')[0]
            if sid not in allsc or sid not in gts: continue
            ao = dd / "agent_outputs.json"
            if ao.exists():
                data = json.loads(ao.read_text())
                dm = safe_dict(data.get('disruption_analysis',{}))
                bl.append({'scenario':sid,
                           'type_correct':eval_dm_type(dm, gts[sid]),
                           'entity_f1':eval_dm_entity_f1(dm, gts[sid])})
    sm = {}
    for lv in ['mild','moderate','high']:
        rs = nr[lv]; n = len(rs)
        if n == 0: sm[lv] = {'n':0}; continue
        ta = sum(1 for r in rs if r['type_correct'])
        f1s = [r['entity_f1'] for r in rs]
        sm[lv] = {'n':n, 'type_acc':ta/n, 'f1_mean':statistics.mean(f1s),
                  'f1_std':statistics.stdev(f1s) if len(f1s)>1 else 0}
    if bl:
        bn = len(bl); bt = sum(1 for r in bl if r['type_correct'])
        bf = [r['entity_f1'] for r in bl]
        sm['clean'] = {'n':bn, 'type_acc':bt/bn if bn else 0,
                       'f1_mean':statistics.mean(bf) if bf else 0,
                       'f1_std':statistics.stdev(bf) if len(bf)>1 else 0}
    return sm

def analyse_ws5(gts):
    models = {}
    wd = RESULTS / "ws5_multi_model"
    for md in wd.iterdir():
        if not md.is_dir(): continue
        rs = []; lats = []
        for f in sorted(md.glob("*.json")):
            if f.stem not in gts: continue
            r = eval_scenario(f, gts[f.stem]); r['scenario'] = f.stem
            rs.append(r); lats.append(r['latency'])
        if not rs: continue
        n = len(rs)
        models[md.name] = {
            'n': n,
            'dm_type': sum(1 for r in rs if r['dm_type'])/n,
            'dm_f1_entity': statistics.mean([r['dm_f1_entity'] for r in rs]),
            'dm_f1_composite': statistics.mean([r['dm_f1_composite'] for r in rs]),
            'dm_strict': sum(1 for r in rs if r['dm_strict'])/n,
            'kg':      sum(1 for r in rs if r['kg_strict'])/n,
            'risk':    sum(1 for r in rs if r['risk_strict'])/n,
            'csco':    sum(1 for r in rs if r['csco_strict'])/n,
            'e2e':     sum(1 for r in rs if r['constrained'])/n,
            'lat_mean': statistics.mean(lats),
            'lat_std':  statistics.stdev(lats) if n>1 else 0,
            'per_scenario': {r['scenario']: r for r in rs},
        }
    return models

def analyse_ws7():
    if not WS67.exists(): return {}
    data = json.loads(WS67.read_text())
    ps = data.get('per_scenario',[])
    if not ps: return {}
    at = defaultdict(list)
    for sc in ps:
        for ag, info in sc.get('token_breakdown',{}).items():
            at[ag].append(info.get('output_tokens_approx', 0))
    return {ag: {'mean': statistics.mean(vs),
                 'median': statistics.median(vs),
                 'std': statistics.stdev(vs) if len(vs)>1 else 0,
                 'min': min(vs), 'max': max(vs)}
            for ag, vs in at.items()}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# ── Fig 5.3: Outcome Stability ───────────────────────────────────────────────
def plot_stability(ws1, baseline=None):
    if not HAS_PLOT: return
    style()

    stab = ws1['stability']
    agents = ['dm', 'kg', 'risk', 'csco', 'full']
    labels = ['Disruption\nMonitor', 'KG Query', 'Risk\nManager', 'CSCO',
              'Full\nSystem']
    stability_rates = [stab[a] for a in agents]

    # Get strict success from baseline (GPT-4o 5-run average)
    if baseline:
        strict_map = {
            'dm': baseline.get('dm_strict', 0.8),
            'kg': baseline.get('kg', 0.933),
            'risk': baseline.get('risk', 1.0),
            'csco': baseline.get('csco', 0.953),
            'full': baseline.get('e2e', 0.693),
        }
    else:
        strict_map = {'dm': 0.8, 'kg': 0.933, 'risk': 1.0,
                      'csco': 0.953, 'full': 0.693}
    strict_rates = [strict_map[a] for a in agents]

    fig, ax = plt.subplots(figsize=(W + 0.5, H + 0.2))
    x = np.arange(len(agents))
    bw = 0.34

    bars1 = ax.bar(x - bw/2, stability_rates, bw, label='Outcome stability',
                   color=B2, edgecolor='white', linewidth=0.4, zorder=3)
    bars2 = ax.bar(x + bw/2, strict_rates, bw, label='Strict success',
                   color=G2, edgecolor='white', linewidth=0.4, zorder=3)

    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    min(yval + 0.02, 1.06),
                    f'{yval:.0%}', ha='center', va='bottom',
                    fontsize=7.2, color='#1f1f1f')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, linespacing=1.05)
    ax.set_ylabel('Rate')
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(loc='upper center', ncol=2, fontsize=7.5, framealpha=0.9)

    # Sample size note — top right, clear of bars and legend
    ax.text(0.99, 0.97, f'n = {ws1["n"]} scenarios, 5 runs each',
            transform=ax.transAxes, fontsize=6.5, color=GY,
            ha='right', va='top', style='italic')

    fig.tight_layout()
    save(fig, 'outcome_stability')


# ── Fig 5.4: Noise Resilience ────────────────────────────────────────────────
def plot_noise(ws3):
    if not HAS_PLOT: return
    style()

    cl = ws3.get('clean', {})
    if not cl or cl.get('n',0) == 0:
        cl = {'type_acc': 0.70, 'f1_mean': 0.89, 'f1_std': 0.10}

    labels = ['Clean', 'Mild', 'Moderate', 'High']
    x = np.arange(len(labels))

    f1  = [cl['f1_mean']]
    err = [cl.get('f1_std', 0)]
    ta  = [cl['type_acc']]
    for lv in ['mild', 'moderate', 'high']:
        d = ws3.get(lv, {})
        f1.append(d.get('f1_mean', 0))
        err.append(d.get('f1_std', 0))
        ta.append(d.get('type_acc', 0))

    fig, ax = plt.subplots(figsize=(W, H))
    f1a, ea = np.array(f1), np.array(err)

    # Shaded std band — very light blue fill
    ax.fill_between(x, np.clip(f1a - ea, 0, 1), np.clip(f1a + ea, 0, 1),
                    color=B4, alpha=0.18, linewidth=0)

    # Entity F1 — strong blue
    ax.plot(x, f1, color=B2, linewidth=2.0, marker='o', markersize=7,
            markerfacecolor=B2, markeredgecolor='white', markeredgewidth=1.0,
            label='Entity extraction F1', zorder=4)

    # Type accuracy — vibrant green
    ax.plot(x, ta, color=G2, linewidth=2.0, linestyle='--',
            marker='s', markersize=6, markerfacecolor=G2,
            markeredgecolor='white', markeredgewidth=1.0,
            label='Type classification accuracy', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Input noise level')
    ax.set_ylabel('Score')
    ax.set_ylim(0.45, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.legend(loc='lower left', framealpha=0)

    fig.tight_layout()
    save(fig, 'noise_degradation')


# ── Fig 5.5: Multi-Model Comparison ──────────────────────────────────────────
def plot_multimodel(ws5, baseline):
    """Generate multi-model comparison figure.
    
    Prefers authoritative 5-run data from ws5_final_multimodel_summary.json
    (generated by ws5_final_analysis.py) over single-run ws5 data.
    """
    if not HAS_PLOT: return
    style()

    # Try to load the authoritative 5-run summary
    summary_path = OUT / 'ws5_final_multimodel_summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            summaries = json.load(f)
        # Use _mean keys from the summary (per-run averaged strict success)
        agents = [
            ('dm_strict',   'Disruption\nMonitor'),
            ('kg_strict',   'KG Query'),
            ('risk_strict', 'Risk\nManager'),
            ('csco_strict', 'CSCO'),
            ('constrained', 'End-to-end'),
        ]
        model_order = ['gpt-4o', 'gpt-4.1', 'gpt-5-mini']
        colours = [B1, B3, G3]
        labels = ['GPT-4o (baseline)', 'gpt-4.1', 'gpt-5-mini']
        series = []
        for m, lab, col in zip(model_order, labels, colours):
            if m in summaries:
                series.append((lab, summaries[m], col))
        nm = len(series)

        fig, ax = plt.subplots(figsize=(W + 0.5, H + 0.2))
        x = np.arange(len(agents))
        bw = 0.72 / nm

        for i, (label, md, col) in enumerate(series):
            vals = [md.get(f'{key}_mean', 0) for key, _ in agents]
            off = (i - nm/2 + 0.5) * bw
            bars = ax.bar(x + off, vals, bw, color=col, edgecolor='white',
                          linewidth=0.4, label=label, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, min(v + 0.025, 1.06),
                        f'{v:.0%}', ha='center', va='bottom',
                        fontsize=7.2, color='#1f1f1f')

        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in agents], linespacing=1.05)
        ax.set_ylabel('Strict success rate')
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.legend(loc='upper right', fontsize=7.5)

        fig.tight_layout()
        save(fig, 'multi_model_comparison')
        return

    # Fallback: use single-run ws5 data + hardcoded baseline (less accurate)
    metrics = [
        ('dm_strict', 'Disruption\nMonitor'),
        ('kg',      'KG Query'),
        ('risk',    'Risk\nManager'),
        ('csco',    'CSCO'),
        ('e2e',     'End-to-end'),
    ]

    series = [
        ('GPT-4o (baseline)', baseline, B1),
        ('gpt-4.1',            ws5.get('gpt-4_1', {}), B3),
        ('gpt-5-mini',         ws5.get('gpt-5-mini', {}), G3),
    ]
    series = [(l, d, c) for l, d, c in series if d]
    nm = len(series)

    fig, ax = plt.subplots(figsize=(W + 0.5, H + 0.2))
    x = np.arange(len(metrics))
    bw = 0.72 / nm

    for i, (label, md, col) in enumerate(series):
        vals = [md.get(m, 0) for m, _ in metrics]
        off = (i - nm/2 + 0.5) * bw
        bars = ax.bar(x + off, vals, bw, color=col, edgecolor='white',
                      linewidth=0.4, label=label, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, min(v + 0.025, 1.06),
                    f'{v:.0%}', ha='center', va='bottom',
                    fontsize=7.2, color='#1f1f1f')

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics], linespacing=1.05)
    ax.set_ylabel('Strict success rate')
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.legend(loc='upper right', fontsize=7.5)

    fig.tight_layout()
    save(fig, 'multi_model_comparison')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading ground truth …")
    gts = load_gt()
    print(f"  {len(gts)} scenarios")

    # Load authoritative 5-run averaged baseline from ws5_final_multimodel_summary.json
    ws5_summary_path = OUT / 'ws5_final_multimodel_summary.json'
    if ws5_summary_path.exists():
        with open(ws5_summary_path) as f:
            ws5_summaries = json.load(f)
        gpt4o = ws5_summaries.get('gpt-4o', {})
        baseline = {
            'dm_type': gpt4o.get('dm_strict_mean', 0.8),
            'dm_f1_entity': 0.886,
            'dm_f1_composite': gpt4o.get('dm_f1_mean', 0.913),
            'dm_strict': gpt4o.get('dm_strict_mean', 0.8),
            'kg': gpt4o.get('kg_strict_mean', 0.933),
            'risk': gpt4o.get('risk_strict_mean', 1.0),
            'csco': gpt4o.get('csco_strict_mean', 0.953),
            'e2e': gpt4o.get('constrained_mean', 0.693),
            'lat_mean': gpt4o.get('runtime_mean', 102.5),
            'lat_std': gpt4o.get('runtime_std', 37.5),
            'n': gpt4o.get('n_scenarios', 30),
            'dm_strict_pass_count': gpt4o.get('dm_strict_pass_count', 24),
            'kg_strict_pass_count': gpt4o.get('kg_strict_pass_count', 28),
            'risk_strict_pass_count': gpt4o.get('risk_strict_pass_count', 30),
            'csco_strict_pass_count': gpt4o.get('csco_strict_pass_count', 30),
            'constrained_pass_count': gpt4o.get('constrained_pass_count', 22),
        }
    else:
        # Fallback hardcoded values (per-run rates from 5-run evaluation)
        baseline = {
            'dm_type': 0.8, 'dm_f1_entity': 0.886, 'dm_f1_composite': 0.913, 'dm_strict': 0.8,
            'kg': 0.933, 'risk': 1.0,
            'csco': 0.953, 'e2e': 0.693,
            'lat_mean': 102.5, 'lat_std': 37.5, 'n': 30,
        }

    print("\n── WS1: Outcome Stability ──")
    ws1 = analyse_ws1(gts)
    for a, r in ws1['stability'].items():
        print(f"  {a}: {r:.0%}  ({ws1['counts'][a]}/{ws1['n']})")

    print("\n── WS2: Prompt Sensitivity ──")
    ws2 = analyse_ws2(gts)
    print(f"  Type: {ws2['type_count']}/{ws2['n']} ({ws2['type_rate']:.0%})")
    print(f"  Entity: {ws2['entity_count']}/{ws2['n']} ({ws2['entity_rate']:.0%})")

    print("\n── WS3: Noise Sensitivity ──")
    ws3 = analyse_ws3(gts)
    cl = ws3.get('clean', {})
    if cl.get('n', 0) > 0:
        print(f"  clean  (n={cl['n']}): type {cl['type_acc']:.0%}  F1 {cl['f1_mean']:.3f}")
    for lv in ['mild', 'moderate', 'high']:
        d = ws3.get(lv, {})
        if d.get('n', 0) > 0:
            print(f"  {lv:8s} (n={d['n']}): type {d['type_acc']:.0%}  "
                  f"F1 {d['f1_mean']:.3f} ± {d.get('f1_std',0):.3f}")

    print("\n── WS5: Multi-Model ──")
    ws5 = analyse_ws5(gts)
    for m, d in sorted(ws5.items()):
        print(f"  {m} (n={d['n']}): DM strict {d['dm_strict']:.0%} | "
              f"KG {d['kg']:.0%} | Risk {d['risk']:.0%} | "
              f"CSCO {d['csco']:.0%} | E2E {d['e2e']:.0%} | "
              f"lat {d['lat_mean']/60:.1f}min")

    print("\n── WS7: Per-Agent Cost ──")
    ws7 = analyse_ws7()
    for ag, i in ws7.items():
        print(f"  {ag}: median {i['median']:,.0f}  "
              f"[{i['min']:,}–{i['max']:,}] tokens")

    # Build ws5 report section from authoritative 5-run summary if available
    ws5_report = {}
    if ws5_summary_path.exists():
        with open(ws5_summary_path) as f:
            ws5_auth = json.load(f)
        for model_name, md in ws5_auth.items():
            if model_name == 'gpt-4o':
                continue  # baseline already captured above
            ws5_report[model_name] = {
                'n': md.get('n_scenarios', 30),
                'n_runs': md.get('n_runs_total', 0),
                'dm_f1_composite': md.get('dm_f1_mean', 0),
                'dm_strict': md.get('dm_strict_mean', 0),
                'kg': md.get('kg_strict_mean', 0),
                'risk': md.get('risk_strict_mean', 0),
                'csco': md.get('csco_strict_mean', 0),
                'csco_f1': md.get('csco_f1_mean', 0),
                'e2e': md.get('constrained_mean', 0),
                'lat_mean': md.get('runtime_mean', 0),
                'lat_std': md.get('runtime_std', 0),
                'dm_strict_pass_count': md.get('dm_strict_pass_count', 0),
                'kg_strict_pass_count': md.get('kg_strict_pass_count', 0),
                'risk_strict_pass_count': md.get('risk_strict_pass_count', 0),
                'csco_strict_pass_count': md.get('csco_strict_pass_count', 0),
                'constrained_pass_count': md.get('constrained_pass_count', 0),
            }
    else:
        ws5_report = {k: {kk: vv for kk, vv in v.items()
                          if kk != 'per_scenario'}
                      for k, v in ws5.items()}

    # save report
    report = {'baseline': baseline, 'ws1': ws1, 'ws2': ws2,
              'ws3': ws3, 'ws5': ws5_report, 'ws7': ws7}
    rp = OUT / 'final_analysis_report.json'
    with open(rp, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport → {rp}")

    if HAS_PLOT:
        print("\nGenerating figures …")
        plot_stability(ws1, baseline)
        plot_noise(ws3)
        plot_multimodel(ws5, baseline)
        print("Done — 3 figures generated.")
    else:
        print("matplotlib not available — skipping figures.")


if __name__ == '__main__':
    main()
