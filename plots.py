"""Reproduce the figures of the SPES paper from the trace and the results of run_all.py.

usage: python plots.py <data_dir> <results_dir> <figures_dir>

  fig03  distribution of function invocations          fig10  average cold-start rate per SPES type
  fig04  concept shifts in three functions             fig11  wasted memory time and EMCR
  fig05  trigger types of functions                    fig12  wasted memory per invocation, per type
  fig06  temporal locality of infrequent functions     fig13  trade-off: theta_prewarm and theta_givenup
  fig07  function types (first pass/forgetting/final)  fig14  ablation: inter-function correlation
  fig08  CDF of function-wise cold-start rate          fig15  ablation: adaptive strategies
  fig09  memory usage and always-cold functions        summary.md  metrics next to the paper's
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import Trace, rle, summarize, TYPE_NAMES, TRIGGERS, TR, END, DAY, T_TEST, NEW_POSS

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # validated categorical order
METHODS = [("spes", "SPES", SLOT[0], "-"), ("faascache", "FaaSCache", SLOT[1], "--"), ("defuse", "Defuse", SLOT[2], "-."),
           ("hybrid_a", "Hybrid-App (HA)", SLOT[3], ":"), ("hybrid_f", "Hybrid-Func (HF)", SLOT[4], (0, (5, 1, 1, 1))),
           ("fixed", "Fixed keep-alive", SLOT[5], (0, (1, 1)))]
PAPER = {"spes": dict(q3_csr=0.108, zero_cold_pct=57.99, always_cold_pct="<8", wmt=21521200, mem_avg_notebook=13982.1, emcr=46.32),
         "fixed": dict(zero_cold_pct=25.61, q3_csr=1.0), "defuse": dict(q3_csr=0.215), "faascache": dict(zero_cold_pct=52.59)}

plt.rcParams.update({"font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "axes.titlesize": 12, "axes.titleweight": "bold"})


def load(rdir, name):
    p = os.path.join(rdir, f"{name}.npz")
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}
    d.update(json.loads(str(z["meta"])))
    return d


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png")); fig.savefig(os.path.join(out, name + ".pdf")); plt.close(fig)
    print("wrote", name)


def bar_labels(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)


def cdf(ax, x, **kw):
    xs = np.sort(x); ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(np.r_[0, xs], np.r_[0, ys], where="post", lw=1.8, **kw)


# ---------------------------------------------------------------- characterization (Figs. 3-7)
def fig03(tr, out):
    tot = np.add.reduceat(tr.C.astype(np.int64), tr.ip[:-1])
    tot[np.diff(tr.ip) == 0] = 0
    edges = 10 ** np.arange(0, 10); labels = ["1–9", "10–99", "100–999", "1K–10K", "10K–100K", "100K–1M", "1M–10M", "10M–100M", "≥100M"]
    cnt = [int(((tot >= edges[i]) & (tot < (edges[i + 1] if i + 1 < len(edges) else np.inf))).sum()) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ax.bar(labels, cnt, color=SLOT[0], width=0.7)
    bar_labels(ax, bars, lambda v: f"{int(v):,}")
    ax.set_xlabel("Invocations per function over 14 days"); ax.set_ylabel("Number of functions")
    ax.set_title("Fig. 3 — Most functions are rarely invoked"); ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    save(fig, out, "fig03_invocation_distribution")


def fig04(tr, out):
    daily = np.zeros((tr.nF, 14))
    f_of = np.repeat(np.arange(tr.nF), np.diff(tr.ip))
    np.add.at(daily, (f_of, tr.M // DAY), tr.C)
    best = []
    for f in np.where((daily > 0).sum(1) == 14)[0]:
        d = daily[f]; sc = 0; kk = 0
        for k in range(3, 12):
            a, b = d[:k], d[k:]
            if a.std() / a.mean() < 0.25 and b.std() / b.mean() < 0.25 and min(a.mean(), b.mean()) >= 100:
                s = abs(np.log10(a.mean()) - np.log10(b.mean()))
                if s > sc: sc, kk = s, k
        if sc > 0.5: best.append((sc, f, kk))
    best.sort(reverse=True)
    picks, owners = [], set()
    for sc, f, k in best:
        if tr.owner[f] in owners: continue
        picks.append((f, k)); owners.add(tr.owner[f])
        if len(picks) == 3: break
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 2.1 * len(picks)), sharex=True)
    for ax, (f, k), lab in zip(np.atleast_1d(axes), picks, "ABC"):
        ms, cs = tr.window(f, 0, END)
        hourly = np.bincount(ms // 60, weights=cs, minlength=END // 60)
        x = np.arange(len(hourly)) / 24
        ax.bar(x[: k * 24], hourly[: k * 24], width=1 / 24, color=SLOT[0], label="before the shift")
        ax.bar(x[k * 24:], hourly[k * 24:], width=1 / 24, color=SLOT[1], label="after the shift")
        ax.axvline(k, color=INK2, lw=1, ls="--")
        ax.set_ylabel("calls / hour")
        ax.set_title(f"Function {lab}: {daily[f, :k].mean():,.0f} → {daily[f, k:].mean():,.0f} calls/day after day {k}",
                     fontsize=10, loc="left", fontweight="normal")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[-1].set_xlabel("Day of the trace")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Fig. 4 — Invocation patterns shift over time", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, out, "fig04_concept_shift")


def fig05(tr, out):
    trig = tr.meta["triggers"].astype(str)
    lab = np.where(trig.str.contains(r"\|"), "combination", trig)
    names, cnt = np.unique(lab, return_counts=True)
    o = np.argsort(cnt)
    pct = cnt[o] / cnt.sum() * 100
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.barh(names[o], pct, color=SLOT[0], height=0.65)
    for b, v in zip(bars, pct):
        ax.annotate(f"{v:.1f}%", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_xlabel("Share of functions (%)"); ax.grid(axis="y", visible=False)
    ax.set_title(f"Fig. 5 — Trigger types of functions ({(lab == 'combination').mean() * 100:.1f}% have more than one)")
    save(fig, out, "fig05_trigger_types")


def fig06(tr, spes, out):
    typ = spes["type_after_forget"]
    cands = []
    for f in np.where(typ == 5)[0]:            # successive functions with few calls
        ms, cs = tr.window(f, 0, END)
        if 20 <= len(ms) <= 400:
            wt, at, an, _, _ = rle(ms, cs, 0, END)
            if len(at) >= 3: cands.append((len(at), f))
    cands.sort()
    picks = [f for _, f in cands[: 5]]
    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 1.4 * len(picks) + 0.6), sharex=True)
    for ax, f, lab in zip(np.atleast_1d(axes), picks, "ABCDE"):
        ms, cs = tr.window(f, 0, END)
        hourly = np.bincount(ms // 60, weights=cs, minlength=END // 60)
        h = (hourly - hourly.min()) / (hourly.max() - hourly.min())
        ax.bar(np.arange(len(h)) / 24, h, width=1 / 24 * 1.6, color=SLOT[0])
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 1]); ax.grid(axis="x", visible=False)
        ax.set_ylabel(lab, rotation=0, labelpad=12, va="center", fontweight="bold")
        ax.text(1.0, 0.9, f"{int(cs.sum()):,} calls in {len(rle(ms, cs, 0, END)[1])} bursts", transform=ax.transAxes,
                ha="right", fontsize=9, color=INK2)
    np.atleast_1d(axes)[-1].set_xlabel("Day of the trace  (bars: min-max normalized calls per hour)")
    fig.suptitle("Fig. 6 — Infrequent functions arrive in short bursts (temporal locality)", fontweight="bold", x=0.02, ha="left")
    save(fig, out, "fig06_temporal_locality")


def fig07(tr, spes, out):
    sets = [("First pass", spes["type_first_pass"]), ("After forgetting", spes["type_after_forget"]),
            ("Final (after assignment)", spes["type_train"])]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(9); h = 0.26
    for i, (lab, t) in enumerate(sets):
        cnt = np.bincount(t[t >= 0], minlength=10)[:9]
        bars = ax.barh(y + (i - 1) * h, cnt, height=h * 0.92, color=SLOT[i], label=lab)
        for b, v in zip(bars, cnt):
            if v > 0:
                ax.annotate(f"{v:,}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(3, 0),
                            textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, TYPE_NAMES[:9]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Number of training functions"); ax.legend(loc="lower right")
    ax.set_title("Function types found by SPES on days 1–12")
    save(fig, out, "fig07_function_types")


# ---------------------------------------------------------------- evaluation (Figs. 8-15)
def fig08(R, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for key, lab, col, ls in METHODS:
        r = R.get(key)
        if r is None: continue
        csr = r["cold"] / r["inv"]
        cdf(ax, csr, color=col, ls=ls, label=f"{lab}  (Q3 = {np.percentile(csr, 75):.3f})")
    ax.axhline(0.75, color=INK2, lw=0.8, ls=":")
    ax.text(0.62, 0.755, "75th percentile", ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("Cold-start rate (cold starts ÷ invocations)"); ax.set_ylabel("Fraction of functions (CDF)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Fig. 8 — Cold-start rate per function: SPES vs. baselines")
    ax.text(0.0, -0.2, "HA is measured per application, as in the paper; all others per function.", transform=ax.transAxes, fontsize=8.5, color=INK2)
    save(fig, out, "fig08_csr_cdf")


def _avail(R):
    return [(k, l, c) for k, l, c, _ in METHODS if R.get(k) is not None]


def fig09(R, S, out):
    ms = _avail(R)
    base = np.mean(R["spes"]["used_t"] + R["spes"]["waste_t"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    data = [(R[k]["used_t"] + R[k]["waste_t"]) / base for k, _, _ in ms]
    bp = a1.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color=INK))
    for p, (_, _, c) in zip(bp["boxes"], ms): p.set_facecolor(c); p.set_edgecolor(INK2)
    a1.set_xticks(range(1, len(ms) + 1), [l for _, l, _ in ms], rotation=25, ha="right")
    a1.set_ylabel("Memory per minute ÷ SPES average"); a1.axhline(1, color=INK2, lw=0.8, ls=":")
    a1.set_title("(a) Memory usage (functions loaded per minute)", fontsize=11)
    if "faascache" in [k for k, _, _ in ms]:
        i = [k for k, _, _ in ms].index("faascache") + 1
        a1.annotate("held at its cap", (i, np.median(data[i - 1])), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ac = [S[k]["always_cold_pct"] for k, _, _ in ms]
    bars = a2.bar([l for _, l, _ in ms], ac, color=[c for _, _, c in ms], width=0.65)
    bar_labels(a2, bars, lambda v: f"{v:.1f}%")
    a2.set_ylabel("Always-cold functions (%)"); plt.setp(a2.get_xticklabels(), rotation=25, ha="right")
    a2.set_title("(b) Functions cold on every invocation", fontsize=11); a2.grid(axis="x", visible=False)
    fig.suptitle("Fig. 9 — Memory usage and always-cold functions", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out, "fig09_memory_always_cold")


def fig10(R, out):
    r = R["spes"]; csr = r["cold"] / r["inv"]; t = r["type"]
    ks = [k for k in range(10) if (t == k).sum() > 0]
    vals = [csr[t == k].mean() for k in ks]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    bars = ax.bar([f"{TYPE_NAMES[k]}\n(n={(t == k).sum():,})" for k in ks], vals, color=SLOT[0], width=0.65)
    bar_labels(ax, bars, lambda v: f"{v:.3f}")
    ax.set_ylabel("Average cold-start rate"); ax.grid(axis="x", visible=False)
    plt.setp(ax.get_xticklabels(), fontsize=8.5, rotation=25, ha="right")
    ax.set_title("Fig. 10 — Average cold-start rate of each SPES type (test days)")
    save(fig, out, "fig10_csr_per_type")


def fig11(R, S, out):
    ms = _avail(R)
    w0 = S["spes"]["wmt"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.1))
    bars = a1.bar([l for _, l, _ in ms], [S[k]["wmt"] / w0 for k, _, _ in ms], color=[c for _, _, c in ms], width=0.65)
    bar_labels(a1, bars, lambda v: f"{v:.2f}×")
    a1.set_ylabel("Wasted memory time ÷ SPES"); a1.set_title("(a) Wasted memory time (WMT)", fontsize=11)
    bars = a2.bar([l for _, l, _ in ms], [S[k]["emcr"] for k, _, _ in ms], color=[c for _, _, c in ms], width=0.65)
    bar_labels(a2, bars, lambda v: f"{v:.1f}%")
    a2.set_ylabel("EMCR (%)"); a2.set_title("(b) Effective memory consumption ratio", fontsize=11)
    for a in (a1, a2):
        plt.setp(a.get_xticklabels(), rotation=25, ha="right"); a.grid(axis="x", visible=False)
    fig.suptitle("Fig. 11 — Wasted memory time and effective memory use", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out, "fig11_wmt_emcr")


def fig12(R, out):
    r = R["spes"]; t = r["type"]; ratio = r["waste"] / r["inv"]
    ks = [k for k in range(10) if k != 1 and (t == k).sum() > 0]
    fig, ax = plt.subplots(figsize=(9, 4))
    bp = ax.boxplot([ratio[t == k] for k in ks], patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color=INK))
    for p in bp["boxes"]: p.set_facecolor(SLOT[0]); p.set_edgecolor(INK2)
    ax.set_xticks(range(1, len(ks) + 1), [TYPE_NAMES[k] for k in ks], rotation=20, ha="right")
    ax.set_yscale("symlog", linthresh=0.1); ax.set_ylim(bottom=0); ax.set_ylabel("Wasted minutes per invocation")
    ax.set_title("Fig. 12 — Ratio of WMT for each function type (always-warm omitted)")
    save(fig, out, "fig12_wmt_ratio_per_type")


def _sweep(ax, R, names, labels, default, title):
    base = R[default]; m0 = np.mean(base["used_t"] + base["waste_t"])
    xs, ys = [], []
    for i, (n, lab) in enumerate(zip(names, labels)):
        r = R.get(n)
        if r is None: continue
        x = np.mean(r["used_t"] + r["waste_t"]) / m0; y = np.percentile(r["cold"] / r["inv"], 75)
        xs.append(x); ys.append(y)
        isdef = n == default
        ax.plot(x, y, marker="*" if isdef else "o", ms=15 if isdef else 8, color=SLOT[1] if isdef else SLOT[0],
                mec="white", mew=1.2, ls="none", zorder=3)
        ax.annotate(lab, (x, y), xytext=(0, 9), textcoords="offset points", fontsize=9, ha="center",
                    fontweight="bold" if isdef else "normal")
    if len(xs) >= 2:
        p = np.polyfit(xs, ys, 1); xx = np.linspace(min(xs), max(xs), 50)
        ax.plot(xx, np.polyval(p, xx), color=INK2, lw=1, ls="--", zorder=1)
    ax.set_xlabel("Memory usage ÷ default setting   (★ = default)"); ax.set_ylabel("75th-percentile cold-start rate")
    ax.set_title(title, fontsize=11)


def fig13(R, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    pw = [f"spes_prewarm{p}" if p != 2 else "spes" for p in range(1, 6)]
    _sweep(a1, R, pw, [f"{p}" for p in range(1, 6)], "spes", "(a) θ_prewarm = 1…5 minutes  (θ=0: see summary.md)")
    gu = ["spes"] + [f"spes_giveup{k}" for k in (2, 3, 4, 5)]
    _sweep(a2, R, gu, [f"×{k}" for k in (1, 2, 3, 4, 5)], "spes", "(b) θ_givenup multiplied by 1…5")
    fig.suptitle("Fig. 13 — Trading memory for cold starts", fontweight="bold", x=0.02, ha="left")
    for a in (a1, a2): a.margins(x=0.12, y=0.1)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out, "fig13_tradeoff")


def _ablation(R, S, out, variants, fname, title, bar_metric):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    present = [(n, l, SLOT[i]) for i, (n, l) in enumerate(variants) if R.get(n) is not None]
    styles = ["-", "--", ":"]
    for i, (n, l, c) in enumerate(present):
        csr = R[n]["cold"] / R[n]["inv"]
        cdf(a1, csr, color=c, ls=styles[i], label=f"{l}  (Q3 = {np.percentile(csr, 75):.3f})", zorder=3 - i)
    a1.axhline(0.75, color=INK2, lw=0.8, ls=":")
    a1.set_xlim(0, 1); a1.set_ylim(0.4, 1.01); a1.legend(loc="lower right", fontsize=9)
    a1.set_xlabel("Cold-start rate"); a1.set_ylabel("Fraction of functions (CDF)")
    a1.set_title("(a) Cold-start rate", fontsize=11)
    qs = [np.percentile(R[n]["cold"] / R[n]["inv"], [50, 75, 90]) for n, _, _ in present]
    if max(np.abs(q - qs[0]).max() for q in qs) < 0.01:
        a1.text(0.98, 0.44, "the curves nearly coincide", transform=a1.transAxes, ha="right", fontsize=9, color=INK2)
    key, ylab, fmt = bar_metric
    vals = [S[n][key] / (S["spes"][key] if key == "wmt" else 1) for n, _, _ in present]
    bars = a2.bar([l for _, l, _ in present], vals, color=[c for _, _, c in present], width=0.6)
    bar_labels(a2, bars, fmt); a2.set_ylabel(ylab); a2.grid(axis="x", visible=False)
    plt.setp(a2.get_xticklabels(), rotation=15, ha="right")
    a2.set_title(f"(b) {ylab}", fontsize=11)
    fig.suptitle(title, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out, fname)


def summary_md(S, R, out_path):
    rows = [("spes", "SPES")] + [(k, l) for k, l, _, _ in METHODS[1:]] + \
           [(f"spes_prewarm{p}", f"SPES θ_prewarm={p}") for p in (0, 1, 3, 4, 5)] + \
           [(f"spes_giveup{k}", f"SPES θ_givenup×{k}") for k in (2, 3, 4, 5)] + \
           [("spes_no_corr", "SPES w/o Corr"), ("spes_no_online_corr", "SPES w/o Online-Corr"),
            ("spes_no_forgetting", "SPES w/o forgetting"), ("spes_no_adjust", "SPES w/o adjusting")]
    L = ["| Run | Q3 CSR | Never-cold % | Always-cold % | Total cold starts | WMT (M fn-min) | Avg memory | EMCR % | Paper |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, lab in rows:
        if k not in S: continue
        s = S[k]; p = PAPER.get(k, {})
        ptxt = ", ".join(f"{a}={b}" for a, b in p.items()) if p else ""
        L.append(f"| {lab} | {s['q3_csr']:.4f} | {s['zero_cold_pct']:.2f} | {s['always_cold_pct']:.2f} | {s['total_cold']:,} | "
                 f"{s['wmt'] / 1e6:.2f} | {s['mem_avg']:,.0f} | {s['emcr']:.1f} | {ptxt} |")
    open(out_path, "w").write("\n".join(L) + "\n\nHA cold-start rates are per application (as in the paper). "
                              "Memory = functions loaded per test minute; the paper's 13,982 uses the notebook's end-of-minute count "
                              f"(ours: {S['spes'].get('mem_avg_notebook', float('nan')):,.0f}).\n")
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir"); ap.add_argument("results_dir"); ap.add_argument("figures_dir")
    a = ap.parse_args()
    os.makedirs(a.figures_dir, exist_ok=True)
    names = ["spes", "fixed", "hybrid_f", "hybrid_a", "defuse", "faascache", "spes_no_corr", "spes_no_online_corr",
             "spes_no_forgetting", "spes_no_adjust"] + [f"spes_prewarm{p}" for p in (0, 1, 3, 4, 5)] + [f"spes_giveup{k}" for k in (2, 3, 4, 5)]
    R = {n: load(a.results_dir, n) for n in names}
    R = {k: v for k, v in R.items() if v is not None}
    S = {k: summarize(v) for k, v in R.items()}
    tr = Trace(a.data_dir)
    fig03(tr, a.figures_dir); fig04(tr, a.figures_dir); fig05(tr, a.figures_dir)
    fig06(tr, R["spes"], a.figures_dir); fig07(tr, R["spes"], a.figures_dir)
    fig08(R, a.figures_dir); fig09(R, S, a.figures_dir); fig10(R, a.figures_dir); fig11(R, S, a.figures_dir)
    fig12(R, a.figures_dir); fig13(R, a.figures_dir)
    _ablation(R, S, a.figures_dir, [("spes", "SPES"), ("spes_no_corr", "w/o Corr"), ("spes_no_online_corr", "w/o Online-Corr")],
              "fig14_correlation_ablation", "Fig. 14 — Ablation: inter-function correlation",
              ("wmt", "WMT ÷ SPES", lambda v: f"{v:.2f}×"))
    _ablation(R, S, a.figures_dir, [("spes", "SPES"), ("spes_no_forgetting", "w/o forgetting"), ("spes_no_adjust", "w/o adjusting")],
              "fig15_adaptivity_ablation", "Fig. 15 — Ablation: adaptive strategies (forgetting, adjusting)",
              ("q3_csr", "75th-percentile cold-start rate", lambda v: f"{v:.3f}"))
    summary_md(S, R, os.path.join(a.figures_dir, "summary.md"))


if __name__ == "__main__":
    main()
