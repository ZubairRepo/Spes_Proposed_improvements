"""Figures for the proposed policy (after run_all.py and run_proposed.py).

usage: python plots_proposed.py <results_dir> <figures_dir>

  figP1_tradeoff        never-cold % and Q3-CSR vs memory: fixed keep-alive, SPES, three versions of the proposal
  figP2_csr_cdf         CDF of cold-start rate at SPES's memory: proposed vs SPES vs baselines
  figP3_calibration     raw chance vs observed call rate (why calibration is needed)
  figP4_gain_breakdown  where the gain comes from, at SPES's memory
  figP5_matched         same memory -> fewer cold starts; same quality -> less memory
  summary_proposed.md   all numbers
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import summarize
from plots import INK, INK2, SLOT, load, save, bar_labels, cdf
import proposed

C_SPES, C_PROP, C_SIG, C_RULE, C_HF, C_FIX = SLOT[0], SLOT[1], SLOT[2], SLOT[3], SLOT[4], SLOT[5]
VLAB = {"rule": "Probability rule only", "signals": "+ clock & sibling", "full": "+ calibration (proposed)"}
VCOL = {"rule": C_RULE, "signals": C_SIG, "full": C_PROP}
VMK = {"rule": "s", "signals": "^", "full": "o"}


def at_memory(curve, mem, key, log=False):
    c = sorted(curve, key=lambda s: s["mem_avg"])
    x = np.array([s["mem_avg"] for s in c]); y = np.array([s[key] for s in c], float)
    if log: return float(np.exp(np.interp(mem, x, np.log(np.maximum(y, 1e-9)))))
    return float(np.interp(mem, x, y))


def figP1(W, out):
    sp = W["spes"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    fx = sorted(W["fixed"], key=lambda s: s["mem_avg"])
    for ax, key, lab in ((a1, "zero_cold_pct", "Functions never cold (%)  ↑ better"),
                         (a2, "q3_csr", "75th-percentile cold-start rate  ↓ better")):
        ax.plot([s["mem_avg"] / 1e3 for s in fx], [s[key] for s in fx], color=C_FIX, marker="D", ms=5, lw=1.8,
                ls=(0, (1, 1)), label="Fixed keep-alive (1–60 min)")
        for v in ("rule", "signals", "full"):
            c = sorted(W[v], key=lambda s: s["mem_avg"])
            ax.plot([s["mem_avg"] / 1e3 for s in c], [s[key] for s in c], color=VCOL[v], marker=VMK[v], ms=6,
                    lw=2.2 if v == "full" else 1.6, ls="-" if v == "full" else "--", label=VLAB[v])
        ax.plot(sp["mem_avg"] / 1e3, sp[key], marker="*", ms=17, color=C_SPES, mec="white", mew=1.2, ls="none", label="SPES", zorder=5)
        ax.axvline(sp["mem_avg"] / 1e3, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("Average functions kept in memory (thousands)  ← cheaper"); ax.set_ylabel(lab)
    a1.set_ylim(0, 85); a2.set_ylim(0, 1.02)
    for ax in (a1, a2): ax.set_xlim(7, 26)
    a1.legend(loc="lower right", fontsize=9)
    a1.set_title("(a) Never-cold functions", fontsize=11); a2.set_title("(b) Cold-start rate of the 75th-percentile function", fontsize=11)
    fig.suptitle("Fig. P1 — Memory vs. cold starts: every setting of each policy's knob (dotted line = SPES memory)",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, out, "figP1_tradeoff")


def figP2(R, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    series = [("proposed_same_memory", "Proposed (same memory as SPES)", C_PROP, "-"), ("spes", "SPES", C_SPES, "--"),
              ("hybrid_f", "Hybrid-Func (HF)", C_HF, "-."), ("fixed", "Fixed keep-alive", C_FIX, (0, (1, 1)))]
    for k, lab, col, ls in series:
        r = R.get(k)
        if r is None: continue
        csr = r["cold"] / r["inv"]; m = np.mean(r["used_t"] + r["waste_t"])
        cdf(ax, csr, color=col, ls=ls, label=f"{lab}  Q3 = {np.percentile(csr, 75):.3f}, memory {m:,.0f}")
    ax.axhline(0.75, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlabel("Cold-start rate (cold starts ÷ invocations)"); ax.set_ylabel("Fraction of functions (CDF)")
    ax.set_title("Fig. P2 — Cold-start rate per function, test days 13–14")
    save(fig, out, "figP2_csr_cdf")


def figP3(calib, out):
    z = np.load(calib); rel = proposed.reliability(z["K"], z["N"])
    rel = rel[rel[:, 0] > 0]
    x = np.sqrt(rel[:, 1] * rel[:, 2]); y = rel[:, 4]
    fig, ax = plt.subplots(figsize=(6.6, 5))
    lim = [1e-6, 1.2]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--", label="perfectly calibrated")
    ax.plot(x, y, marker="o", ms=6, lw=2, color=C_PROP, label="observed on held-out days 11–12")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(1e-5, 1.2)
    i = np.argmin(np.abs(np.log10(x) + 4))
    ax.annotate(f"said ≈ {x[i] * 1e4:.1f} in 10,000\nreally {y[i] * 1e4:.0f} in 10,000", (x[i], y[i]), xytext=(-10, 30),
                textcoords="offset points", ha="right", fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("Raw idle-time chance of a call in the next minute"); ax.set_ylabel("Observed call rate")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Fig. P3 — Raw chances for rare calls are far too low")
    save(fig, out, "figP3_calibration")


def figP4(W, R, out):
    sp = W["spes"]; mem = sp["mem_avg"]
    labs = ["SPES"] + [VLAB[v] for v in ("rule", "signals", "full")]
    cols = [C_SPES] + [VCOL[v] for v in ("rule", "signals", "full")]
    nc = [sp["zero_cold_pct"]] + [at_memory(W[v], mem, "zero_cold_pct") for v in ("rule", "signals", "full")]
    tc = [sp["total_cold"]] + [at_memory(W[v], mem, "total_cold", log=True) for v in ("rule", "signals", "full")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    b = a1.bar(labs, nc, color=cols, width=0.62); bar_labels(a1, b, lambda v: f"{v:.1f}%")
    a1.set_ylabel("Functions never cold (%)"); a1.set_ylim(0, max(nc) * 1.18)
    b = a2.bar(labs, tc, color=cols, width=0.62); bar_labels(a2, b, lambda v: f"{v:,.0f}")
    a2.set_ylabel("Total cold starts, days 13–14"); a2.set_ylim(0, max(tc) * 1.18)
    for a in (a1, a2):
        plt.setp(a.get_xticklabels(), rotation=15, ha="right"); a.grid(axis="x", visible=False)
    a1.set_title("(a) Never-cold functions  ↑", fontsize=11); a2.set_title("(b) Total cold starts  ↓", fontsize=11)
    fig.suptitle(f"Fig. P4 — Where the gain comes from (each version read off its curve at SPES's memory, {mem:,.0f} functions)",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, out, "figP4_gain_breakdown")


def figP5(S, out):
    sp, pm, pq, fx = S["spes"], S["proposed_same_memory"], S["proposed_same_quality"], S["fixed"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    labs = ["SPES", "Proposed"]; cols = [C_SPES, C_PROP]
    panels = [(axes[0], "zero_cold_pct", "Never-cold functions (%) ↑", [sp, pm], lambda v: f"{v:.1f}%"),
              (axes[1], "q3_csr", "75th-pct cold-start rate ↓", [sp, pm], lambda v: f"{v:.3f}"),
              (axes[2], "total_cold", "Total cold starts ↓", [sp, pm], lambda v: f"{v:,.0f}")]
    for ax, k, t, ss, fmt in panels:
        b = ax.bar(labs, [s[k] for s in ss], color=cols, width=0.6); bar_labels(ax, b, fmt)
        ax.set_title(t, fontsize=10.5); ax.grid(axis="x", visible=False); ax.set_ylim(0, max(s[k] for s in ss) * 1.2)
    ax = axes[3]
    labs3 = ["Fixed 10 min", "SPES", "Proposed"]
    b = ax.bar(labs3, [fx["wmt"] / 1e6, sp["wmt"] / 1e6, pq["wmt"] / 1e6], color=[C_FIX, C_SPES, C_PROP], width=0.6)
    bar_labels(ax, b, lambda v: f"{v:.1f}M"); ax.set_ylim(0, fx["wmt"] / 1e6 * 1.2)
    ax.set_title("Wasted memory (M function-min) ↓", fontsize=10.5); ax.grid(axis="x", visible=False)
    fig.text(0.02, 0.92, f"Same memory as SPES ({pm['mem_avg']:,.0f} vs {sp['mem_avg']:,.0f} functions)", fontsize=11, fontweight="bold")
    fig.text(0.765, 0.92, f"Same quality as SPES\n({pq['mem_avg']:,.0f} vs {sp['mem_avg']:,.0f} functions in memory)", fontsize=11, fontweight="bold", va="bottom")
    fig.suptitle("Fig. P5 — The proposal at two operating points", fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, out, "figP5_matched")


def summary(S, W, out_path):
    rows = [("fixed", "Fixed keep-alive 10 min"), ("hybrid_f", "Hybrid-Func (HF)"), ("spes", "SPES (port of the authors' code)"),
            ("proposed_same_memory", "**Proposed, same memory as SPES**"), ("proposed_same_quality", "**Proposed, same quality as SPES** (μ=10)")]
    L = ["| Policy | λ | Avg memory | WMT (M) | Never-cold % | Q3 CSR | Always-cold % | Total cold starts |", "|---|---|---|---|---|---|---|---|"]
    for k, lab in rows:
        if k not in S: continue
        s = S[k]
        L.append(f"| {lab} | {s.get('lam', ''):} | {s['mem_avg']:,.0f} | {s['wmt'] / 1e6:.2f} | {s['zero_cold_pct']:.2f} | "
                 f"{s['q3_csr']:.4f} | {s['always_cold_pct']:.2f} | {s['total_cold']:,} |")
    sp, pm, pq = S["spes"], S.get("proposed_same_memory"), S.get("proposed_same_quality")
    if pm and pq:
        L += ["", f"* Same memory: never-cold {sp['zero_cold_pct']:.1f}% → {pm['zero_cold_pct']:.1f}%, "
              f"Q3-CSR {sp['q3_csr']:.3f} → {pm['q3_csr']:.3f} ({(1 - pm['q3_csr'] / sp['q3_csr']) * 100:.0f}% lower), "
              f"total cold starts {(1 - pm['total_cold'] / sp['total_cold']) * 100:.0f}% fewer.",
              f"* Same quality: average memory {(1 - pq['mem_avg'] / sp['mem_avg']) * 100:.0f}% lower, "
              f"wasted memory {(1 - pq['wmt'] / sp['wmt']) * 100:.0f}% lower than SPES."]
    L += ["", "Threshold sweeps (each row: one λ):", "", "| Version | λ | Avg memory | Never-cold % | Q3 CSR | Total cold starts |", "|---|---|---|---|---|---|"]
    for v in ("rule", "signals", "full"):
        for s in sorted(W.get(v, []), key=lambda s: s["lam"]):
            L.append(f"| {VLAB[v]} | {s['lam']:g} | {s['mem_avg']:,.0f} | {s['zero_cold_pct']:.2f} | {s['q3_csr']:.4f} | {s['total_cold']:,} |")
    open(out_path, "w").write("\n".join(L) + "\n")
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir"); ap.add_argument("figures_dir")
    a = ap.parse_args()
    os.makedirs(a.figures_dir, exist_ok=True)
    W = json.load(open(os.path.join(a.results_dir, "proposed_sweep.json")))
    R = {n: load(a.results_dir, n) for n in ("spes", "fixed", "hybrid_f", "proposed_same_memory", "proposed_same_quality")}
    R = {k: v for k, v in R.items() if v is not None}
    S = {k: summarize(v) for k, v in R.items()}
    for k in ("proposed_same_memory", "proposed_same_quality"):
        if k in S: S[k]["lam"] = f"{R[k]['lam']:.4g}"
    figP1(W, a.figures_dir); figP2(R, a.figures_dir); figP3(os.path.join(a.results_dir, "proposed_calib.npz"), a.figures_dir)
    figP4(W, R, a.figures_dir)
    if "proposed_same_memory" in S and "proposed_same_quality" in S: figP5(S, a.figures_dir)
    summary(S, W, os.path.join(a.figures_dir, "summary_proposed.md"))


if __name__ == "__main__":
    main()
