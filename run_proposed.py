"""Experiments for the proposed policy (run after run_all.py, which provides results/spes.npz).

usage: python run_proposed.py <data_dir> <results_dir> [--workers 2]

  1. calibration table (days 1-10 -> 11-12)                      results/proposed_calib.npz
  2. threshold sweeps for three versions (Figs. P1, P4)           results/proposed_sweep.json
       rule      idle-time chance only, uncalibrated
       signals   + clock + sibling (+ timing jitter), uncalibrated
       full      + calibration  (the proposed policy)
     and fixed keep-alive K = 1..60 for reference
  3. two operating points of the full policy, found by bisection on lam:
       proposed_same_memory   largest memory not above SPES's average memory
       proposed_same_quality  least memory that is at least as good as SPES on never-cold %, Q3-CSR and always-cold %
                              (uses the rare-function boost mu=10)                       results/proposed_*.npz
"""
import os, sys, json, argparse, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

LAMS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
VARIANTS = {"rule": dict(flags=dict(use_prof=False, use_sib=False, jitter=False), calibrated=False),
            "signals": dict(flags=dict(use_prof=True, use_sib=True, jitter=True), calibrated=False),
            "full": dict(flags=dict(use_prof=True, use_sib=True, jitter=True), calibrated=True)}
_W = {}


def _init(data_dir, calib):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Trace
    _W["tr"] = Trace(data_dir); _W["calib"] = calib; _W["models"] = {}


def _model(variant):
    import proposed
    from common import TR, END
    key = json.dumps(VARIANTS[variant]["flags"], sort_keys=True)
    if key not in _W["models"]:
        _W["models"].clear()
        m = proposed.Model(_W["tr"], TR, END, **VARIANTS[variant]["flags"], log=lambda *a: None)
        z = np.load(_W["calib"]); m.tabK[:], m.tabN[:] = z["K"], z["N"]
        _W["models"][key] = m
    return _W["models"][key]


def _sweep_job(variant, lam):
    from common import summarize
    r = _model(variant).run(lam, 0.0, VARIANTS[variant]["calibrated"])
    s = summarize(r); s["lam"] = lam
    print(f"[{variant} lam={lam:g}] never-cold {s['zero_cold_pct']:.2f}%  Q3 {s['q3_csr']:.4f}  memory {s['mem_avg']:.0f}", flush=True)
    return variant, s


def _bisect_job(name, target, lo, hi, mu, out_dir, iters):
    """Search log(lam) in [lo, hi]; memory falls as lam grows. Keep the best lam that satisfies the target."""
    from common import summarize
    m = _model("full")
    def ok(s):
        if name == "proposed_same_memory":
            return s["mem_avg"] <= target["mem_avg"]
        return (s["zero_cold_pct"] >= target["zero_cold_pct"] and s["q3_csr"] <= target["q3_csr"]
                and s["always_cold_pct"] <= target["always_cold_pct"])
    best = None
    for _ in range(iters):
        lam = float(np.sqrt(lo * hi))
        r = m.run(lam, mu, True); s = summarize(r)
        good = ok(s)
        print(f"[{name}] lam={lam:.5g} mu={mu:g} -> memory {s['mem_avg']:.0f}, never-cold {s['zero_cold_pct']:.2f}%, "
              f"Q3 {s['q3_csr']:.4f}, always-cold {s['always_cold_pct']:.2f}%  {'ok' if good else ''}", flush=True)
        if name == "proposed_same_memory":
            if good: best = (lam, r, s); hi = lam      # try more memory (smaller lam)
            else: lo = lam
        else:
            if good: best = (lam, r, s); lo = lam      # try less memory (larger lam)
            else: hi = lam
    lam, r, s = best
    np.savez_compressed(os.path.join(out_dir, f"{name}.npz"), **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
                        meta=json.dumps({k: v for k, v in r.items() if not isinstance(v, np.ndarray)}))
    return name, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir"); ap.add_argument("results_dir"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Trace, summarize
    import proposed, baselines, spes
    R = a.results_dir; os.makedirs(R, exist_ok=True)
    calib = os.path.join(R, "proposed_calib.npz")
    tr = Trace(a.data_dir)
    if not os.path.exists(calib):
        K, N = proposed.fit_calibration(tr); np.savez(calib, K=K, N=N)
    # SPES reference
    sp = os.path.join(R, "spes.npz")
    if os.path.exists(sp):
        z = np.load(sp); spes_res = {k: z[k] for k in z.files if k != "meta"}; spes_res.update(json.loads(str(z["meta"])))
    else:
        spes_res = spes.run(tr, {}, os.path.join(R, "cache"))
    target = summarize(spes_res)
    print(f"SPES target: memory {target['mem_avg']:.0f}, never-cold {target['zero_cold_pct']:.2f}%, Q3 {target['q3_csr']:.4f}, "
          f"always-cold {target['always_cold_pct']:.2f}%", flush=True)
    sweep = {"spes": target, "fixed": []}
    for K in (1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60):
        s = summarize(baselines.run_fixed(tr, K)); s["K"] = K; sweep["fixed"].append(s)
    del tr
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.data_dir, calib)) as ex:
        futs = [ex.submit(_sweep_job, v, lam) for v in VARIANTS for lam in LAMS]
        for f in futs:
            v, s = f.result(); sweep.setdefault(v, []).append(s)
        json.dump(sweep, open(os.path.join(R, "proposed_sweep.json"), "w"), indent=1)
        # bracket each operating point from the sweep of the full policy, then bisect
        full = sorted(sweep["full"], key=lambda s: s["lam"])
        lams = [s["lam"] for s in full]
        i = max([k for k, s in enumerate(full) if s["mem_avg"] > target["mem_avg"]] + [0])
        b1 = (lams[i], lams[min(i + 1, len(lams) - 1)] * 1.0001)
        jobs = [ex.submit(_bisect_job, "proposed_same_memory", target, b1[0], b1[1], 0.0, R, 7),
                ex.submit(_bisect_job, "proposed_same_quality", target, 1e-4, 0.3, 10.0, R, 10)]
        for f in jobs:
            n, s = f.result(); sweep[n] = s
    json.dump(sweep, open(os.path.join(R, "proposed_sweep.json"), "w"), indent=1)
    print("done ->", os.path.join(R, "proposed_sweep.json"))


if __name__ == "__main__":
    main()
