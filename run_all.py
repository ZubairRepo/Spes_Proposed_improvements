"""Run every experiment needed for the paper's figures and write results/*.npz + results/summary.json.

usage: python run_all.py <data_dir> <results_dir> [--workers 2] [--candidates compute|path/to/func_candi.json]

Experiments
  spes                          SPES with the notebook's settings          (Figs. 8-12)
  fixed hybrid_f hybrid_a defuse faascache   baselines                     (Figs. 8, 9, 11)
  spes_prewarm{0,1,3,4,5}       theta_prewarm sweep (2 = default)           (Fig. 13a)
  spes_giveup{2,3,4,5}          theta_givenup x2..x5                        (Fig. 13b)
  spes_no_corr, spes_no_online_corr                                         (Fig. 14)
  spes_no_forgetting, spes_no_adjust                                        (Fig. 15)
"""
import os, sys, json, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor

_TR = None


def _init(data_dir):
    global _TR
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import Trace
    _TR = Trace(data_dir)


def _save(path, res):
    np.savez_compressed(path, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                        meta=json.dumps({k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
                                         for k, v in res.items() if not isinstance(v, np.ndarray)}))


def _job(name, kind, params, out_dir, cache_dir):
    import spes, baselines
    from common import summarize
    t0 = time.time()
    if kind == "spes":
        res = spes.run(_TR, params, cache_dir, log=lambda *a: print(f"[{name}]", *a, flush=True))
    elif kind == "fixed": res = baselines.run_fixed(_TR, 10)
    elif kind == "hybrid_f": res = baselines.run_hybrid(_TR, False)
    elif kind == "hybrid_a": res = baselines.run_hybrid(_TR, True)
    elif kind == "defuse": res = baselines.run_defuse(_TR)
    elif kind == "faascache": res = baselines.run_faascache(_TR, params["cap"])
    res["wall_seconds"] = time.time() - t0
    _save(os.path.join(out_dir, f"{name}.npz"), res)
    s = summarize(res)
    print(f"[{name}] done in {time.time() - t0:.0f}s: never-cold {s['zero_cold_pct']:.2f}%  Q3-CSR {s['q3_csr']:.4f}  "
          f"WMT {s['wmt'] / 1e6:.2f}M  memory {s['mem_avg']:.0f}", flush=True)
    return name, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir"); ap.add_argument("results_dir")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--candidates", default="compute")
    ap.add_argument("--only", default="", help="comma-separated experiment names to (re)run")
    a = ap.parse_args()
    os.makedirs(a.results_dir, exist_ok=True)
    cache = os.path.join(a.results_dir, "cache")
    base = dict(candidates=a.candidates)
    jobs = [("spes", "spes", dict(base))]
    jobs += [(f"spes_prewarm{p}", "spes", dict(base, prewarm=p)) for p in (0, 1, 3, 4, 5)]
    jobs += [(f"spes_giveup{k}", "spes", dict(base, giveup_scale=k)) for k in (2, 3, 4, 5)]
    jobs += [("spes_no_corr", "spes", dict(base, use_corr=False, use_corr_sim=False)),
             ("spes_no_online_corr", "spes", dict(base, online_corr="off")),
             ("spes_no_forgetting", "spes", dict(base, forgetting=False)),
             ("spes_no_adjust", "spes", dict(base, adjust=False))]
    jobs += [(k, k, {}) for k in ("fixed", "hybrid_f", "hybrid_a", "defuse")]
    only = set(filter(None, a.only.split(",")))
    if only:
        jobs = [j for j in jobs if j[0] in only]
    summary_path = os.path.join(a.results_dir, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    # categorization/assignment caches are built once in the main process, then shared by the parallel runs
    global _TR
    _init(a.data_dir)
    import spes, pickle, gc
    needed = {}
    for n, k, p in jobs:
        if k == "spes":
            cfg = dict(spes.DEFAULT, **p)
            key = f"forget{int(cfg['forgetting'])}_corr{int(cfg['use_corr'])}_{os.path.basename(str(cfg['candidates']))}"
            needed[key] = cfg
    for key, cfg in needed.items():
        path = os.path.join(cache, f"spes_cat_{key}.pkl")
        if os.path.exists(path):
            continue
        print(f"building categorization cache {key} ...", flush=True)
        os.makedirs(cache, exist_ok=True)
        typ, forget, before = spes.categorize(_TR, cfg["forgetting"])
        candi = spes.find_candidates(_TR, typ, forget) if cfg["candidates"] == "compute" else spes.load_candidates_json(_TR, cfg["candidates"])
        typ_final = spes.assign(_TR, typ, forget, candi, cfg["use_corr"])
        pickle.dump((typ, forget, before, candi, typ_final), open(path, "wb"))
    _TR = None; gc.collect()
    # SPES runs in parallel; the app/set-level baselines need ~3 GB each, so they run one at a time
    for group, workers in (([j for j in jobs if j[1] == "spes"], a.workers), ([j for j in jobs if j[1] != "spes"], 1)):
        if not group:
            continue
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(a.data_dir,)) as ex:
            futs = [ex.submit(_job, n, k, p, a.results_dir, cache) for n, k, p in group]
            for f in futs:
                n, s = f.result(); summary[n] = s
                json.dump(summary, open(summary_path, "w"), indent=1)
    # FaaSCache needs SPES's peak memory as its cap (paper, Sec. V-A)
    if not only or "faascache" in only:
        _init(a.data_dir)
        z = np.load(os.path.join(a.results_dir, "spes.npz"))
        cap = int(z["mem_end_t"].max())
        n, s = _job("faascache", "faascache", dict(cap=cap), a.results_dir, cache)
        summary[n] = s
        json.dump(summary, open(summary_path, "w"), indent=1)
    print("all done ->", summary_path)


if __name__ == "__main__":
    main()
