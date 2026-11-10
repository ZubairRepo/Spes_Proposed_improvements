"""Baselines compared in the SPES paper, evaluated with the same metrics as SPES.

  fixed      fixed keep-alive (10 min)
  hybrid_f   Hybrid histogram policy (Shahrad et al., ATC'20) per function       ("HF" in SPES)
  hybrid_a   Hybrid histogram policy per application (the original granularity)  ("HA" in SPES)
  defuse     dependency sets (strong: frequent co-invocation; weak: PPMI) + histogram windows (Shen et al., ICDCS'21)
  faascache  Greedy-Dual-Size-Frequency keep-alive under a memory cap = SPES's peak memory (Fuerst & Sharma, ASPLOS'21)

The original authors did not release code for these; SPES also re-implemented them. Parameters follow
the original papers where they are given; where a paper is silent, the choice is stated below and in README.md.
All policies replay days 1-14 (days 1-12 warm up the histograms) and are scored on days 13-14 only.
"""
import os, sys, json, time, argparse
import numpy as np
from numba import njit
from common import Trace, summarize, DAY, TR, END, T_TEST

NONE = -(1 << 40)
RANGE = 240                  # histogram range: 4 hours of 1-minute bins (Shahrad et al.)


# ------------------------------------------------------------------ Fenwick tree helpers (percentiles in O(log n))
@njit(cache=True)
def _bit_add(bit, base, i, v):
    i += 1
    while i <= RANGE:
        bit[base + i - 1] += v; i += i & (-i)


@njit(cache=True)
def _bit_find(bit, base, target):
    """smallest bin b with cumulative count >= target (target >= 1)."""
    pos = 0; rem = target; step = 128
    while step > 0:
        nxt = pos + step
        if nxt <= RANGE and bit[base + nxt - 1] < rem:
            pos = nxt; rem -= bit[base + nxt - 1]
        step >>= 1
    return pos  # 0-based bin


# ------------------------------------------------------------------ unit-based keep-alive / pre-warm engine
@njit(cache=True)
def _run_units(mode, nU, u_ptr, u_min, identity, e_ptr, e_f, e_c, size, nF, K, cv_th, p_lo, p_hi, margin, fallback, oob_frac):
    """mode 0 = fixed keep-alive K; 1 = Hybrid (Shahrad); 2 = Defuse windows (P5/P95, fixed-10 fallback).

    After a unit's call at minute t it is loaded during t+k for lo < k <= hi. Next call after IT idle minutes is warm iff lo < IT+1 <= hi.
    u_min[u_ptr[u]:u_ptr[u+1]]: busy minutes of unit u; for event q the called functions are e_f[e_ptr[q]:e_ptr[q+1]]
    (identity=True: units are functions, event q is trace entry q with count e_c[q]).
    """
    cold = np.zeros(nF, np.int64); inv = np.zeros(nF, np.int64); invm = np.zeros(nF, np.int64)
    ucold = np.zeros(nU, np.int64); uinv = np.zeros(nU, np.int64)
    diff = np.zeros(T_TEST + 1, np.int64)       # loaded function-count per test minute (difference array)
    used_t = np.zeros(T_TEST, np.int64)
    nb = nU if mode > 0 else 1
    bit = np.zeros(nb * RANGE, np.int64); bins = np.zeros(nb * RANGE, np.int64)
    ring = np.zeros((nb, 10));
    n_fallback_units = 0
    for u in range(nU):
        prev = NONE; lo = 0; hi = fallback if mode > 0 else K
        n_in = 0; n_oob = 0; sumsq = 0.0; rn = 0; rpos = 0
        base = u * RANGE if mode > 0 else 0
        for q in range(u_ptr[u], u_ptr[u + 1]):
            m = u_min[q]
            if prev == NONE:
                loaded = False
            else:
                d = m - prev
                loaded = lo < d and d <= hi
                a = prev + lo + 1; b = prev + min(hi, d - 1)          # idle loaded minutes
                a = max(a, TR);
                if b >= a:
                    diff[a - TR] += size[u]; diff[b - TR + 1] -= size[u]
            if m >= TR:
                diff[m - TR] += size[u]; diff[m - TR + 1] -= size[u]
                if not loaded: ucold[u] += 1
                if identity:
                    inv[u] += e_c[q]; invm[u] += 1; uinv[u] += e_c[q]; used_t[m - TR] += 1
                    if not loaded: cold[u] += 1
                else:
                    for e in range(e_ptr[q], e_ptr[q + 1]):
                        f = e_f[e]
                        inv[f] += e_c[e]; invm[f] += 1; uinv[u] += e_c[e]; used_t[m - TR] += 1
                        if not loaded: cold[f] += 1
            # update the policy with the new idle time and choose the next window
            if mode == 0:
                lo = 0; hi = K
            elif prev != NONE:
                it = m - prev - 1
                ring[u, rpos] = it; rpos = (rpos + 1) % 10; rn = min(rn + 1, 10)
                if it < RANGE:
                    c0 = bins[base + it]; bins[base + it] = c0 + 1
                    sumsq += 2 * c0 + 1; n_in += 1
                    _bit_add(bit, base, it, 1)
                else:
                    n_oob += 1
                use_ts = mode == 1 and n_oob >= 2 and n_oob > oob_frac * (n_in + n_oob)
                mean = n_in / RANGE
                rep = False
                if n_in > 0:
                    var = sumsq / RANGE - mean * mean
                    rep = np.sqrt(max(var, 0.0)) / mean > cv_th
                if mode == 2 and n_oob > oob_frac * (n_in + n_oob):
                    rep = False
                if use_ts:                                   # time-series mode (ARIMA in the paper; mean of last 10 ITs here)
                    pred = 0.0
                    for r in range(rn): pred += ring[u, r]
                    pred /= rn
                    lo = int(np.floor(0.85 * pred)); hi = lo + max(1, int(np.ceil(0.30 * pred))) + 1
                elif rep:
                    head = _bit_find(bit, base, max(1, int(np.ceil(p_lo / 100.0 * n_in))))
                    tail = _bit_find(bit, base, max(1, int(np.ceil(p_hi / 100.0 * n_in))))
                    lo = int(np.floor(head * (1 - margin)))
                    hi = int(np.ceil(tail * (1 + margin))) + 1
                    if hi <= lo: hi = lo + 1
                else:
                    lo = 0; hi = fallback
            prev = m
        if prev != NONE:                                     # loaded after the last call
            a = max(prev + lo + 1, TR); b = min(prev + hi, END - 1)
            if b >= a:
                diff[a - TR] += size[u]; diff[b - TR + 1] -= size[u]
            if mode > 0 and lo == 0 and hi == fallback:
                n_fallback_units += 1
    mem_t = np.cumsum(diff)[:T_TEST]
    return cold, inv, invm, ucold, uinv, mem_t, used_t, n_fallback_units


@njit(cache=True)
def _group_events(ip, M, C, mem_ptr, mem_f, nU):
    """Merge the call minutes of each unit's member functions into unit events (memory-lean, int32)."""
    nnz = len(M)
    u_min = np.empty(nnz, np.int32); e_ptr = np.empty(nnz + 1, np.int64); e_f = np.empty(nnz, np.int32); e_c = np.empty(nnz, np.int32)
    u_ptr = np.zeros(nU + 1, np.int64)
    ne = 0; nq = 0
    for u in range(nU):
        u_ptr[u] = nq
        tot = 0
        for k in range(mem_ptr[u], mem_ptr[u + 1]):
            f = mem_f[k]; tot += ip[f + 1] - ip[f]
        mm = np.empty(tot, np.int64); ff = np.empty(tot, np.int32); cc = np.empty(tot, np.int32); i = 0
        for k in range(mem_ptr[u], mem_ptr[u + 1]):
            f = mem_f[k]
            for q in range(ip[f], ip[f + 1]):
                mm[i] = M[q]; ff[i] = f; cc[i] = C[q]; i += 1
        o = np.argsort(mm, kind="mergesort")
        prev = -1
        for i in range(tot):
            j = o[i]
            if mm[j] != prev:
                u_min[nq] = mm[j]; e_ptr[nq] = ne; nq += 1; prev = mm[j]
            e_f[ne] = ff[j]; e_c[ne] = cc[j]; ne += 1
    u_ptr[nU] = nq; e_ptr[nq] = ne
    return u_ptr, u_min[:nq], e_ptr[:nq + 1], e_f, e_c


def _units_from_groups(tr, unit_of):
    """Unit event lists for a function -> unit map."""
    nU = int(unit_of.max()) + 1
    o = np.argsort(unit_of, kind="stable").astype(np.int64)
    mem_ptr = np.searchsorted(unit_of[o], np.arange(nU + 1)).astype(np.int64)
    u_ptr, u_min, e_ptr, e_f, e_c = _group_events(tr.ip, tr.M, tr.C, mem_ptr, o, nU)
    size = np.bincount(unit_of, minlength=nU).astype(np.int64)
    return nU, u_ptr, u_min, e_ptr, e_f, e_c, size


def _identity_units(tr):
    one = np.zeros(1, np.int64)
    return tr.nF, tr.ip, tr.M, one, one.astype(np.int32), tr.C, np.ones(tr.nF, np.int64)


def _pack(tr, out, unit_of=None, app_level=False):
    cold, inv, invm, ucold, uinv, mem_t, used_t, nfb = out
    te = tr.in_test
    res = dict(cold=cold[te], inv=inv[te], invm=invm[te], func_idx=np.where(te)[0], used_t=used_t,
               waste_t=mem_t - used_t, wmt=int((mem_t - used_t).sum()), fallback_units=int(nfb))
    if app_level:                       # SPES reports HA's cold-start rate per application
        sel = uinv > 0
        res["cold"], res["inv"] = ucold[sel], uinv[sel]
        res["func_level_cold"], res["func_level_inv"] = cold[te], inv[te]
    return res


def run_fixed(tr, K=10):
    nU, u_ptr, um, e_ptr, ef, ec, size = _identity_units(tr)
    out = _run_units(0, nU, u_ptr, um, True, e_ptr, ef, ec, size, tr.nF, K, 2.0, 5.0, 99.0, 0.1, K, 0.5)
    return _pack(tr, out)


def run_hybrid(tr, app_level=False, cv_th=2.0, fallback=RANGE):
    unit_of = tr.app if app_level else np.arange(tr.nF)
    nU, u_ptr, um, e_ptr, ef, ec, size = _units_from_groups(tr, unit_of) if app_level else _identity_units(tr)
    out = _run_units(1, nU, u_ptr, um, not app_level, e_ptr, ef, ec, size, tr.nF, 10, cv_th, 5.0, 99.0, 0.10, fallback, 0.5)
    return _pack(tr, out, unit_of, app_level)


# ------------------------------------------------------------------ Defuse: dependency mining on days 1-12
@njit(cache=True)
def _it_stats(ip, M, cv_th):
    """predictable = training idle-time histogram is representative (CV of 4h-histogram bins > cv_th, >= 3 ITs, <= 50% beyond 4h)."""
    nF = len(ip) - 1
    pred = np.zeros(nF, np.bool_); nbusy = np.zeros(nF, np.int64)
    bins = np.zeros(RANGE, np.int64)
    for f in range(nF):
        bins[:] = 0; n_in = 0; n_oob = 0; prev = NONE
        for k in range(ip[f], ip[f + 1]):
            m = M[k]
            if m >= TR: break
            nbusy[f] += 1
            if prev != NONE:
                it = m - prev - 1
                if it < RANGE:
                    bins[it] += 1; n_in += 1
                else:
                    n_oob += 1
            prev = m
        if n_in >= 3 and n_oob <= 0.5 * (n_in + n_oob):
            mean = n_in / RANGE; var = 0.0
            for b in range(RANGE): var += (bins[b] - mean) ** 2
            pred[f] = np.sqrt(var / RANGE) / mean > cv_th
    return pred, nbusy


@njit(cache=True)
def _co(ip, M, a, b):
    """# training minutes in which both a and b were called."""
    ma = M[ip[a]:ip[a + 1]]; mb = M[ip[b]:ip[b + 1]]
    ma = ma[:np.searchsorted(ma, TR)]; mb = mb[:np.searchsorted(mb, TR)]
    if len(ma) > len(mb): ma, mb = mb, ma
    c = 0
    for t in ma:
        j = np.searchsorted(mb, t)
        if j < len(mb) and mb[j] == t: c += 1
    return c


@njit(cache=True)
def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x


@njit(cache=True)
def _mine(ip, M, g_ptr, g_f, pred, nbusy, conf, parent):
    n_strong = 0; n_weak = 0
    for gi in range(len(g_ptr) - 1):
        fs = g_f[g_ptr[gi]:g_ptr[gi + 1]]
        n = len(fs)
        for i in range(n):
            a = fs[i]
            if nbusy[a] == 0: continue
            best = -1; best_ppmi = 0.0
            for j in range(n):
                b = fs[j]
                if a == b or nbusy[b] == 0: continue
                if pred[a] and pred[b] and j > i:                     # strong: frequent co-invocation of predictable functions
                    c = _co(ip, M, a, b)
                    if c >= 3 and c >= conf * min(nbusy[a], nbusy[b]):
                        ra, rb = _find(parent, a), _find(parent, b)
                        if ra != rb: parent[ra] = rb
                        n_strong += 1
                elif (not pred[a]) and pred[b]:                      # weak: unpredictable a -> predictable b (top-1 PPMI)
                    c = _co(ip, M, a, b)
                    if c >= 2:
                        ppmi = np.log2(c * TR / (nbusy[a] * nbusy[b]))
                        if ppmi > best_ppmi:
                            best_ppmi = ppmi; best = b
            if best >= 0:
                ra, rb = _find(parent, a), _find(parent, best)
                if ra != rb: parent[ra] = rb
                n_weak += 1
    for x in range(len(parent)): parent[x] = _find(parent, x)
    return n_strong, n_weak


def run_defuse(tr, cv_th=2.0, conf=0.8, max_group=200):
    pred, nbusy = _it_stats(tr.ip, tr.M, cv_th)
    # groups = functions of one owner ("client"); very large owners are split by application
    keys = tr.owner.copy()
    osz = np.bincount(tr.owner)
    big = osz[tr.owner] > max_group
    keys[big] = tr.owner.max() + 1 + tr.app[big]
    o = np.argsort(keys, kind="stable"); ks = keys[o]
    g_ptr = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True]).astype(np.int64)
    parent = np.arange(tr.nF, dtype=np.int64)
    ns, nw = _mine(tr.ip, tr.M, g_ptr, o.astype(np.int64), pred, nbusy, conf, parent)
    unit_of = np.unique(parent, return_inverse=True)[1].astype(np.int64)
    nU, u_ptr, um, e_ptr, ef, ec, size = _units_from_groups(tr, unit_of)
    out = _run_units(2, nU, u_ptr, um, False, e_ptr, ef, ec, size, tr.nF, 10, cv_th, 5.0, 95.0, 0.0, 10, 0.5)
    res = _pack(tr, out)
    res.update(strong_edges=int(ns), weak_edges=int(nw), dependency_sets=int(nU), predictable_funcs=int(pred.sum()))
    return res


# ------------------------------------------------------------------ FaaSCache: greedy-dual keep-alive under a memory cap
@njit(cache=True)
def _heap_push(hp, hf, hv, n, p, f, v):
    i = n; hp[i] = p; hf[i] = f; hv[i] = v
    while i > 0:
        par = (i - 1) >> 1
        if hp[par] <= hp[i]: break
        hp[par], hp[i] = hp[i], hp[par]; hf[par], hf[i] = hf[i], hf[par]; hv[par], hv[i] = hv[i], hv[par]
        i = par
    return n + 1


@njit(cache=True)
def _heap_pop(hp, hf, hv, n):
    p, f, v = hp[0], hf[0], hv[0]
    n -= 1
    hp[0], hf[0], hv[0] = hp[n], hf[n], hv[n]
    i = 0
    while True:
        l = 2 * i + 1; r = l + 1; s = i
        if l < n and hp[l] < hp[s]: s = l
        if r < n and hp[r] < hp[s]: s = r
        if s == i: break
        hp[s], hp[i] = hp[i], hp[s]; hf[s], hf[i] = hf[i], hf[s]; hv[s], hv[i] = hv[i], hv[s]
        i = s
    return p, f, v, n


@njit(cache=True)
def _faascache(nF, mp, ef, ec, cap):
    cold = np.zeros(nF, np.int64); inv = np.zeros(nF, np.int64); invm = np.zeros(nF, np.int64)
    cached = np.zeros(nF, np.bool_); freq = np.zeros(nF); ver = np.zeros(nF, np.int64); inuse = np.full(nF, -1, np.int64)
    H = 4 * cap + 200000
    hp = np.empty(H); hf = np.empty(H, np.int64); hv = np.empty(H, np.int64); n = 0
    stash_f = np.empty(nF, np.int64)
    mem_t = np.zeros(T_TEST, np.int64); used_t = np.zeros(T_TEST, np.int64)
    clock = 0.0; ncached = 0
    for m in range(END):
        for e in range(mp[m], mp[m + 1]):
            f = ef[e]
            freq[f] += ec[e]; inuse[f] = m
            if m >= TR:
                inv[f] += ec[e]; invm[f] += 1; used_t[m - TR] += 1
            if not cached[f]:
                if m >= TR: cold[f] += 1
                cached[f] = True; ncached += 1
            ver[f] += 1
            if n >= H:                                  # compact stale entries
                k = 0
                for i in range(n):
                    if cached[hf[i]] and hv[i] == ver[hf[i]]:
                        hp[k] = hp[i]; hf[k] = hf[i]; hv[k] = hv[i]; k += 1
                n = 0
                tp = hp[:k].copy(); tf = hf[:k].copy(); tv = hv[:k].copy()
                for i in range(k): n = _heap_push(hp, hf, hv, n, tp[i], tf[i], tv[i])
            n = _heap_push(hp, hf, hv, n, clock + freq[f], f, ver[f])
        ns = 0
        while ncached > cap and n > 0:
            p, f, v, n = _heap_pop(hp, hf, hv, n)
            if not cached[f] or v != ver[f]: continue
            if inuse[f] == m:
                stash_f[ns] = f; ns += 1; continue
            cached[f] = False; ncached -= 1; clock = p; freq[f] = 0.0
        for i in range(ns):
            f = stash_f[i]
            n = _heap_push(hp, hf, hv, n, clock + freq[f], f, ver[f])
        if m >= TR:
            mem_t[m - TR] = ncached
    return cold, inv, invm, mem_t, used_t


@njit(cache=True)
def _by_minute(ip, M, C):
    """Counting sort of trace entries by minute (function order kept inside a minute)."""
    mp = np.zeros(END + 1, np.int64)
    for q in range(len(M)): mp[M[q] + 1] += 1
    for m in range(END): mp[m + 1] += mp[m]
    pos = mp[:-1].copy(); ef = np.empty(len(M), np.int32); ec = np.empty(len(M), np.int32)
    for f in range(len(ip) - 1):
        for q in range(ip[f], ip[f + 1]):
            m = M[q]; ef[pos[m]] = f; ec[pos[m]] = C[q]; pos[m] += 1
    return mp, ef, ec


def run_faascache(tr, cap):
    mp, ef, ec = _by_minute(tr.ip, tr.M, tr.C)
    cold, inv, invm, mem_t, used_t = _faascache(tr.nF, mp, ef, ec, int(cap))
    te = tr.in_test
    return dict(cold=cold[te], inv=inv[te], invm=invm[te], func_idx=np.where(te)[0], used_t=used_t,
                waste_t=mem_t - used_t, wmt=int((mem_t - used_t).sum()), memory_cap=int(cap))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir"); ap.add_argument("method", choices=["fixed", "hybrid_f", "hybrid_a", "defuse", "faascache"])
    ap.add_argument("out"); ap.add_argument("--cap", type=int, default=0, help="FaaSCache memory cap (functions); default = SPES peak")
    a = ap.parse_args()
    tr = Trace(a.data_dir)
    t0 = time.time()
    if a.method == "fixed": res = run_fixed(tr)
    elif a.method == "hybrid_f": res = run_hybrid(tr, False)
    elif a.method == "hybrid_a": res = run_hybrid(tr, True)
    elif a.method == "defuse": res = run_defuse(tr)
    else: res = run_faascache(tr, a.cap)
    res["sim_seconds"] = time.time() - t0
    np.savez_compressed(a.out, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                        meta=json.dumps({k: v for k, v in res.items() if not isinstance(v, np.ndarray)}))
    print(json.dumps(summarize(res), indent=1))
