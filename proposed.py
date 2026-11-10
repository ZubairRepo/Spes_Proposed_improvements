"""Proposed policy: one calibrated probability per function per minute, one threshold.

Every minute, for every function f, estimate p = P(f is called in the next minute) and keep f loaded iff p >= lam/(1+lam).

  1. idle-time chance  (d_f(a) + h0_g(a)) / (n_f(a) + 1): of the times f had been idle a minutes, how often did it wake up
     at the next minute (d = gaps of exactly a, n = gaps of at least a). h0_g = the same estimate pooled over functions with
     the same trigger and call-frequency band (helps new/rare functions). Counts keep updating during the simulation.
  2. clock chance      share of training days on which f ran at this exact minute of the day (needs >= 2 hits)
  3. sibling chance    a function of the same app just started a burst, and f followed it within 2 minutes in >= 20%
                       of >= 3 past cases (rate = followed / (cases + 1))
     raw p = max(1, 2, 3); for gaps >= 30 min, gaps one minute shorter/longer get half credit (timing jitter).
  4. calibration       raw p is replaced by the call rate actually observed for similar situations on held-out days:
                       the table is learned by replaying days 11-12 with a model trained on days 1-10 (never days 13-14);
                       p = (calls + 20 * raw p) / (minutes + 20) in the matching cell
                       (cell = raw idle-time chance x evidence x clock x sibling x idle age x size of the last burst).

Optional rare-function boost: threshold uses lam / (1 + mu / N_f), N_f = expected calls of f over the evaluation window.
Same simulator assumptions and metrics as spes.py / baselines.py.

usage: python proposed.py <data_dir> <out.npz> [--lam 0.003] [--mu 0] [--raw] [--no-clock] [--no-sibling] [--no-jitter]
"""
import os, sys, json, time, argparse
import numpy as np
from numba import njit
from common import Trace, summarize, DAY, TR, END

A = 2880                       # prior hazard length (minutes)
MSM = 20.0                     # calibration smoothing
NCELL = 25 * 5 * 6 * 4 * 8 * 4


# ------------------------------------------------------------------ training statistics (window [0, t0))
@njit(cache=True)
def _basic(ip, M, C, t0):
    nF = len(ip) - 1
    ntr = np.zeros(nF, np.int64); ctr = np.zeros(nF, np.int64); last = np.full(nF, -1, np.int64); lastc = np.ones(nF, np.int64)
    for f in range(nF):
        for k in range(ip[f], ip[f + 1]):
            if M[k] < t0:
                ntr[f] += 1; ctr[f] += C[k]; last[f] = M[k]; lastc[f] = C[k]
    return ntr, ctr, last, lastc


@njit(cache=True)
def _gap_hist(ip, M, t0):
    """Per function: distinct gaps U (sorted), their counts D, and suffix counts S (# gaps >= U)."""
    nF = len(ip) - 1
    cnt = np.zeros(nF, np.int64)
    for f in range(nF):
        prev = -1; seen = 0
        for k in range(ip[f], ip[f + 1]):
            if M[k] >= t0: break
            seen += 1
        cnt[f] = max(seen - 1, 0)
    gp = np.zeros(nF + 1, np.int64); gp[1:] = np.cumsum(cnt)
    U = np.empty(gp[-1], np.int64); D = np.empty(gp[-1], np.int64); S = np.empty(gp[-1], np.int64)
    used = np.zeros(nF, np.int64)
    for f in range(nF):
        n = cnt[f]
        if n == 0: continue
        g = np.empty(n, np.int64)
        for i in range(n):
            g[i] = M[ip[f] + i + 1] - M[ip[f] + i]
        g = np.sort(g)
        k = gp[f]; i = 0
        while i < n:
            j = i
            while j < n and g[j] == g[i]: j += 1
            U[k] = g[i]; D[k] = j - i; k += 1; i = j
        used[f] = k - gp[f]
        s = 0
        for q in range(k - 1, gp[f] - 1, -1):
            s += D[q]; S[q] = s
    # compact (distinct gaps < number of gaps)
    gp2 = np.zeros(nF + 1, np.int64); gp2[1:] = np.cumsum(used)
    U2 = np.empty(gp2[-1], np.int64); D2 = np.empty(gp2[-1], np.int64); S2 = np.empty(gp2[-1], np.int64)
    for f in range(nF):
        for q in range(used[f]):
            U2[gp2[f] + q] = U[gp[f] + q]; D2[gp2[f] + q] = D[gp[f] + q]; S2[gp2[f] + q] = S[gp[f] + q]
    return gp2, U2, D2, S2


@njit(cache=True)
def _prior(gp, U, D, S, last, grp, nG, t0):
    """Pooled hazard per group; every function weighted equally; the still-open last gap counted as censored."""
    d = np.zeros((nG, A + 2)); n = np.zeros((nG, A + 2))
    for f in range(len(gp) - 1):
        if gp[f + 1] == gp[f]: continue
        w = 1.0 / (S[gp[f]] + 1.0); g = grp[f]
        for i in range(gp[f], gp[f + 1]):
            a = U[i]
            if a <= A: d[g, a] += w * D[i]
            n[g, 1] += w * D[i]; n[g, min(a, A) + 1] -= w * D[i]
        c = t0 - last[f]
        n[g, 1] += w; n[g, min(c, A) + 1] -= w
    for g in range(nG):
        for a in range(1, A + 2): n[g, a] += n[g, a - 1]
    h = np.zeros((nG, A + 1))
    for g in range(nG):
        for a in range(1, A + 1):
            if n[g, a] > 0: h[g, a] = d[g, a] / n[g, a]
    return h


@njit(cache=True)
def _profile(ip, M, ntr, maxn, t0, t1):
    """Clock chance: for every minute of [t0, t1) at a minute-of-day f ran on >= 2 training days."""
    nF = len(ip) - 1; ndays = t0 // DAY; edays = (t1 - t0) // DAY
    of = []; om = []; op = []
    cnt = np.zeros(DAY, np.int64)
    for f in range(nF):
        if ntr[f] == 0 or ntr[f] > maxn: continue
        cnt[:] = 0; touched = []
        for k in range(ip[f], ip[f + 1]):
            m = M[k]
            if m >= t0: break
            tau = m % DAY
            if cnt[tau] == 0: touched.append(tau)
            cnt[tau] += 1
        for tau in touched:
            if cnt[tau] >= 2:
                for d in range(edays):
                    of.append(f); om.append(t0 + d * DAY + tau); op.append(cnt[tau] / ndays)
    return np.array(of, np.int64), np.array(om, np.int64), np.array(op)


@njit(cache=True)
def _has(ms, x):
    i = np.searchsorted(ms, x)
    return i < len(ms) and ms[i] == x


@njit(cache=True)
def _any_in(ms, lo, hi):
    i = np.searchsorted(ms, lo)
    return i < len(ms) and ms[i] <= hi


@njit(cache=True)
def _siblings(ip, M, app_ptr, app_f, L, pmin, bmin, maxapp, t0):
    src = []; dst = []; pp = []
    for a in range(len(app_ptr) - 1):
        fs = app_f[app_ptr[a]:app_ptr[a + 1]]
        if len(fs) < 2 or len(fs) > maxapp: continue
        for g in fs:
            mg = M[ip[g]:ip[g + 1]]; mg = mg[:np.searchsorted(mg, t0)]
            if len(mg) == 0: continue
            starts = []
            for i in range(len(mg)):
                if i == 0 or mg[i] - mg[i - 1] > 1: starts.append(mg[i])
            for f in fs:
                if f == g: continue
                mf = M[ip[f]:ip[f + 1]]; mf = mf[:np.searchsorted(mf, t0)]
                if len(mf) == 0: continue
                hit = 0; cases = 0
                for t in starts:
                    if _has(mf, t) or _any_in(mf, t - 10, t - 1): continue   # f already active / warm anyway
                    cases += 1
                    if _any_in(mf, t + 1, t + L): hit += 1
                if cases >= bmin:
                    p = hit / (cases + 1.0)
                    if p >= pmin:
                        src.append(g); dst.append(f); pp.append(p)
    return np.array(src, np.int64), np.array(dst, np.int64), np.array(pp)


@njit(cache=True)
def _events(ip, M, C, t0, t1):
    mp = np.zeros(t1 - t0 + 1, np.int64)
    for q in range(len(M)):
        if M[q] >= t0 and M[q] < t1: mp[M[q] - t0 + 1] += 1
    for m in range(t1 - t0): mp[m + 1] += mp[m]
    pos = mp[:-1].copy(); ef = np.empty(mp[-1], np.int64); ec = np.empty(mp[-1], np.int64)
    for f in range(len(ip) - 1):
        for q in range(ip[f], ip[f + 1]):
            if M[q] >= t0 and M[q] < t1:
                m = M[q] - t0; ef[pos[m]] = f; ec[pos[m]] = C[q]; pos[m] += 1
    return mp, ef, ec


# ------------------------------------------------------------------ simulation
@njit(cache=True)
def _hazard(f, a, gp, U, D, S, cens_on, cens_len, ring, rlen, h0, grp, jitter):
    n = 0.0; d = 0.0
    lo = gp[f]; hi = gp[f + 1]
    if hi > lo:
        i = lo + np.searchsorted(U[lo:hi], a)
        if i < hi:
            n += S[i]
            if U[i] == a: d += D[i]
        if jitter and a >= 30:
            j = lo + np.searchsorted(U[lo:hi], a - 1)
            if j < hi and U[j] == a - 1: d += 0.5 * D[j]; n += 0.5 * D[j]
            j = lo + np.searchsorted(U[lo:hi], a + 1)
            if j < hi and U[j] == a + 1: d += 0.5 * D[j]
    if cens_on[f] and cens_len[f] >= a: n += 1
    for r in range(rlen[f]):
        g = ring[f, r]
        if g >= a: n += 1
        if g == a: d += 1
    pr = h0[grp[f], a] if a <= A else 0.0
    return (d + pr) / (n + 1.0), n


@njit(cache=True)
def _cell(ph, n, pc, ps, age, lc):
    hb = 0 if ph < 1e-6 else min(24, 1 + int(4.0 * (np.log10(ph) + 6.0)))
    eb = 0 if n == 0 else (1 if n < 3 else (2 if n < 10 else (3 if n < 100 else 4)))
    cb = 0 if pc <= 0 else (1 if pc <= 0.1 else (2 if pc <= 0.3 else (3 if pc <= 0.6 else (4 if pc <= 0.9 else 5))))
    sb = 0 if ps <= 0 else (1 if ps < 0.3 else (2 if ps < 0.6 else 3))
    if age < 0: ab = 7
    elif age == 1: ab = 0
    elif age == 2: ab = 1
    elif age <= 5: ab = 2
    elif age <= 10: ab = 3
    elif age <= 30: ab = 4
    elif age <= 120: ab = 5
    elif age <= 720: ab = 6
    else: ab = 7
    lb = 0 if lc <= 1 else (1 if lc <= 5 else (2 if lc <= 50 else 3))
    return ((((hb * 5 + eb) * 6 + cb) * 4 + sb) * 8 + ab) * 4 + lb


@njit(cache=True)
def _simulate(t0, t1, calibrated, collect, lam, mu, use_prof, use_sib, jitter, L,
              nF, ev_ptr, ev_f, ev_c, last0, lastc0, nhat, gp, U, D, S, h0, grp,
              pf_ptr, pf_m, pf_p, sp_ptr, sp_dst, sp_p, tabK, tabN):
    T = t1 - t0; RING = 16
    loaded = np.zeros(nF, np.bool_)
    last = last0.copy(); lastc = lastc0.copy()
    cold = np.zeros(nF, np.int64); inv = np.zeros(nF, np.int64); invm = np.zeros(nF, np.int64); wastef = np.zeros(nF, np.int64)
    used_t = np.zeros(T, np.int64); waste_t = np.zeros(T, np.int64); mem_end = np.zeros(T, np.int64)
    inv_now = np.full(nF, -1, np.int64)
    cens_on = last0 >= 0; cens_len = np.where(last0 >= 0, t0 - last0, 0)
    ring = np.zeros((nF, RING), np.int64); rlen = np.zeros(nF, np.int64); rpos = np.zeros(nF, np.int64)
    sib_p = np.zeros(nF); sib_until = np.full(nF, -1, np.int64)
    pptr = pf_ptr[:-1].copy(); cellc = np.full(nF, -1, np.int64)
    theta = np.empty(nF)
    for f in range(nF):
        x = lam / (1.0 + mu / nhat[f]); theta[f] = x / (1.0 + x)
    wmt = 0
    for t in range(t0 - 1, t1):
        if t >= t0:
            j = t - t0
            for e in range(ev_ptr[j], ev_ptr[j + 1]):
                f = ev_f[e]
                inv_now[f] = t; inv[f] += ev_c[e]; invm[f] += 1
                if not loaded[f]: cold[f] += 1
                runstart = last[f] != t - 1
                if last[f] >= 0:
                    cens_on[f] = False
                    ring[f, rpos[f]] = t - last[f]; rpos[f] = (rpos[f] + 1) % RING
                    if rlen[f] < RING: rlen[f] += 1
                last[f] = t; lastc[f] = ev_c[e]
                if use_sib and runstart:
                    for s in range(sp_ptr[f], sp_ptr[f + 1]):
                        q = sp_dst[s]
                        if sib_until[q] < t + 1 or sp_p[s] > sib_p[q]: sib_p[q] = sp_p[s]
                        sib_until[q] = t + L
            for f in range(nF):
                if collect and cellc[f] >= 0:
                    tabN[cellc[f]] += 1.0
                    if inv_now[f] == t: tabK[cellc[f]] += 1.0
                if inv_now[f] == t:
                    used_t[j] += 1
                elif loaded[f]:
                    waste_t[j] += 1; wastef[f] += 1; wmt += 1
        nt = t + 1
        if nt >= t1: break
        nl = 0
        for f in range(nF):
            ph = 0.0; n = 0.0; age = -1
            if last[f] >= 0:
                age = nt - last[f]
                ph, n = _hazard(f, age, gp, U, D, S, cens_on, cens_len, ring, rlen, h0, grp, jitter)
            pc = 0.0
            if use_prof:
                while pptr[f] < pf_ptr[f + 1] and pf_m[pptr[f]] < nt: pptr[f] += 1
                if pptr[f] < pf_ptr[f + 1] and pf_m[pptr[f]] == nt: pc = pf_p[pptr[f]]
            ps = 0.0
            if use_sib and sib_until[f] >= nt: ps = sib_p[f]
            p = max(ph, max(pc, ps))
            if collect or calibrated:
                c = _cell(ph, n, pc, ps, age, lastc[f]); cellc[f] = c
                if calibrated:
                    p = (tabK[c] + MSM * p) / (tabN[c] + MSM)
            loaded[f] = p >= theta[f]
            if loaded[f]: nl += 1
        if t >= t0: mem_end[t - t0] = nl
    return cold, inv, invm, wastef, used_t, waste_t, mem_end, wmt


class Model:
    """All statistics for one train/evaluate split: trained on [0, t0), evaluated on [t0, t1)."""

    def __init__(self, tr, t0, t1, use_prof=True, use_sib=True, jitter=True, log=print):
        self.tr, self.t0, self.t1 = tr, t0, t1
        self.use_prof, self.use_sib, self.jitter = use_prof, use_sib, jitter
        ip, M, C = tr.ip, tr.M, tr.C
        tt = time.time()
        self.ntr, ctr, self.last0, self.lastc0 = _basic(ip, M, C, t0)
        self.nhat = np.maximum(ctr * (t1 - t0) / t0, 1.0)
        first_trig = tr.meta["triggers"].astype(str).str.split("|").str[0]
        tcode = np.unique(first_trig, return_inverse=True)[1]
        self.grp = (tcode * 6 + np.digitize(self.ntr, [1, 3, 13, 145, 1441])).astype(np.int64)
        self.gp, self.U, self.D, self.S = _gap_hist(ip, M, t0)
        self.h0 = _prior(self.gp, self.U, self.D, self.S, self.last0, self.grp, int(self.grp.max()) + 1, t0)
        f, m, p = _profile(ip, M, self.ntr, 3000, t0, t1) if use_prof else (np.zeros(0, np.int64),) * 2 + (np.zeros(0),)
        o = np.lexsort((m, f)); self.pf_ptr = np.searchsorted(f[o], np.arange(tr.nF + 1)).astype(np.int64)
        self.pf_m, self.pf_p = m[o], p[o]
        if use_sib:
            o = np.argsort(tr.app, kind="stable")
            app_ptr = np.searchsorted(tr.app[o], np.arange(tr.app.max() + 2)).astype(np.int64)
            s, d, pp = _siblings(ip, M, app_ptr, o.astype(np.int64), 2, 0.2, 3, 40, t0)
        else:
            s, d, pp = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
        o = np.argsort(s, kind="stable")
        self.sp_ptr = np.searchsorted(s[o], np.arange(tr.nF + 1)).astype(np.int64); self.sp_dst, self.sp_p = d[o], pp[o]
        self.ev_ptr, self.ev_f, self.ev_c = _events(ip, M, C, t0, t1)
        self.tabK = np.zeros(NCELL); self.tabN = np.zeros(NCELL)
        log(f"model [{t0 // DAY} train days -> {(t1 - t0) // DAY} eval days]: {len(self.pf_m):,} clock entries, "
            f"{len(s):,} sibling links ({time.time() - tt:.0f}s)")

    def run(self, lam, mu=0.0, calibrated=True, collect=False):
        t = time.time()
        out = _simulate(self.t0, self.t1, calibrated, collect, lam, mu, self.use_prof, self.use_sib, self.jitter, 2,
                        self.tr.nF, self.ev_ptr, self.ev_f, self.ev_c, self.last0, self.lastc0, self.nhat,
                        self.gp, self.U, self.D, self.S, self.h0, self.grp,
                        self.pf_ptr, self.pf_m, self.pf_p, self.sp_ptr, self.sp_dst, self.sp_p, self.tabK, self.tabN)
        cold, inv, invm, wastef, used_t, waste_t, mem_end, wmt = out
        ev = invm > 0
        return dict(cold=cold[ev], inv=inv[ev], invm=invm[ev], waste=wastef[ev], func_idx=np.where(ev)[0], used_t=used_t,
                    waste_t=waste_t, mem_end_t=mem_end, wmt=int(wmt), lam=lam, mu=mu, calibrated=calibrated,
                    sim_seconds=time.time() - t)


def fit_calibration(tr, use_prof=True, use_sib=True, jitter=True, log=print):
    """Learn the correction table: train on days 1-10, observe days 11-12 (the test days 13-14 are never touched)."""
    m = Model(tr, TR - 2 * DAY, TR, use_prof, use_sib, jitter, log)
    m.run(lam=1.0, calibrated=False, collect=True)          # features do not depend on lam
    log(f"calibration table: {m.tabN.sum():,.0f} function-minutes, {m.tabK.sum():,.0f} with a call, "
        f"{(m.tabN > 0).sum():,} non-empty cells")
    return m.tabK.copy(), m.tabN.copy()


def reliability(tabK, tabN):
    """Observed call rate per raw idle-time-chance bin (for the calibration figure)."""
    hb = np.arange(NCELL) // (5 * 6 * 4 * 8 * 4)
    rows = []
    for b in range(25):
        s = hb == b; n = tabN[s].sum()
        if n > 0:
            lo = 0.0 if b == 0 else 10 ** (-6 + (b - 1) / 4); hi = 1e-6 if b == 0 else 10 ** (-6 + b / 4)
            rows.append((b, lo, hi, n, tabK[s].sum() / n))
    return np.array(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir"); ap.add_argument("out")
    ap.add_argument("--lam", type=float, default=0.003, help="threshold knob: load iff p >= lam/(1+lam)")
    ap.add_argument("--mu", type=float, default=0.0, help="rare-function boost (0 = off)")
    ap.add_argument("--raw", action="store_true", help="no calibration")
    ap.add_argument("--no-clock", action="store_true"); ap.add_argument("--no-sibling", action="store_true")
    ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--calib", default=None, help="cache file for the calibration table (.npz)")
    a = ap.parse_args()
    tr = Trace(a.data_dir)
    flags = dict(use_prof=not a.no_clock, use_sib=not a.no_sibling, jitter=not a.no_jitter)
    model = Model(tr, TR, END, **flags)
    if not a.raw:
        if a.calib and os.path.exists(a.calib):
            z = np.load(a.calib); model.tabK[:], model.tabN[:] = z["K"], z["N"]
        else:
            K, N = fit_calibration(tr, **flags); model.tabK[:], model.tabN[:] = K, N
            if a.calib: np.savez(a.calib, K=K, N=N)
    res = model.run(a.lam, a.mu, calibrated=not a.raw)
    np.savez_compressed(a.out, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                        meta=json.dumps({k: v for k, v in res.items() if not isinstance(v, np.ndarray)}))
    print(json.dumps(summarize(res), indent=1))
