"""SPES (Lee et al., ICDE 2024): a fast, faithful port of the authors' released notebook
(github.com/BEbillionaireUSD/SPES, run.ipynb + cal_corr.py).

Pipeline (train = days 1-12, test = days 13-14):
  1. categorize()      always-warm / regular / appro-regular / dense / successive (+ forgetting)
  2. find_candidates() T-lagged co-occurrence candidates of unknown functions (cal_corr.py)
  3. assign()          validate pulsed / possible / correlated on days 10-12 and pick one
  4. simulate()        minute-by-minute provisioning of days 13-14 with adaptive updates

Every constant below is the value used in the notebook. Places where the notebook could not have
run as written (it would raise) are handled as noted in README.md ("Port notes").
"""
import os, sys, json, time, argparse, pickle
import numpy as np
from numba import njit
from common import (Trace, rle, top_modes, summarize, DAY, TR, END, T_TEST,
                    UNKNOWN, WARM, REGULAR, APPRO, DENSE, SUCCESSIVE, PULSED, POSSIBLE, CORR, NEW_POSS)

SEQ_NUM_BOUND, IDLE_NUM_MAX, IDLE_PERCEN = 3, 3, 0.9
DENSE_UPPER_BOUND, DENSE_PERCEN = 5, 90
GAMMA1, GAMMA2 = 10, 11
FORGET_DAYS, VALID_SPLIT_DAY = 6, 9
PULSED_GIVEUP, POSS_GIVEUP, CORR_GIVEUP = 3, 1, 1
DISCRETE_TH, EN_STD, ALPHA = 10, 2, 0.5
WARM_TH, MAX_LAG = 0.7, 10
NONE = -(1 << 40)          # "None" for integer times
CAP_MODES, CAP_SLOT = 256, 24


# ============================================================ 1. categorization
def merge(lst):
    """Slacking rule: merge small waiting times next to the dominant one (notebook `merge`)."""
    lst = [int(x) for x in lst]
    unique, counts = np.unique(lst, return_counts=True)
    mode = unique[np.argmax(counts)]
    if mode < 500 or max(counts) < len(counts) / 2:
        return lst
    used = [False] * len(lst); new = []
    for i in range(len(lst)):
        if lst[i] <= mode * 0.25:
            continue
        cur = lst[i]
        for k in range(1, i + 1):
            ci = max(0, i - k)
            if lst[ci] >= mode * 0.75: break
            if cur + lst[ci] <= mode and lst[ci] <= mode * 0.25 and not used[ci]:
                used[ci] = True; cur += lst[ci]
        for k in range(1, len(lst) - i):
            ci = min(len(lst) - 1, i + k)
            if lst[ci] >= mode * 0.75: break
            if cur + lst[ci] <= mode and lst[ci] <= mode * 0.25 and not used[ci]:
                used[ci] = True; cur += lst[ci]
        new.append(cur)
    if sum(used) / len(used) > 0.3:
        return lst
    return new


def _regular(w):
    w = np.asarray(w, float)
    return (np.percentile(w, 95) - np.percentile(w, 5) < 1) or (np.std(w) / np.mean(w) <= 0.01)


def _appro(wt):
    u, c = np.unique(wt, return_counts=True)
    return c.max() > 1 and np.sort(c)[::-1][:IDLE_NUM_MAX].sum() >= c.sum() * IDLE_PERCEN


def first_pass(ms, cs):
    zeros = TR - len(ms)
    if zeros == 0 or zeros < 0.001 * TR:
        return WARM
    wt, at, an, fi, li = rle(ms, cs, 0, TR)
    if len(at) >= SEQ_NUM_BOUND and len(wt) > 0:
        u, c = np.unique(wt, return_counts=True)
        mode = u[np.argmax(c)]
        w = wt
        if mode - wt[0] > 2:
            w = wt[1:] if li else wt[1:-1]
        if len(w) > SEQ_NUM_BOUND and _regular(w):
            return REGULAR
        if _regular(merge(w)):
            return REGULAR
    if len(wt) >= SEQ_NUM_BOUND and _appro(wt):
        return APPRO
    if np.percentile(wt, DENSE_PERCEN) < DENSE_UPPER_BOUND:
        return DENSE
    if at.min() >= GAMMA1 or an.min() >= GAMMA2:
        return SUCCESSIVE
    return UNKNOWN


def recategorize(ms, cs, a, b):
    """`re_categorzing` used by forgetting (note: slightly different from the first pass, as in the notebook)."""
    zeros = (b - a) - len(ms)
    if zeros == 0 or zeros < 0.001 * (b - a):
        return WARM
    wt, at, an, fi, li = rle(ms, cs, a, b)
    if len(at) >= SEQ_NUM_BOUND:
        w = wt if li else wt[:-1]
        if _regular(w) or _regular(merge(w)):
            return REGULAR
        if _appro(wt):
            return APPRO
    if np.percentile(wt, DENSE_PERCEN) < DENSE_UPPER_BOUND:
        return DENSE
    if at.min() >= GAMMA1 or an.min() > GAMMA2:
        return SUCCESSIVE
    return UNKNOWN


def categorize(tr, forgetting=True, log=print):
    typ = np.full(tr.nF, -1, np.int64)        # -1: not in the training data
    forget = np.zeros(tr.nF, np.int64)
    train = np.where(tr.in_train)[0]
    for f in train:
        ms, cs = tr.window(f, 0, TR)
        typ[f] = first_pass(ms, cs)
    first = np.bincount(typ[train], minlength=6)[:6]
    log(f"first pass  unknown/warm/regular/appro/dense/successive = {first.tolist()}")
    before = typ.copy()
    if forgetting:
        per_day = []
        for day in range(1, FORGET_DAYS + 1):
            n = 0
            for f in train[typ[train] == UNKNOWN]:
                ms, cs = tr.window(f, day * DAY, TR)
                if len(ms) == 0:
                    continue
                t = recategorize(ms, cs, day * DAY, TR)
                if t > 0:
                    typ[f] = t; forget[f] = day; n += 1
            per_day.append(n)
        log(f"forgetting re-categorized per day {per_day} (total {sum(per_day)})")
    after = np.bincount(typ[train], minlength=6)[:6]
    log(f"after forgetting = {after.tolist()}")
    return typ, forget, before


# ============================================================ 2. correlation candidates (cal_corr.py)
@njit(cache=True)
def _count_range(ms, a, b):
    return np.searchsorted(ms, b) - np.searchsorted(ms, a)


@njit(cache=True)
def _candidates(targets, ip, M, forget, cand_ptr, cand_idx, out_g, out_lag, out_corr, out_ptr):
    k = 0
    for ti in range(len(targets)):
        f = targets[ti]
        mf = M[ip[f]:ip[f + 1]]
        mf = mf[:np.searchsorted(mf, TR)]
        out_ptr[ti] = k
        for ci in range(cand_ptr[ti], cand_ptr[ti + 1]):
            g = cand_idx[ci]
            mg = M[ip[g]:ip[g + 1]]
            mg = mg[:np.searchsorted(mg, TR)]
            s = forget[g] * DAY
            L = TR - s
            for lag in range(MAX_LAG + 1):
                # pre = g on [s, TR-lag), cur = f on [s+lag, TR); aligned position p: g at s+p, f at s+p+lag
                cur_cnt = _count_range(mf, s + lag, TR)
                if cur_cnt == 0:
                    continue
                pre_cnt = _count_range(mg, s, TR - lag)
                nz = (L - lag) - cur_cnt
                lo_both = pre_cnt - (1.0 - WARM_TH) * nz  # needed for zero_inconsistency <= 0.3
                hi_both = min(pre_cnt, cur_cnt)
                if hi_both < WARM_TH * cur_cnt or hi_both < lo_both:
                    continue
                both = 0
                if cur_cnt <= pre_cnt:
                    a0 = np.searchsorted(mf, s + lag); a1 = np.searchsorted(mf, TR)
                    for q in range(a0, a1):
                        t = mf[q] - lag
                        j = np.searchsorted(mg, t)
                        if j < len(mg) and mg[j] == t:
                            both += 1
                else:
                    a0 = np.searchsorted(mg, s); a1 = np.searchsorted(mg, TR - lag)
                    for q in range(a0, a1):
                        t = mg[q] + lag
                        j = np.searchsorted(mf, t)
                        if j < len(mf) and mf[j] == t:
                            both += 1
                corr = both / cur_cnt
                zinc = (pre_cnt - both) / nz if nz > 0 else 1e9
                if corr >= WARM_TH and zinc <= 1.0 - WARM_TH:
                    out_g[k] = g; out_lag[k] = lag; out_corr[k] = corr; k += 1
                    break
    out_ptr[len(targets)] = k
    return k


def find_candidates(tr, typ, forget, log=print):
    """For each unknown training function: same-owner/same-app functions whose calls precede its calls."""
    train = tr.in_train
    groups = {}
    for key_arr, name in ((tr.owner, "o"), (tr.app, "a")):
        idx = np.where(train)[0]
        order = idx[np.argsort(key_arr[idx], kind="stable")]
        keys = key_arr[order]
        bounds = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
        groups[name] = {keys[bounds[i]]: order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)}
    targets, cand_ptr, cand_idx = [], [0], []
    for f in np.where(train & (typ == UNKNOWN))[0]:
        c = np.union1d(groups["o"][tr.owner[f]], groups["a"][tr.app[f]])
        c = c[c != f]
        if len(c) == 0:
            continue
        targets.append(f); cand_idx.extend(c.tolist()); cand_ptr.append(len(cand_idx))
    targets = np.array(targets, np.int64); cand_ptr = np.array(cand_ptr, np.int64); cand_idx = np.array(cand_idx, np.int64)
    n = len(cand_idx)
    og, ol, oc = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n)
    optr = np.empty(len(targets) + 1, np.int64)
    t0 = time.time()
    _candidates(targets, tr.ip, tr.M, forget, cand_ptr, cand_idx, og, ol, oc, optr)
    candi = {}
    for i, f in enumerate(targets):
        a, b = optr[i], optr[i + 1]
        if b > a:
            o = np.argsort(-oc[a:b], kind="stable")
            candi[int(f)] = [(int(og[a + j]), int(ol[a + j]), float(oc[a + j])) for j in o]
    log(f"correlation candidates: {len(candi):,} unknown functions with >=1 candidate "
        f"({len(targets):,} examined, {n:,} pairs, {time.time() - t0:.0f}s)")
    return candi


def load_candidates_json(tr, path):
    h2i = {h: i for i, h in enumerate(tr.hash)}
    raw = json.load(open(path))
    return {h2i[f]: [(h2i[g], int(l), float(c)) for g, l, c in lst] for f, lst in raw.items() if f in h2i}


# ============================================================ 3. assignment (validation on days 10-12)
V0 = VALID_SPLIT_DAY * DAY          # first validation minute
VL = TR - V0                        # 4320 validation minutes


@njit(cache=True)
def _sim_pulsed(vm):
    inv = np.zeros(VL, np.bool_)
    for t in vm: inv[t] = True
    state = False; cold = 0; waste = 0
    for i in range(VL):
        if inv[i]:
            if not state:
                state = True; cold += 1
        elif state:
            waste += 1
            quiet = True
            for q in range(max(0, i + 1 - PULSED_GIVEUP), i):
                if inv[q]: quiet = False
            if quiet: state = False
    return cold, waste


@njit(cache=True)
def _flag(intervals_l, intervals_r, n, i, pw):
    for q in range(n):
        if intervals_l[q] <= i + pw and intervals_r[q] >= i - pw:
            return True
    return False


@njit(cache=True)
def _sim_possible(vm, pred, disc, last_call, state0):
    inv = np.zeros(VL, np.bool_)
    for t in vm: inv[t] = True
    il = np.empty(len(pred), np.float64); ir = np.empty(len(pred), np.float64); n = 0
    if last_call != NONE:
        if disc:
            for q in range(len(pred)):
                il[q] = last_call + pred[q] + 1; ir[q] = il[q]
            n = len(pred)
        else:
            il[0] = last_call + pred[0] + 1; ir[0] = last_call + pred[1] + 1; n = 1
    state = state0 or _flag(il, ir, n, 0, 2)
    cold = 0; waste = 0
    for i in range(VL):
        if inv[i]:
            if not state:
                state = True; cold += 1
            if disc:
                for q in range(len(pred)):
                    il[q] = i + pred[q] + 1; ir[q] = il[q]
                n = len(pred)
            else:
                il[0] = i + pred[0] + 1; ir[0] = i + pred[1] + 1; n = 1
        else:
            fl = _flag(il, ir, n, i, 2)
            if state:
                waste += 1
                if not fl: state = False
            elif fl:
                state = True
    return cold, waste


@njit(cache=True)
def _build_corr(il, ir, last, c_lag, c_kind, c_pred, c_pptr):
    n = 0
    for k in range(len(c_lag)):
        cl = last[k]
        if cl == NONE: continue
        lag = c_lag[k]
        il[n] = cl; ir[n] = cl + lag; n += 1
        p0, p1 = c_pptr[k], c_pptr[k + 1]
        if c_kind[k] == REGULAR or c_kind[k] == APPRO:
            for q in range(p0, p1):
                il[n] = cl + 1 + c_pred[q]; ir[n] = cl + 1 + c_pred[q] + lag; n += 1
        elif c_kind[k] == DENSE:
            il[n] = cl + 1 + c_pred[p0]; ir[n] = cl + 1 + c_pred[p0 + 1] + lag; n += 1
    return n


@njit(cache=True)
def _sim_corr(vm, state0, c_min, c_ptr, c_lag, c_kind, c_pred, c_pptr):
    """c_min[c_ptr[k]:c_ptr[k+1]]: minutes (relative to V0, may be negative) when candidate k was called."""
    inv = np.zeros(VL, np.bool_)
    for t in vm: inv[t] = True
    nc = len(c_lag)
    last = np.full(nc, NONE, np.int64); ptr = c_ptr[:-1].copy()
    for k in range(nc):  # last call before the validation window
        while ptr[k] < c_ptr[k + 1] and c_min[ptr[k]] < 0:
            last[k] = c_min[ptr[k]]; ptr[k] += 1
    cap = 1
    for k in range(nc): cap += 1 + (c_pptr[k + 1] - c_pptr[k])
    il = np.empty(cap); ir = np.empty(cap)
    n = _build_corr(il, ir, last, c_lag, c_kind, c_pred, c_pptr)
    state = state0 or _flag(il, ir, n, 0, 2)
    cold = 0; waste = 0
    for i in range(VL):
        for k in range(nc):
            while ptr[k] < c_ptr[k + 1] and c_min[ptr[k]] <= i:
                last[k] = c_min[ptr[k]]; ptr[k] += 1
        n = _build_corr(il, ir, last, c_lag, c_kind, c_pred, c_pptr)
        if inv[i]:
            if not state:
                state = True; cold += 1
        else:
            fl = _flag(il, ir, n, i, 2)
            if state:
                waste += 1
                if not fl: state = False
            elif fl:
                state = True
    return cold, waste


def _valid_minutes(tr, f):
    ms, cs = tr.window(f, V0, TR)
    return (ms - V0).astype(np.int64), int(cs.sum()), len(ms)


def _last_before(tr, f, t):
    ms, _ = tr.window(f, 0, t)
    return int(ms[-1] - t) if len(ms) else NONE


def candidate_pred(tr, g, typ, forget):
    """candi_pred_value (notebook cell 36)."""
    ms, cs = tr.window(g, forget[g] * DAY, TR)
    wt = rle(ms, cs, forget[g] * DAY, TR)[0]
    if typ[g] == REGULAR:
        return [float(np.median(wt))]
    if typ[g] in (APPRO, DENSE):
        u, c = np.unique(wt, return_counts=True)
        o = np.argsort(-c, kind="stable")[:min(5, int((c > 1).sum()))]
        if typ[g] == APPRO:
            return [float(x) for x in u[o][:IDLE_NUM_MAX]]
        if len(o) > 0:
            return [float(u[o].min()), float(min(DENSE_UPPER_BOUND, u[o].max()))]
        return [float(u.min()), float(min(DENSE_UPPER_BOUND, u.max()))]
    return []


def assign(tr, typ, forget, candi, use_corr=True, log=print):
    typ = typ.copy()
    unk = np.where(tr.in_train & (typ == UNKNOWN))[0]
    BIG = 20000.0
    res_cold = {f: [BIG] * 3 for f in unk}; res_waste = {f: [BIG] * 3 for f in unk}
    possible = set(); n_pul = n_pos = n_cor = useless = 0
    for f in unk:                                   # pulsed
        vm, inv_num, used = _valid_minutes(tr, f)
        if inv_num <= 0:
            useless += 1; continue
        c, w = _sim_pulsed(vm); n_pul += 1
        res_cold[f][0], res_waste[f][0] = c / inv_num, w / used
    for f in unk:                                   # possible
        vm, inv_num, used = _valid_minutes(tr, f)
        ms, cs = tr.window(f, 0, TR)
        wt = rle(ms, cs, 0, TR)[0]
        u, c = np.unique(wt, return_counts=True)
        if inv_num == 0:
            if c.max() >= 2: possible.add(f)
            continue
        if c.max() < 2:
            continue
        n_pos += 1
        modes = top_modes(wt, min(5, int((c > 1).sum())))[0].astype(float)
        disc = modes.max() - modes.min() > DISCRETE_TH
        pred = np.sort(modes) if disc else np.array([modes.min(), modes.max()])
        lc = _last_before(tr, f, V0)
        st0 = len(tr.window(f, V0 - 1, V0)[0]) > 0
        cc, w = _sim_possible(vm, pred, disc, lc, st0)
        res_cold[f][1], res_waste[f][1] = cc / inv_num, w / used
    if use_corr:
        pred_cache = {}
        for f, lst in candi.items():                # correlated
            if typ[f] != UNKNOWN: continue
            vm, inv_num, used = _valid_minutes(tr, f)
            if inv_num <= 0: continue
            n_cor += 1
            c_min, c_ptr, c_lag, c_kind, c_pred, c_pptr = [], [0], [], [], [], [0]
            for g, lag, _ in lst:
                ms, _ = tr.window(g, 0, TR)
                c_min.extend((ms - V0).tolist()); c_ptr.append(len(c_min))
                c_lag.append(lag); c_kind.append(typ[g])
                if g not in pred_cache: pred_cache[g] = candidate_pred(tr, g, typ, forget)
                c_pred.extend(pred_cache[g]); c_pptr.append(len(c_pred))
            ms, _ = tr.window(f, V0 - 1, V0)
            cc, w = _sim_corr(vm, len(ms) > 0, np.array(c_min, np.int64), np.array(c_ptr, np.int64),
                              np.array(c_lag, np.int64), np.array(c_kind, np.int64),
                              np.array(c_pred, float), np.array(c_pptr, np.int64))
            res_cold[f][2], res_waste[f][2] = cc / inv_num, w / used
    final = {}; good = bad = 0; forced = [0, 0, 0]; left_unknown = 0
    for f in unk:
        rc, rw = res_cold[f], res_waste[f]
        if min(rc) < BIG:
            ci, wi = int(np.argmin(rc)), int(np.argmin(rw))
            if ci == wi and rc[ci] < 1:
                final[f] = ci; good += 1
            elif ci < wi and rc[ci] == rc[wi]:
                final[f] = wi; good += 1
            elif wi < ci and rw[ci] == rw[wi]:
                final[f] = ci; good += 1
            elif rc[ci] <= 0.1:
                final[f] = ci
            elif rc[ci] == 1:
                bad += 1
                final[f] = 1 if f in possible else 0
            else:
                if f in possible:
                    final[f] = 1
                else:
                    dc = ALPHA * (rc[wi] - rc[ci]) / rc[ci]
                    dw = (rw[ci] - rw[wi]) / rw[wi]
                    final[f] = ci if dc <= dw else wi
                forced[final[f]] += 1
        elif f in possible:
            final[f] = 1
        else:
            left_unknown += 1
    for f, k in final.items():
        typ[f] = PULSED + k
    log(f"assignment: tried pulsed/possible/corr = {n_pul}/{n_pos}/{n_cor}; useless {useless}; "
        f"assigned {len(final):,} (smooth {good}, bad grace {bad}, forced {forced}); still unknown {left_unknown:,}")
    return typ


# ============================================================ 4. provisioning simulation
def prepare(tr, typ, forget, candi, cfg):
    """Per-function initial state at the start of day 13 (notebook cells 52-65)."""
    nF = tr.nF
    prewarm = cfg["prewarm"]
    giveup = np.array([POSS_GIVEUP, 1440 * 14, 1, 1, DENSE_UPPER_BOUND, 1, PULSED_GIVEUP, POSS_GIVEUP, CORR_GIVEUP, POSS_GIVEUP], np.int64)
    giveup = np.where(np.arange(10) == WARM, giveup, giveup * cfg["giveup_scale"])
    T = typ.copy(); T[T < 0] = UNKNOWN                       # unseen test-only functions -> unknown
    last = np.full(nF, NONE, np.int64); wait = np.full(nF, -1, np.int64)
    lmax = np.zeros(nF, np.int64); nseq = np.zeros(nF, np.int64); std = np.zeros(nF)
    reg_pred = np.zeros(nF); reg_l = np.zeros(nF); reg_r = np.zeros(nF)
    d_lo = np.zeros(nF); d_hi = np.zeros(nF)
    anchor = np.full(nF, NONE, np.int64); nis_on = np.zeros(nF, np.bool_)
    ap_val = [[] for _ in range(nF)]
    pslot = np.full(nF, -1, np.int64)
    pool_modes, pool_cnt, pool_len, pdisc = [], [], [], []
    for f in np.where(tr.in_train)[0]:
        if T[f] == WARM:
            continue
        a = forget[f] * DAY
        ms, cs = tr.window(f, a, TR)
        wt, at, an, fi, li = rle(ms, cs, a, TR)
        last[f] = ms[-1] - TR
        wait[f] = 0 if li else wt[-1]
        lmax[f], nseq[f], std[f] = at.max(), len(at), np.std(wt)
        s = min(std[f], EN_STD)
        if T[f] == REGULAR:
            reg_pred[f] = np.median(wt)
            p = last[f] + 1
            if p < 0: p += reg_pred[f] + 1
            reg_l[f], reg_r[f] = p - s, p + s
            continue
        u, c = np.unique(wt, return_counts=True)
        cut = min(5, int((c > 1).sum()))
        o = np.argsort(-c, kind="stable")
        modes, mcnt = u[o[:cut]], c[o[:cut]]
        if T[f] == APPRO:
            ap_val[f] = [float(u[j]) for j in o[:IDLE_NUM_MAX] if c[j] > 0.1 * c.sum()]
            anchor[f] = last[f]; nis_on[f] = True
        elif T[f] == DENSE:
            if cut > 0: d_lo[f], d_hi[f] = modes.min(), min(10, modes.max())
            else: d_lo[f], d_hi[f] = u.min(), min(IDLE_NUM_MAX, u.max())
            anchor[f] = last[f]; nis_on[f] = True
        if T[f] == UNKNOWN and cut > 0:
            T[f] = POSSIBLE
        if T[f] in (POSSIBLE, UNKNOWN):
            pslot[f] = len(pool_len)
            pool_modes.append(modes.astype(float)); pool_cnt.append(mcnt.astype(np.int64)); pool_len.append(len(modes))
            pdisc.append(len(modes) > 0 and modes.max() - modes.min() > DISCRETE_TH)
            if T[f] == POSSIBLE:
                anchor[f] = last[f]; nis_on[f] = True
    for f in np.where(~tr.in_train)[0]:                      # unseen functions may become newly-possible
        pslot[f] = len(pool_len); pool_modes.append(np.zeros(0)); pool_cnt.append(np.zeros(0, np.int64)); pool_len.append(0); pdisc.append(False)
    P = len(pool_len)
    pm = np.zeros((P, CAP_MODES)); pc = np.zeros((P, CAP_MODES), np.int64)
    for k in range(P):
        pm[k, :pool_len[k]] = pool_modes[k]; pc[k, :pool_len[k]] = pool_cnt[k]
    pl = np.array(pool_len, np.int64); pd_ = np.array(pdisc, np.bool_)
    ap_ptr = np.zeros(nF + 1, np.int64); ap_ptr[1:] = np.cumsum([len(x) for x in ap_val])
    ap_flat = np.array([v for x in ap_val for v in x], float)

    def own_intervals(g):
        """Current next-invocation intervals of g as (l, r) pairs (empty if none)."""
        t = T[g]
        if t == REGULAR: return [(reg_l[g], reg_r[g])]
        if not nis_on[g]: return []
        a = anchor[g]
        if t == APPRO: return [(a + p + 1, a + p + 1) for p in ap_val[g]]
        if t == DENSE: return [(a + d_lo[g] + 1, a + d_hi[g] + 1)]
        if t in (POSSIBLE, NEW_POSS):
            k = pslot[g]; m = pm[k, :pl[k]]
            if pd_[k]: return [(a + p + 1, a + p + 1) for p in m]
            return [(a + m.min() + 1, a + m.max() + 1)]
        return []

    # correlated targets: one slot per (target, candidate); initial intervals in training order (cell 63)
    slots_t, slots_g, slots_lag, slots_int = [], [], [], []
    corr_targets = [f for f in range(nF) if T[f] == CORR]
    warm_cands = 0
    for f in corr_targets:
        for g, lag, _ in candi[f]:
            ints = []
            if last[g] == NONE:
                warm_cands += 1
            else:
                ints.append((last[g], last[g] + lag))
                if T[g] != CORR and g < f:
                    ints += [(l, r + lag) for (l, r) in own_intervals(g)]
            slots_t.append(f); slots_g.append(g); slots_lag.append(lag); slots_int.append(ints[:CAP_SLOT])
    S = len(slots_t)
    s_int = np.zeros((max(S, 1), CAP_SLOT, 2)); s_n = np.zeros(max(S, 1), np.int64)
    for k, ints in enumerate(slots_int):
        s_n[k] = len(ints)
        for q, (l, r) in enumerate(ints): s_int[k, q] = (l, r)
    s_t = np.array(slots_t, np.int64); s_g = np.array(slots_g, np.int64); s_lag = np.array(slots_lag, np.int64)
    o = np.argsort(s_t, kind="stable"); tptr = np.searchsorted(s_t[o], np.arange(nF + 1)); t_slots = o.astype(np.int64)
    o = np.argsort(s_g, kind="stable"); gptr = np.searchsorted(s_g[o], np.arange(nF + 1)); g_slots = o.astype(np.int64)
    if not cfg["use_corr_sim"]:
        gptr[:] = 0
    # online correlation for unseen functions (cell 56)
    src, dst = [], []
    if cfg["online_corr"] != "off":
        te = np.where(tr.in_test)[0]
        by_owner, by_app = {}, {}
        for f in te:
            by_owner.setdefault(tr.owner[f], []).append(f); by_app.setdefault(tr.app[f], []).append(f)
        for f in te[~tr.in_train[te]]:
            cands = set(by_owner[tr.owner[f]]) | set(by_app[tr.app[f]])
            cands.discard(f)
            for g in cands:
                if tr.trig[f] & tr.trig[g]:
                    if cfg["online_corr"] == "code":   # as in the released code: when unseen f is called, pre-load g
                        src.append(f); dst.append(g)
                    else:                              # as described in the paper: when g is called, pre-load unseen f
                        src.append(g); dst.append(f)
    src = np.array(src, np.int64); dst = np.array(dst, np.int64)
    o = np.argsort(src, kind="stable"); oc_ptr = np.searchsorted(src[o], np.arange(nF + 1)).astype(np.int64); oc_dst = dst[o]
    # initial memory (cell 65)
    state = np.zeros(nF, np.bool_)
    for f in np.where(tr.in_train)[0]:
        t = T[f]
        if t == WARM: state[f] = True
        elif t in (SUCCESSIVE, CORR, POSSIBLE, NEW_POSS, UNKNOWN) and last[f] == -1: state[f] = True
        elif t == PULSED and last[f] + giveup[PULSED] >= 0: state[f] = True
        elif t in (REGULAR, APPRO, DENSE) and last[f] == -1 and (lmax[f] > 1 or nseq[f] == 1): state[f] = True
        if t == CORR:
            for k in range(tptr[f], tptr[f + 1]):
                sl = t_slots[k]
                for q in range(s_n[sl]):
                    if s_int[sl, q, 0] <= prewarm and s_int[sl, q, 1] >= -prewarm: state[f] = True
        elif t in (REGULAR, APPRO, DENSE, POSSIBLE, NEW_POSS):
            for l, r in own_intervals(f):
                if l <= prewarm and r >= -prewarm: state[f] = True
    order = np.r_[np.where(T != CORR)[0], np.where(T == CORR)[0]].astype(np.int64)
    info = dict(warm_candidates_skipped=warm_cands, corr_slots=S, online_corr_pairs=len(src))
    arrays = (T, state, wait, last, std, reg_pred, reg_l, reg_r, d_lo, d_hi, anchor, nis_on,
              ap_ptr, ap_flat, pslot, pm, pc, pl, pd_,
              s_int, s_n, s_lag, tptr.astype(np.int64), t_slots, gptr.astype(np.int64), g_slots,
              oc_ptr, oc_dst, giveup, order)
    return arrays, info


@njit(cache=True)
def _gather(head, nxt, val, f, n):
    out = np.empty(n); k = head[f]; i = 0
    while k >= 0:
        out[i] = val[k]; i += 1; k = nxt[k]
    return out


@njit(cache=True)
def _own_flag(f, t, j, pw, reg_l, reg_r, anchor, nis_on, ap_ptr, ap_flat, d_lo, d_hi, pslot, pm, pl, pd_):
    if t == REGULAR:
        return reg_l[f] <= j + pw and reg_r[f] >= j - pw
    if not nis_on[f]:
        return False
    a = anchor[f]
    if t == APPRO:
        for q in range(ap_ptr[f], ap_ptr[f + 1]):
            x = a + ap_flat[q] + 1
            if x <= j + pw and x >= j - pw: return True
        return False
    if t == DENSE:
        return a + d_lo[f] + 1 <= j + pw and a + d_hi[f] + 1 >= j - pw
    if t == POSSIBLE or t == NEW_POSS:
        k = pslot[f]; n = pl[k]
        if n == 0: return False
        if pd_[k]:
            for q in range(n):
                x = a + pm[k, q] + 1
                if x <= j + pw and x >= j - pw: return True
            return False
        lo = pm[k, 0]; hi = pm[k, 0]
        for q in range(n):
            lo = min(lo, pm[k, q]); hi = max(hi, pm[k, q])
        return a + lo + 1 <= j + pw and a + hi + 1 >= j - pw
    return False


@njit(cache=True)
def _simulate(nF, ev_ptr, ev_f, ev_c, pw, adjust,
              T, state, wait, last, std, reg_pred, reg_l, reg_r, d_lo, d_hi, anchor, nis_on,
              ap_ptr, ap_flat, pslot, pm, pc, pl, pd_,
              s_int, s_n, s_lag, tptr, t_slots, gptr, g_slots, oc_ptr, oc_dst, giveup, order):
    cold = np.zeros(nF, np.int64); inv = np.zeros(nF, np.int64); invm = np.zeros(nF, np.int64)
    wastef = np.zeros(nF, np.int64)
    used_t = np.zeros(T_TEST, np.int64); waste_t = np.zeros(T_TEST, np.int64); mem_end = np.zeros(T_TEST, np.int64)
    stamp = np.full(nF, -1, np.int64); cnt = np.zeros(nF, np.int64)
    extra = np.full(nF, NONE, np.int64)
    nev = len(ev_f) + 1
    adp_val = np.empty(nev); adp_nxt = np.full(nev, -1, np.int64)
    adp_head = np.full(nF, -1, np.int64); adp_tail = np.full(nF, -1, np.int64); adp_len = np.zeros(nF, np.int64)
    nadp = 0; trunc = 0; wmt = 0
    nload = 0
    for f in range(nF):
        if state[f]: nload += 1
    for j in range(T_TEST):
        for e in range(ev_ptr[j], ev_ptr[j + 1]):
            stamp[ev_f[e]] = j; cnt[ev_f[e]] = ev_c[e]
        for oi in range(nF):
            f = order[oi]
            t = T[f]
            if stamp[f] == j:
                inv[f] += cnt[f]; invm[f] += 1; used_t[j] += 1
                if not state[f]:
                    cold[f] += 1; state[f] = True; nload += 1
                if wait[f] == -1 or wait[f] > 0:
                    if wait[f] != -1:
                        adp_val[nadp] = wait[f]
                        if adp_head[f] < 0: adp_head[f] = nadp
                        else: adp_nxt[adp_tail[f]] = nadp
                        adp_tail[f] = nadp; adp_len[f] += 1; nadp += 1
                    if adjust and adp_len[f] >= 5 and (t == REGULAR or t == DENSE or t == POSSIBLE or t == NEW_POSS or t == UNKNOWN):
                        w = _gather(adp_head, adp_nxt, adp_val, f, adp_len[f])
                        if t == REGULAR:
                            med = np.median(w)
                            if abs(med - reg_pred[f]) > max(1.0, std[f]):
                                reg_pred[f] = (reg_pred[f] + med) / 2
                        elif t == DENSE:
                            if abs(w.min() - d_lo[f]) > 1: d_lo[f] = w.min()
                            if abs(w.max() - d_hi[f]) > 1 and w.max() <= DENSE_UPPER_BOUND: d_hi[f] = w.max()
                        else:
                            u = np.unique(w); c = np.zeros(len(u), np.int64)
                            for x in w: c[np.searchsorted(u, x)] += 1
                            o = np.argsort(-c, kind="mergesort")
                            k = pslot[f]
                            if t == UNKNOWN:
                                nm = 0
                                for x in c:
                                    if x > 1: nm += 1
                                nm = min(5, nm)
                                if nm > 0:
                                    T[f] = NEW_POSS
                                    lo = u[o[0]]; hi = u[o[0]]
                                    for q in range(nm):
                                        pm[k, q] = u[o[q]]; pc[k, q] = c[o[q]]
                                        lo = min(lo, u[o[q]]); hi = max(hi, u[o[q]])
                                    pl[k] = nm; pd_[k] = hi - lo > DISCRETE_TH
                            elif pl[k] > 0 and c.max() >= max(2, pc[k, pl[k] - 1]):
                                for q in range(len(o)):
                                    val = u[o[q]]; cq = c[o[q]]
                                    present = False
                                    for r in range(pl[k]):
                                        if pm[k, r] == val: present = True
                                    if present: continue
                                    pos = 0
                                    while pos < pl[k] and pc[k, pos] > cq: pos += 1
                                    if pos >= pl[k]: continue
                                    if cq > pc[k, pos]:
                                        pc[k, pos] = cq; pm[k, pos] = val
                                    elif cq == pc[k, pos]:
                                        if pl[k] >= CAP_MODES:
                                            trunc += 1; continue
                                        for r in range(pl[k], pos, -1):
                                            pm[k, r] = pm[k, r - 1]; pc[k, r] = pc[k, r - 1]
                                        pm[k, pos] = val; pc[k, pos] = cq; pl[k] += 1
                                lo = pm[k, 0]; hi = pm[k, 0]
                                for q in range(pl[k]):
                                    lo = min(lo, pm[k, q]); hi = max(hi, pm[k, q])
                                pd_[k] = hi - lo > DISCRETE_TH
                wait[f] = 0; last[f] = j
                # new next-invocation prediction (uses the type at the start of this step, as in the notebook)
                if t == REGULAR:
                    s = min(std[f], EN_STD)
                    reg_l[f] = j + 1 + reg_pred[f] - s; reg_r[f] = j + 1 + reg_pred[f] + s; extra[f] = NONE
                elif t == APPRO or t == DENSE or t == POSSIBLE or t == NEW_POSS:
                    anchor[f] = j; nis_on[f] = True; extra[f] = NONE
                # f is a candidate of correlated targets
                for q in range(gptr[f], gptr[f + 1]):
                    sl = g_slots[q]; lag = s_lag[sl]
                    s_int[sl, 0, 0] = j; s_int[sl, 0, 1] = j + lag; n = 1
                    if t == REGULAR:
                        s_int[sl, 1, 0] = reg_l[f]; s_int[sl, 1, 1] = reg_r[f] + lag; n = 2
                    elif t == APPRO:
                        for r in range(ap_ptr[f], ap_ptr[f + 1]):
                            if n < CAP_SLOT:
                                x = j + ap_flat[r] + 1; s_int[sl, n, 0] = x; s_int[sl, n, 1] = x + lag; n += 1
                    elif t == DENSE:
                        s_int[sl, 1, 0] = j + d_lo[f] + 1; s_int[sl, 1, 1] = j + d_hi[f] + 1 + lag; n = 2
                    elif t == POSSIBLE or t == NEW_POSS:
                        k = pslot[f]
                        if pl[k] > 0:
                            if pd_[k]:
                                for r in range(pl[k]):
                                    if n < CAP_SLOT:
                                        x = j + pm[k, r] + 1; s_int[sl, n, 0] = x; s_int[sl, n, 1] = x + lag; n += 1
                            else:
                                lo = pm[k, 0]; hi = pm[k, 0]
                                for r in range(pl[k]):
                                    lo = min(lo, pm[k, r]); hi = max(hi, pm[k, r])
                                s_int[sl, 1, 0] = j + lo + 1; s_int[sl, 1, 1] = j + hi + 1 + lag; n = 2
                    s_n[sl] = n
                # online correlation for unseen functions
                for q in range(oc_ptr[f], oc_ptr[f + 1]):
                    g = oc_dst[q]
                    extra[g] = max(extra[g], j + 1 + pw)
            else:
                wait[f] = 1 if wait[f] == -1 else wait[f] + 1
                if state[f]:
                    wmt += 1; wastef[f] += 1; waste_t[j] += 1
                flag = False
                if t == REGULAR and reg_r[f] <= j:
                    reg_l[f] += reg_pred[f] + 1
                    reg_r[f] = reg_l[f] + 2 * min(std[f], EN_STD)
                if t == REGULAR or t == APPRO or t == DENSE or t == POSSIBLE or t == NEW_POSS or t == UNKNOWN:
                    flag = j <= extra[f] or _own_flag(f, t, j, pw, reg_l, reg_r, anchor, nis_on, ap_ptr, ap_flat, d_lo, d_hi, pslot, pm, pl, pd_)
                elif t == CORR:
                    flag = j <= extra[f]
                    if not flag:
                        for q in range(tptr[f], tptr[f + 1]):
                            sl = t_slots[q]
                            for r in range(s_n[sl]):
                                if s_int[sl, r, 0] <= j + pw and s_int[sl, r, 1] >= j - pw:
                                    flag = True; break
                            if flag: break
                if state[f] and not flag and wait[f] >= giveup[t]:
                    state[f] = False; nload -= 1
                elif flag and not state[f]:
                    state[f] = True; nload += 1
        mem_end[j] = nload
    return cold, inv, invm, wastef, used_t, waste_t, mem_end, wmt, trunc


@njit(cache=True)
def _test_events(ip, M, C):
    """Test-day calls grouped by minute (function order inside a minute)."""
    mp = np.zeros(T_TEST + 1, np.int64)
    for q in range(len(M)):
        if M[q] >= TR: mp[M[q] - TR + 1] += 1
    for m in range(T_TEST): mp[m + 1] += mp[m]
    pos = mp[:-1].copy(); ef = np.empty(mp[-1], np.int64); ec = np.empty(mp[-1], np.int64)
    for f in range(len(ip) - 1):
        for q in range(ip[f], ip[f + 1]):
            if M[q] >= TR:
                m = M[q] - TR; ef[pos[m]] = f; ec[pos[m]] = C[q]; pos[m] += 1
    return mp, ef, ec


def simulate(tr, typ, forget, candi, cfg):
    arrays, info = prepare(tr, typ, forget, candi, cfg)
    ev_ptr, ev_f, ev_c = _test_events(tr.ip, tr.M, tr.C)
    t0 = time.time()
    out = _simulate(tr.nF, ev_ptr, ev_f, ev_c, cfg["prewarm"], cfg["adjust"], *arrays)
    secs = time.time() - t0
    cold, inv, invm, wastef, used_t, waste_t, mem_end, wmt, trunc = out
    T_final = arrays[0]
    te = tr.in_test
    res = dict(cold=cold[te], inv=inv[te], invm=invm[te], waste=wastef[te], type=T_final[te], func_idx=np.where(te)[0],
               used_t=used_t, waste_t=waste_t, mem_end_t=mem_end, wmt=wmt, sim_seconds=secs, modes_truncated=trunc, **info)
    return res


# ============================================================ driver with caching
DEFAULT = dict(forgetting=True, use_corr=True, use_corr_sim=True, online_corr="code", adjust=True,
               prewarm=2, giveup_scale=1, candidates="compute")


def run(tr, cfg=None, cache_dir=None, log=print):
    cfg = dict(DEFAULT, **(cfg or {}))
    key = f"forget{int(cfg['forgetting'])}_corr{int(cfg['use_corr'])}_{os.path.basename(str(cfg['candidates']))}"
    path = os.path.join(cache_dir, f"spes_cat_{key}.pkl") if cache_dir else None
    if path and os.path.exists(path):
        typ, forget, before, candi, typ_final = pickle.load(open(path, "rb"))
    else:
        typ, forget, before = categorize(tr, cfg["forgetting"], log)
        if cfg["candidates"] == "compute":
            candi = find_candidates(tr, typ, forget, log)
        else:
            candi = load_candidates_json(tr, cfg["candidates"]); log(f"loaded {len(candi):,} candidate lists from {cfg['candidates']}")
        typ_final = assign(tr, typ, forget, candi, cfg["use_corr"], log)
        if path:
            os.makedirs(cache_dir, exist_ok=True); pickle.dump((typ, forget, before, candi, typ_final), open(path, "wb"))
    res = simulate(tr, typ_final, forget, candi, cfg)
    res["type_train"] = typ_final; res["type_first_pass"] = before; res["type_after_forget"] = typ
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir"); ap.add_argument("out")
    ap.add_argument("--cache", default=None, help="directory to cache categorization/assignment")
    ap.add_argument("--prewarm", type=int, default=2, help="theta_prewarm (minutes)")
    ap.add_argument("--giveup-scale", type=int, default=1, help="multiply every theta_givenup (Fig. 13b)")
    ap.add_argument("--no-forgetting", action="store_true"); ap.add_argument("--no-adjust", action="store_true")
    ap.add_argument("--no-corr", action="store_true", help="no 'correlated' type in training (Fig. 14 w/o Corr)")
    ap.add_argument("--online-corr", default="code", choices=["code", "paper", "off"])
    ap.add_argument("--candidates", default="compute", help="'compute' or path to the authors' mid-data/func_candi.json")
    a = ap.parse_args()
    tr = Trace(a.data_dir)
    cfg = dict(prewarm=a.prewarm, giveup_scale=a.giveup_scale, forgetting=not a.no_forgetting, adjust=not a.no_adjust,
               use_corr=not a.no_corr, use_corr_sim=not a.no_corr, online_corr=a.online_corr, candidates=a.candidates)
    res = run(tr, cfg, a.cache)
    np.savez_compressed(a.out, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                        meta=json.dumps({k: v for k, v in res.items() if not isinstance(v, np.ndarray)}))
    print(json.dumps(summarize(res), indent=1))
