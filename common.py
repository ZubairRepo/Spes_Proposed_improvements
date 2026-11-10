"""Shared helpers: trace loading, run-length encoding of per-minute call series, metrics.

Time axis: minute 0 = first minute of day 1. Train = days 1-12 [0, 17280), test = days 13-14 [17280, 20160).
"""
import os
import numpy as np
import pandas as pd
from numba import njit

DAY = 1440
TR = 12 * DAY          # first test minute
END = 14 * DAY         # end of trace
T_TEST = END - TR      # 2880 test minutes

# function types (same numbering as the authors' notebook)
UNKNOWN, WARM, REGULAR, APPRO, DENSE, SUCCESSIVE, PULSED, POSSIBLE, CORR, NEW_POSS = range(10)
TYPE_NAMES = ["Unknown", "Always warm", "Regular", "Appro-regular", "Dense", "Successive",
              "Pulsed", "Possible", "Correlated", "Newly-possible"]
TRIGGERS = ["http", "timer", "queue", "event", "orchestration", "storage", "others"]


class Trace:
    """Sparse trace: for function f, minutes M[ip[f]:ip[f+1]] (sorted) with call counts C[...]."""

    def __init__(self, data_dir):
        z = np.load(os.path.join(data_dir, "trace.npz"))
        self.ip = z["indptr"].astype(np.int64)
        self.M = z["minute"].astype(np.int32)    # int32 keeps the 140M-entry trace at ~1.1 GB
        self.C = z["count"].astype(np.int32)
        self.meta = pd.read_csv(os.path.join(data_dir, "meta.csv"))
        self.nF = len(self.ip) - 1
        self.owner = pd.factorize(self.meta["owner"])[0].astype(np.int64)
        self.app = pd.factorize(self.meta["app"])[0].astype(np.int64)
        # trigger sets as bit masks
        bits = {t: 1 << i for i, t in enumerate(TRIGGERS)}
        self.trig = np.array([sum(bits.get(t, bits["others"]) for t in str(s).split("|")) for s in self.meta["triggers"]], np.int64)
        self.hash = self.meta["func"].to_numpy()
        n_tr, n_te = count_windows(self.ip, self.M)
        self.in_train = n_tr > 0
        self.in_test = n_te > 0

    def window(self, f, a, b):
        lo, hi = self.ip[f], self.ip[f + 1]
        ms = self.M[lo:hi]
        i, j = np.searchsorted(ms, a), np.searchsorted(ms, b)
        return ms[i:j], self.C[lo + i: lo + j]


@njit(cache=True)
def count_windows(ip, M):
    nF = len(ip) - 1
    a = np.zeros(nF, np.int64); b = np.zeros(nF, np.int64)
    for f in range(nF):
        for k in range(ip[f], ip[f + 1]):
            if M[k] < TR: a[f] += 1
            else: b[f] += 1
    return a, b


@njit(cache=True)
def rle(ms, cs, a, b):
    """Run-length encoding of the 0/1 call series on [a, b).

    Returns (wt, at, an, first_invoked, last_invoked), matching the notebook's
    conj_seq_lst(arr) / conj_seq_lst(arr, True) / cal_AN_seq(arr):
      wt: lengths of idle runs in time order (leading, interior..., trailing)
      at: lengths of busy runs, an: calls in each busy run.
    """
    n = len(ms)
    if n == 0:
        wt = np.empty(1, np.int64); wt[0] = b - a
        return wt, np.empty(0, np.int64), np.empty(0, np.int64), False, False
    starts = [np.int64(ms[0])]; ends = [np.int64(0)]; ends.pop(); sums = [np.int64(cs[0])]
    for i in range(1, n):
        if ms[i] == ms[i - 1] + 1:
            sums[-1] += cs[i]
        else:
            ends.append(np.int64(ms[i - 1])); starts.append(np.int64(ms[i])); sums.append(np.int64(cs[i]))
    ends.append(np.int64(ms[n - 1]))
    r = len(starts)
    first_inv = starts[0] == a
    last_inv = ends[r - 1] == b - 1
    nw = r - 1 + (0 if first_inv else 1) + (0 if last_inv else 1)
    wt = np.empty(nw, np.int64); k = 0
    if not first_inv:
        wt[k] = starts[0] - a; k += 1
    for i in range(r - 1):
        wt[k] = starts[i + 1] - ends[i] - 1; k += 1
    if not last_inv:
        wt[k] = b - 1 - ends[r - 1]; k += 1
    at = np.empty(r, np.int64); an = np.empty(r, np.int64)
    for i in range(r):
        at[i] = ends[i] - starts[i] + 1; an[i] = sums[i]
    return wt, at, an, first_inv, last_inv


def top_modes(vals, k):
    """np.unique + argsort(-counts) with ties broken by smaller value (stable). Returns (values, counts) of the top k."""
    u, c = np.unique(vals, return_counts=True)
    o = np.argsort(-c, kind="stable")[:k]
    return u[o], c[o]


def summarize(res):
    """Paper metrics from per-function arrays of the test functions."""
    cold, inv = res["cold"].astype(float), res["inv"].astype(float)
    csr = cold / inv
    used = res["used_t"].astype(float); waste_t = res["waste_t"].astype(float)
    mem_t = used + waste_t
    out = dict(
        q3_csr=float(np.percentile(csr, 75)), q50_csr=float(np.percentile(csr, 50)), q90_csr=float(np.percentile(csr, 90)),
        zero_cold_pct=float((cold == 0).mean() * 100), always_cold_pct=float((csr >= 1.0).mean() * 100),
        total_cold=int(cold.sum()), wmt=int(res["wmt"]), mem_avg=float(mem_t.mean()), mem_max=float(mem_t.max()),
        emcr=float(np.mean(used / np.maximum(mem_t, 1)) * 100), n_funcs=int(len(csr)),
    )
    if "mem_end_t" in res:
        out["mem_avg_notebook"] = float(np.mean(res["mem_end_t"]))
    return out
