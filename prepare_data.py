"""Convert the 14 daily invocation CSVs of the Azure Functions 2019 trace into a compact sparse file.

usage: python prepare_data.py <extracted_dataset_dir> <out_dir>
writes <out_dir>/trace.npz (per-function sorted minutes + counts) and <out_dir>/meta.csv
(owner, app, func, triggers). Functions are numbered in order of first appearance, which is the
iteration order of the authors' notebook (needed to reproduce it exactly). Peak RAM ~3 GB.
"""
import os, sys, time
import numpy as np
import pandas as pd


def main(raw, out):
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "_days"); os.makedirs(tmp, exist_ok=True)
    key2idx, owners, apps, funcs, trig = {}, [], [], [], []
    t0 = time.time()
    for d in range(14):
        fn = os.path.join(raw, "invocations_per_function_md.anon.d%02d.csv" % (d + 1))
        df = pd.read_csv(fn)
        o, a, f, tr = (df[c].to_numpy() for c in ["HashOwner", "HashApp", "HashFunction", "Trigger"])
        mat = df.iloc[:, 4:].to_numpy(dtype=np.int64)
        assert mat.shape[1] == 1440, fn
        g = np.empty(len(df), np.int64)
        for i in range(len(df)):
            j = key2idx.get(f[i])
            if j is None:
                j = len(key2idx); key2idx[f[i]] = j
                owners.append(o[i]); apps.append(a[i]); funcs.append(f[i]); trig.append({tr[i]})
            else:
                trig[j].add(tr[i])
            g[i] = j
        del df
        r, c = np.nonzero(mat)
        v = mat[r, c]; del mat
        key = g[r] * 1440 + c
        o2 = np.argsort(key, kind="stable"); key, v = key[o2], v[o2]
        uk, st = np.unique(key, return_index=True)
        v = np.add.reduceat(v, st)
        np.savez(os.path.join(tmp, f"d{d:02d}.npz"), g=(uk // 1440).astype(np.int32),
                 m=(uk % 1440 + d * 1440).astype(np.int32), v=np.minimum(v, 2**31 - 1).astype(np.int32))
        print(f"day {d + 1:2d}: {len(uk):>9,} busy function-minutes, {len(key2idx):,} functions so far ({time.time() - t0:.0f}s)", flush=True)
    nF = len(key2idx)
    cnt = np.zeros(nF, np.int64)
    for d in range(14):
        cnt += np.bincount(np.load(os.path.join(tmp, f"d{d:02d}.npz"))["g"], minlength=nF)
    ip = np.zeros(nF + 1, np.int64); ip[1:] = np.cumsum(cnt)
    M = np.empty(ip[-1], np.int32); C = np.empty(ip[-1], np.int32); pos = ip[:-1].copy()
    for d in range(14):
        z = np.load(os.path.join(tmp, f"d{d:02d}.npz"))
        g, m, v = z["g"], z["m"], z["v"]
        ug, st, ln = np.unique(g, return_index=True, return_counts=True)
        dst = np.repeat(pos[ug] - st, ln) + np.arange(len(g))
        M[dst] = m; C[dst] = v; pos[ug] += ln
    np.savez(os.path.join(out, "trace.npz"), indptr=ip, minute=M, count=C)
    pd.DataFrame({"owner": owners, "app": apps, "func": funcs,
                  "triggers": ["|".join(sorted(s)) for s in trig]}).to_csv(os.path.join(out, "meta.csv"), index=False)
    for d in range(14):
        os.remove(os.path.join(tmp, f"d{d:02d}.npz"))
    os.rmdir(tmp)
    print(f"done: {nF:,} functions, {ip[-1]:,} busy function-minutes, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
