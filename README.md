# SPES reproduction: simulation + every figure of the paper

Code to reproduce **"SPES: Towards Optimizing Performance-Resource Trade-Off for Serverless Functions"**
(Lee et al., ICDE 2024) on the Azure Functions 2019 trace. It covers SPES, the four baselines, the
parameter sweeps and the ablations, and produces each figure of the paper's evaluation and characterization.

| File | What it does |
|---|---|
| `prepare_data.py` | 14 daily CSVs → compact sparse trace (`trace.npz`, `meta.csv`) |
| `spes.py` | SPES: categorization, forgetting, correlation candidates, assignment, provisioning simulation |
| `baselines.py` | Fixed keep-alive, Hybrid histogram (per function / per app), Defuse, FaaSCache |
| `run_all.py` | Runs all 19 experiments (in parallel) → `results/*.npz`, `results/summary.json` |
| `plots.py` | Figures 3–15 (+ a type-distribution figure) and `summary.md` |
| `common.py` | Trace loader, run-length encoding, metrics |

## 1. Install

Python 3.9+ and:

```bash
pip install numpy pandas numba matplotlib
```

## 2. Get the dataset (≈140 MB download, 2 GB extracted)

The trace is Microsoft's public **Azure Functions Trace 2019** (the one used by SPES and by "Serverless in the Wild").

```bash
mkdir -p azure2019 && cd azure2019
# official mirror on GitHub (Azure/AzurePublicDataset release):
wget https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz
# (or the original Azure blob:
#  https://azurecloudpublicdataset2.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz)
tar -xJf azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz
cd ..
```

Only the 14 files `invocations_per_function_md.anon.d01.csv … d14.csv` are used (per-minute call counts of each function).

## 3. Run

```bash
python prepare_data.py azure2019 data                 # ~1 min, writes data/trace.npz + data/meta.csv
python run_all.py data results --workers 2            # ~25 min on 2 cores, peak ~5 GB RAM
python plots.py data results figures                  # ~2 min, writes figures/*.png|pdf + figures/summary.md
```

Single runs:

```bash
python spes.py data results/spes.npz --cache results/cache          # default SPES (θ_prewarm=2)
python spes.py data out.npz --prewarm 4                              # Fig. 13a point
python spes.py data out.npz --giveup-scale 3                         # Fig. 13b point
python spes.py data out.npz --no-corr | --online-corr off | --no-forgetting | --no-adjust   # ablations
python spes.py data out.npz --candidates SPES/mid-data/func_candi.json   # use the authors' candidate file
python baselines.py data fixed|hybrid_f|hybrid_a|defuse|faascache out.npz [--cap N]
```

Each script prints the metrics: 75th-percentile cold-start rate (Q3-CSR), % of functions never cold,
% always cold (CSR = 1), total cold starts, wasted memory time (WMT), average memory and EMCR.

## 4. Experimental setup (same as the paper)

* Train on days 1–12, simulate days 13–14 (minute resolution).
* Every execution finishes within its minute; every cold start costs the same; every instance uses the same memory.
* **Cold-start rate (CSR)** of a function = cold-start minutes ÷ invocations. HA is measured per application, as in the paper.
* **WMT** = function-minutes loaded but not invoked. **Memory** = functions loaded per minute.
  **EMCR** = mean over minutes of (invoked ÷ loaded).
* FaaSCache's memory cap = SPES's peak memory, as in the paper.

## 5. Figures produced

| File | Paper figure |
|---|---|
| `fig03_invocation_distribution` | Fig. 3 — distribution of invocations per function |
| `fig04_concept_shift` | Fig. 4 — concept shifts in invocation patterns |
| `fig05_trigger_types` | Fig. 5 — trigger types (incl. "combination") |
| `fig06_temporal_locality` | Fig. 6 — bursts of infrequent functions |
| `fig07_function_types` | type counts: first pass / after forgetting / final (the notebook's pie charts) |
| `fig08_csr_cdf` | Fig. 8 — CDF of cold-start rate, SPES vs baselines |
| `fig09_memory_always_cold` | Fig. 9 — (a) memory normalized to SPES, (b) always-cold % |
| `fig10_csr_per_type` | Fig. 10 — average CSR per SPES type |
| `fig11_wmt_emcr` | Fig. 11 — (a) WMT normalized to SPES, (b) EMCR |
| `fig12_wmt_ratio_per_type` | Fig. 12 — WMT per invocation, per type |
| `fig13_tradeoff` | Fig. 13 — θ_prewarm sweep and θ_givenup ×1..5 |
| `fig14_correlation_ablation` | Fig. 14 — w/o Corr, w/o Online-Corr |
| `fig15_adaptivity_ablation` | Fig. 15 — w/o forgetting, w/o adjusting |

## 6. How faithful is the SPES port?

`spes.py` is a line-by-line port of the authors' released notebook
([github.com/BEbillionaireUSD/SPES](https://github.com/BEbillionaireUSD/SPES), `run.ipynb` + `cal_corr.py`)
into compiled (numba) code: same constants, same rules, same order of operations. Checks against the notebook's own printed output:

| Check | Authors' notebook output | This port |
|---|---|---|
| First pass: always-warm / regular / appro-regular / dense / successive | 1,839 / 16,193 / 6,242 / 405 / 1,683 | **identical** |
| Forgetting, days 1–5 | 169, 171, 218, 230, 269 | **identical** |
| Forgetting, day 6 | 3 | 92 (the authors' verbatim code also gives 92, see note 1) |
| Functions still unknown after assignment | 10,223 | **10,223** |
| Fixed 10-min keep-alive: never-cold functions | 25.61% (paper) | **25.61%** |
| Fixed keep-alive WMT implied by the paper (SPES −13.51%) | 24.88 M | **24.88 M** |

SPES itself (days 13–14):

| Metric | Paper / notebook | Port (code as written) | Port + authors' `func_candi.json` |
|---|---|---|---|
| 75th-percentile cold-start rate | 0.108 | 0.091 | 0.096 |
| Functions never cold | 57.99% | 59.02% | 58.61% |
| Functions always cold | < 8% | 7.51% | 7.64% |
| Wasted memory time | 21.52 M | 24.61 M | 22.83 M |
| Avg memory (notebook's end-of-minute count) | 13,982 | 15,058 | 14,439 |
| EMCR | 46.32% | 43.6% | 45.4% |

The remaining gap comes from the notebook's out-of-order execution (note 1), which left fewer functions in the
"correlated" type; correlated functions are pre-loaded whenever a partner fires, so more of them means more memory.

**What matches the paper:** SPES has far fewer cold starts than fixed keep-alive, FaaSCache and Defuse, and uses less
memory and wastes less than every baseline except fixed keep-alive (about equal); memory and cold starts trade off almost
linearly in θ_prewarm, with θ_prewarm = 2 below the fitted line; larger θ_givenup buys little (Q3 0.091 → 0.081 for +16%
memory); removing correlation doubles Q3-CSR (0.091 → 0.182) and removing online correlation changes it slightly.

**What does not:** with the original Hybrid settings (4-hour fallback keep-alive), Hybrid per function (Q3 0.074) and per
application (0.018) have *fewer* cold starts than SPES, at 1.17× and 3.39× SPES's wasted memory; the SPES paper's
re-implementation of Hybrid must have been much weaker. Forgetting has no measurable effect here and adjusting a small one
(Q3 0.091 vs 0.093). Correlation costs memory (w/o Corr: 29% less WMT) rather than saving it.

### Port notes (where the notebook cannot be run literally)

1. **Saved outputs vs. code.** The notebook's execution counters show cells were re-run out of order, so a few printed
   counts come from a state that the code, run top to bottom, does not produce. Running the authors' own
   `re_categorzing` (verbatim) on the forgetting-day-6 functions agrees with this port on all 92 of them (notebook printed 3),
   and the notebook validated only 7,077 "correlated" candidates where a top-to-bottom run validates ~17,900.
   This port follows the code as written.
2. **Candidate file.** The authors' `mid-data/func_candi.json` was produced before forgetting and with fewer cross-app
   candidates than `cal_corr.py` yields; 99.9% of its pairs are reproduced here, and every extra pair here passes the
   authors' own `cal_correlation` thresholds. Use `--candidates func_candi.json` to run with their file instead.
3. **Unseen functions' first waiting time** is `None` in the notebook and would crash `np.unique`/`np.median` once a
   function reaches 5 waits; it is skipped here.
4. **Online correlation direction.** The released code pre-loads the *candidates* when an unseen function is called
   (`--online-corr code`, default); the paper describes the reverse (`--online-corr paper`).
5. **Order inside a minute.** The notebook iterates a Python `set` (random order) and shuffles correlated functions;
   here non-correlated functions come first in index order, then correlated ones (deterministic).
6. Ties between equally frequent waiting times are broken by the smaller value (stable sort).

## 7. Baselines (re-implemented; no public code)

SPES also re-implemented these. Parameters follow each original paper where it states them:

* **Fixed keep-alive**: 10 minutes after each call.
* **Hybrid histogram** (Shahrad et al., ATC'20): 1-minute bins over 4 hours; histogram used when the CV of its bin counts
  > 2; pre-warm = 5th percentile −10%, keep-alive up to the 99th percentile +10%; otherwise standard keep-alive of
  4 hours (the histogram range). Out-of-range idle times (>4 h) for most calls → time-series mode with 15% margins;
  **ARIMA is replaced by the mean of the last 10 idle times**. Run per function (HF) and per application (HA).
* **Defuse** (Shen et al., ICDCS'21): strong dependencies = predictable functions of the same owner called in the same
  minute in ≥ 80% of the rarer one's minutes (≥ 3 times); weak dependencies = top-1 PPMI link from each unpredictable
  function to a predictable one; union-find → dependency sets; each set uses histogram windows at the 5th/95th percentile,
  or a fixed 10-minute keep-alive when its histogram is not representative. "Predictable" = representative training
  histogram. The paper's slides do not give its support thresholds, so these values are our choice.
* **FaaSCache** (Fuerst & Sharma, ASPLOS'21): Greedy-Dual-Size-Frequency keep-alive (priority = clock + frequency;
  equal cost and size) under a cap equal to SPES's peak memory.

Because the baselines are re-implementations, their absolute numbers differ from the SPES paper's.

## 8. Results included

`figures/` holds the figures produced from the Azure trace by exactly these scripts, and `figures/summary.md` /
`results/summary.json` the metrics of all 19 runs.

## 9. Runtime

On 2 cores: data preparation ~1 min; SPES categorization + candidates + assignment ~2.5 min (cached afterwards);
one SPES simulation of the two test days ~1.5 min; all baselines ~1 min together.

## 10. Proposed policy (our improvement over SPES)

| File | What it does |
|---|---|
| `proposed.py` | the policy + its simulator (same assumptions and metrics as SPES) |
| `run_proposed.py` | calibration table, threshold sweeps, and the two matched comparison points |
| `plots_proposed.py` | Figs. P1–P5 and `summary_proposed.md` |

**The rule.** Every minute, for every function, estimate p = chance of a call in the next minute and keep the function
loaded iff p ≥ λ/(1+λ). One knob (λ) replaces SPES's nine types and ~15 thresholds.

1. **Idle-time chance** from the function's own history: of the times it had been idle *a* minutes, how often did it wake up
   at the next minute — (gaps of exactly *a* + prior) / (gaps of at least *a* + 1). The prior is the same estimate pooled over
   functions with the same trigger and call frequency. Counts keep updating while simulating.
2. **Clock chance**: share of training days on which it ran at this exact minute of the day (≥ 2 hits).
3. **Sibling chance**: a function of the same app just started, and this one followed within 2 minutes in ≥ 20% of ≥ 3 past cases.
   Raw p = the highest of the three.
4. **Calibration**: replay days 11–12 with a model trained on days 1–10 (the test days 13–14 are never used), count how
   often a call really came in each situation, and use p = (calls + 20 × raw p) / (minutes + 20) for the matching cell.

Run (after `run_all.py`, which produces the SPES reference `results/spes.npz`):

```bash
python run_proposed.py data results --workers 2      # ~35 min: calibration, 21 sweep runs, 2 bisections
python plots_proposed.py results figures             # figures/figP*.png|pdf + figures/summary_proposed.md
```

Single runs:

```bash
python proposed.py data out.npz --lam 0.003 --calib results/proposed_calib.npz      # proposed policy
python proposed.py data out.npz --lam 0.003 --raw                                  # without calibration
python proposed.py data out.npz --lam 0.003 --raw --no-clock --no-sibling --no-jitter   # probability rule only
python proposed.py data out.npz --lam 0.03 --mu 10 --calib results/proposed_calib.npz   # with the rare-function boost
```

Larger λ → less memory, more cold starts. `run_proposed.py` finds λ automatically for the two comparison points:
*same memory as SPES* and *same quality as SPES* (never-cold %, Q3-CSR and always-cold % all at least as good).

Results on days 13–14 (same simulator and metrics as the SPES port):

| Policy | Avg memory | Wasted memory | Never-cold | Q3 cold-start rate | Always-cold | Total cold starts |
|---|---|---|---|---|---|---|
| Fixed keep-alive 10 min | 15,227 | 24.88 M | 25.6% | 1.000 | 34.9% | 721,949 |
| SPES (port) | 15,133 | 24.61 M | 59.0% | 0.091 | 7.5% | 216,271 |
| **Proposed, same memory as SPES** (λ = 0.0019) | 14,995 | 24.21 M | **71.0%** | **0.004** | **5.2%** | **21,531** |
| **Proposed, same quality as SPES** (λ = 0.022, μ = 10) | **10,523** | **11.33 M** | 59.1% | 0.049 | 6.2% | 88,600 |

* Same memory: 12 more percentage points of never-cold functions, 96% lower Q3 cold-start rate, 90% fewer cold starts.
* Same quality: 30% less memory and 54% less wasted memory than SPES.
* Fig. P4 breaks the gain down at SPES's memory: the probability rule alone 59.0% → 67.0% never-cold
  (216k → 41k cold starts), clock & sibling → 71.3% (30k), calibration → 70.6% and 23k cold starts.
