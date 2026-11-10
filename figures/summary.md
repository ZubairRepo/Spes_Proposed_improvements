| Run | Q3 CSR | Never-cold % | Always-cold % | Total cold starts | WMT (M fn-min) | Avg memory | EMCR % | Paper |
|---|---|---|---|---|---|---|---|---|
| SPES | 0.0909 | 59.02 | 7.51 | 216,271 | 24.61 | 15,133 | 43.6 | q3_csr=0.108, zero_cold_pct=57.99, always_cold_pct=<8, wmt=21521200, mem_avg_notebook=13982.1, emcr=46.32 |
| FaaSCache | 1.0000 | 43.52 | 29.66 | 186,253 | 44.41 | 22,006 | 29.9 | zero_cold_pct=52.59 |
| Defuse | 0.1667 | 47.96 | 14.43 | 198,389 | 58.05 | 26,744 | 24.4 | q3_csr=0.215 |
| Hybrid-App (HA) | 0.0177 | 53.29 | 5.18 | 38,011 | 83.51 | 35,584 | 18.4 |  |
| Hybrid-Func (HF) | 0.0741 | 48.42 | 8.10 | 78,864 | 28.73 | 16,563 | 39.6 |  |
| Fixed keep-alive | 1.0000 | 25.61 | 34.88 | 721,949 | 24.88 | 15,227 | 43.3 | zero_cold_pct=25.61, q3_csr=1.0 |
| SPES θ_prewarm=0 | 0.9958 | 26.89 | 23.12 | 4,561,345 | 15.91 | 12,113 | 54.7 |  |
| SPES θ_prewarm=1 | 0.1241 | 51.92 | 7.72 | 278,732 | 19.63 | 13,402 | 49.0 |  |
| SPES θ_prewarm=3 | 0.0833 | 60.16 | 7.37 | 192,851 | 28.09 | 16,341 | 40.4 |  |
| SPES θ_prewarm=4 | 0.0769 | 61.47 | 7.30 | 176,157 | 30.81 | 17,283 | 38.2 |  |
| SPES θ_prewarm=5 | 0.0714 | 62.10 | 7.23 | 165,023 | 31.95 | 17,681 | 37.3 |  |
| SPES θ_givenup×2 | 0.0862 | 59.65 | 7.36 | 199,619 | 27.60 | 16,172 | 40.8 |  |
| SPES θ_givenup×3 | 0.0833 | 60.33 | 7.26 | 190,781 | 29.89 | 16,964 | 38.9 |  |
| SPES θ_givenup×4 | 0.0833 | 60.52 | 7.18 | 185,960 | 30.67 | 17,237 | 38.4 |  |
| SPES θ_givenup×5 | 0.0812 | 60.63 | 7.13 | 181,561 | 31.43 | 17,500 | 37.7 |  |
| SPES w/o Corr | 0.1818 | 54.56 | 8.98 | 245,263 | 17.44 | 12,644 | 52.3 |  |
| SPES w/o Online-Corr | 0.0939 | 58.79 | 7.64 | 216,924 | 24.55 | 15,110 | 43.6 |  |
| SPES w/o forgetting | 0.0907 | 59.25 | 7.51 | 214,981 | 24.74 | 15,176 | 43.4 |  |
| SPES w/o adjusting | 0.0931 | 59.04 | 7.55 | 226,251 | 24.58 | 15,124 | 43.6 |  |

HA cold-start rates are per application (as in the paper). Memory = functions loaded per test minute; the paper's 13,982 uses the notebook's end-of-minute count (ours: 15,058).
