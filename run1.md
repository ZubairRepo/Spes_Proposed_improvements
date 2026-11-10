Invoke-WebRequest -UseBasicParsing `
  -Uri "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz" `
  -OutFile "azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"


     python prepare_data.py azure2019 data         # ~1 min
   python run_all.py data results --workers 2    # ~25 min, about 5 GB RAM
   python plots.py data results figures          # ~2 min

   It follows the paper’s setup: learn on days 1–12, simulate days 13–14.

What’s included

spes.py: a direct port of the authors’ released notebook, with the same rules and constants, compiled with numba so one run takes about 1.5 minutes.
baselines.py: fixed keep-alive, Hybrid (per function and per app), Defuse and FaaSCache. None of these have public code, so they are my reimplementations.
run_all.py: 19 runs: SPES, the 5 baselines, the θ_prewarm sweep, θ_givenup ×2–×5, and the four ablations.
plots.py: Figs. 3–15, plus the function-type chart and summary.md. Ready-made figures (PNG and PDF) and results are already in the zip.




How to run (after run_all.py, which produces the SPES reference):

python run_proposed.py data results --workers 2    # ~35 min
python plots_proposed.py results figures            # figures/figP1…P5 + summary_proposed.md

Single run:

python proposed.py data out.npz --lam 0.003 --calib results/proposed_calib.npz

Add --raw to turn off calibration, and --no-clock / --no-sibling to drop those signals. A larger λ uses less memory and gives more cold starts.