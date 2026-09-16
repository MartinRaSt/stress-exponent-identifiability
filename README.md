# Alpha-Sammon mapping: adaptive weighting, a scalable solver and a temporal extension

Source code for the paper *Identifiability of the stress exponent in weighted
multidimensional scaling*. The work generalises Sammon mapping to the power
family of weighted stresses `w_ij = (D_ij + eps_D)^(-alpha)`, gives a criterion
that says **when the exponent can be identified from the data at all**, and
compares the resulting family against dimensionality-reduction methods and graph
layouts.

**Authors**

- Martin Radvansky — <martin.radvansky@vsb.cz>, ORCID 0000-0002-8421-7658
- Martin Radvansky, Jr. — <martin.radvansky1@vsb.cz>, ORCID 0000-0003-0901-7397

Department of Computer Science, Faculty of Electrical Engineering and Computer
Science, VSB – Technical University of Ostrava, 17. listopadu 2172/15,
708 00 Ostrava-Poruba, Czech Republic.

---

## 1. What this repository contains — and what it does not

**Source code and documentation only.** Nothing that can be obtained by running
the code is kept under version control.

| Not in the repository | Why | How to obtain it |
|---|---|---|
| Public datasets (`src/data/`) | Third-party data under their own licences | Downloaded by the scripts — section 4 |
| Synthetic datasets | Generated deterministically from the configuration | Created when an experiment runs |
| Cached distance matrices (`src/data/graphs/cache/`, ~2.7 GB) | Intermediate result, recomputed on demand | Created when an experiment runs |
| Results (`results/`) | Output of the computations | By running the experiments — section 6 |
| Figures and the macro file of the paper | Generated from the results | `src/make_figures.*`, `src/run_export_numbers.*` |
| Virtual environment (`venv/`) | Installed from `requirements.txt` | Section 3 |

The manuscript itself is not part of this repository. Two scripts nevertheless
write outputs meant for it: `make_figures` copies the final PDF figures and
`export_numbers` writes the LaTeX macro file with every number quoted in the
paper. Both create the target directory (`clanek/`, Czech for *article*) on
demand next to `results/`; you can delete it if you only need the results.

An archive of the data and results in exactly the state that produced the
published version of the paper is deposited on Zenodo (see the data availability
statement in the paper for the DOI).

---

## 2. Quick start

```bash
# Linux / macOS
git clone https://github.com/MartinRaSt/stress-exponent-identifiability.git
cd stress-exponent-identifiability
python -m venv venv && ./venv/bin/python -m pip install -r requirements.txt
./src/run_exp0_smoke.sh            # checks that the whole pipeline runs
```

```bat
REM Windows
git clone https://github.com/MartinRaSt/stress-exponent-identifiability.git
cd stress-exponent-identifiability
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
src\run_exp0_smoke.bat
```

Every launcher exists in two forms: `*.bat` for Windows and `*.sh` for
Linux/macOS. Behaviour and arguments are identical.

---

## 3. Environment

- **Python 3.12**, packages pinned in `requirements.txt` (a `pip freeze`).
  The main ones: numpy 2.2, scipy 1.17, scikit-learn 1.8, pandas 2.3,
  matplotlib 3.10, networkx 3.6, umap-learn 0.5 (which also provides densMAP),
  pacmap, trimap, phate, pynndescent, numba, pytest.
- **PyTorch 2.14 + CUDA 12.6** — optional. The GPU is used only for dense SMACOF
  on large problems; without a GPU everything runs on the CPU (`sammon.device`
  in the configuration). Development and timing used an NVIDIA RTX 3070 Ti (8 GB).
- In the authors' environment `venv/` is a *conda prefix* environment, so the
  interpreter is `venv\python.exe` (Windows) or `venv/bin/python` (Linux).
  The `.sh` scripts honour the `PYTHON` environment variable if your interpreter
  lives elsewhere.

**Compute limits.** Parallelism is driven by the configuration
(`parallel.n_workers`, `parallel.blas_threads_per_worker` in
`src/common/config.yaml`); the default is 12 processes with one BLAS thread
each. Higher settings overheated the CPU on the test machine. Every script that
starts a long computation disables system sleep for the duration of the run and
restores the original setting afterwards, including after a failure.

---

## 4. Datasets: where they come from

The repository contains no data. Everything is downloaded or generated into
`src/data/` (git-ignored). The dataset registry is `src/datasets/registry.py`;
a table of everything available is produced by:

```
src\run_list_datasets.bat        # Windows
./src/run_list_datasets.sh       # Linux/macOS
```

This writes `results/tables/datasets.csv` and `datasets.tex` (name, n, d, number
of classes, kind, source, whether it is standardised, status).

Downloading is handled by `src/datasets/download.py`: every downloaded file is
verified by its SHA-256 checksum and recorded in `src/data/manifest.json`
together with the URL, the download date and the source. A missing or
unreachable source fails loudly; no data are ever silently substituted.

### 4.1 Vector datasets (`src/datasets/sklearn_sets.py`)

- **Bundled with scikit-learn**: `iris`, `digits`, `wine`, `breast_cancer`,
  `olivetti_faces` — nothing to download.
- **OpenML** via `sklearn.datasets.fetch_openml` with a local cache
  (`src/data/openml_cache/`): `mnist_784`, `fashion_mnist`, `coil20`, `cnae9`,
  `usps`, `letter`, `isolet`, `har` and others. An internet connection is needed
  on first use.
- Large datasets are stratified-subsampled to `n_max`
  (`src/datasets/subsample.py`), because Sammon- and MDS-type methods are O(n²).

### 4.2 Hold-out datasets (`src/datasets/uci_extra.py`)

An independent set used for the confirmatory test of the rule that predicts
`alpha`, so that overfitting to the datasets the rule was estimated on can be
ruled out. The sources are UCI and OpenML, downloaded by the same mechanism.

### 4.3 Synthetic datasets (`src/datasets/synthetic.py`)

Generated locally, nothing is downloaded: `swiss_roll`, `s_curve`, `sphere`,
`severed_sphere`, `helix`, `torus`, `two_moons`, `gaussian_clusters`,
`hierarchical_clusters`. All generators are seeded from the configuration and
are therefore deterministic.

### 4.4 Synthetic scenarios with known ground truth (`src/datasets/synthetic_truth.py`)

Used by the metric-fidelity task (Experiment 9): data with a **known** latent
geometry, against which one can measure how faithfully a method preserves the
true distances rather than only the ranking of neighbours.

### 4.5 Graph datasets (`src/datasets/graphs.py`)

- **Bundled with networkx**: `karate`, `les_miserables`, `florentine`.
- **Downloaded**: `dolphins`, `football`, `polbooks` from M. Newman's collection
  (`websites.umich.edu/~mejn/netdata/`) and `email_eu_core` from SNAP
  (`snap.stanford.edu/data/email-Eu-core.txt.gz`).
- Graphs are turned into distance matrices in two ways
  (`src/datasets/graph_distance.py`): shortest paths (geodesic distance) and
  **resistance distance**. Both are cached as `.npy` files under
  `src/data/graphs/cache/` because they are expensive to compute. **This cache is
  an intermediate result** — roughly 2.7 GB — and belongs neither in the
  repository nor in the archive; it is recreated automatically.

### 4.6 Temporal (dynamic) networks (`src/datasets/temporal.py`)

Contact networks from the **SocioPatterns** project (`primary_school`,
`hospital`, `high_school`, `invs13`, `sfhh`, `ht09`) and a temporal variant of
`email_eu_core`. They are downloaded from the public URLs listed in
`src/datasets/temporal.py`. Each time series is split into snapshots of
`time_bin_sec`; snapshots smaller than `min_snapshot_nodes` are skipped (for
example night-time gaps with no contacts).

> **Data licences.** Every source has its own terms of use. This repository does
> not redistribute any data, it only downloads it from the original locations.
> Please cite the original dataset authors when you use them.

---

## 5. Repository layout

```
src/
  common/       infrastructure (configuration, logging, checkpoints, parallelism)
  datasets/     loading, downloading and generating data
  methods/      adapters that put every compared method behind one interface
  sammon/       the proposed method: stress, weights, solvers, alpha selection, metrics
    solvers/    pseudo-Newton, SMACOF, SGD, sparse
  experiments/  experiments E0-E9 and the analyses over their results
  figures/      one script per figure of the paper
  tools/        helpers (runtime estimation)
  run_*.bat / run_*.sh   launchers
tests/          pytest (unit and regression tests)
```

### 5.1 `src/common/` — infrastructure

| File | Purpose |
|---|---|
| `config.py` | Loads `src/common/config.yaml`, resolves paths; mode-aware paths (`full`/`quick`/`smoke`) |
| `config.yaml` | Central configuration: paths, parallelism, downloads, methods, metrics, figures, method parameters |
| `checkpoint.py` | Resumability: one finished run is one row in `results/data/<experiment>_results.csv`; finished combinations are skipped on restart |
| `logging_utils.py` | Uniform logging to the console and to `results/logs/`, wall-clock timing |
| `parallel.py` | Order-preserving `ProcessPoolExecutor.map` (never `as_completed` — determinism), BLAS thread limits per worker, orphaned-worker watchdog |
| `progress.py` | Uniform progress bar with ETA |
| `seeding.py` | Uniform seeding of every source of randomness |
| `no_sleep_on.bat`, `no_sleep_off.bat`, `no_sleep.sh` | Disable system sleep during a run and restore it afterwards |

### 5.2 `src/datasets/` — data

| File | Purpose |
|---|---|
| `registry.py` | Central registry: name → loader, the common `Dataset` container |
| `download.py` | Downloads with SHA-256 verification and a manifest entry |
| `sklearn_sets.py` | Bundled and OpenML datasets, local cache |
| `uci_extra.py` | Hold-out candidates for the confirmatory test |
| `synthetic.py` | Synthetic manifolds and cluster datasets |
| `synthetic_truth.py` | Scenarios with a known latent geometry (E9) |
| `graphs.py` | Graph datasets (networkx and downloaded) |
| `graph_distance.py` | Graph → distance matrix: shortest paths and resistance distance, with caching |
| `temporal.py` | Dynamic networks (SocioPatterns, email-Eu-core), snapshot splitting |
| `subsample.py` | Stratified subsampling to `n_max` |
| `list_datasets.py` | Overview table of all datasets |

### 5.3 `src/methods/` — compared methods

All methods share the interface
`Method.fit_transform(X_or_D, kind, seed, n_components)` and are listed in
`registry.py`. Adapters: `pca`, `mds`, `isomap`, `lle`, `tsne`, `tsne_auto`,
`umap_`, `umap_auto`, `densmap`, `pacmap_`, `trimap_`, `phate_`, `spectral`,
`spring` (Fruchterman–Reingold), `kamada_kawai`, `sammon_classic` (a reference
implementation of Sammon 1969), `sammon_alpha` and `sammon_alpha_pred` (the
proposed method). `hyperparam_selection.py` provides fairly tuned baselines
(`tsne_auto`, `umap_auto`) that get the same tuning budget as our method.

### 5.4 `src/sammon/` — the proposed method

| File | Purpose |
|---|---|
| `weights.py` | The weight family `w_ij = (D_ij + eps_D)^(-alpha)` |
| `stress.py` | Stress `E_alpha`, gradient, diagonal Hessian, optimal scaling; numpy and torch variants |
| `estimator.py` | `SammonAlpha` — a scikit-learn style estimator over all solvers |
| `init.py` | Initialisation of `Y0`: random, PCA, classical (Torgerson) MDS |
| `solvers/newton.py` | Pseudo-Newton solver (a generalisation of Sammon 1969) |
| `solvers/smacof.py` | SMACOF majorisation for general weights (Guttman transform), CPU and GPU |
| `solvers/sgd.py` | Stochastic gradient descent and its stabilised variant |
| `solvers/sparse.py` | Sparse stress model with pivots |
| `alpha_selection.py` | Grid search for alpha with an independent validation metric |
| `alpha_predict.py` | **Predicts alpha from distance concentration** without a single tuning run |
| `temporal.py` | Temporal Sammon: stability regularisation between snapshots |
| `dynamic_tsne.py` | The external dynamic t-SNE baseline (Rauber et al. 2016) |
| `metrics.py` | Embedding quality metrics (stress, Shepard, trustworthiness, AUC R_NX, …) |
| `cluster_geometry.py` | Preservation of cluster sizes and inter-cluster distances |
| `graph_metrics.py` | Graph-specific layout metrics |
| `temporal_metrics.py` | Stability and quality of dynamic embeddings |
| `truth_metrics.py` | Metric fidelity against a known latent ground truth |
| `prop2_check.py` | The numerical part of the empirical test of Proposition 2 |
| `device.py` | CPU/CUDA selection and the largest `n` that fits in GPU memory |

### 5.5 `src/experiments/` — experiments

| Experiment | Script | Question |
|---|---|---|
| E0 | `exp0_smoke.py` | Smoke test of the whole pipeline on small datasets |
| E1 | `exp1_dr_benchmark.py` | Main benchmark: all datasets × all methods |
| E1b | `exp1_cluster_geometry.py` | Cluster geometry over the embeddings already stored by E1 |
| E1c | `exp1_regime_stratified.py` | E1 stratified by distance-concentration regime |
| E1d | `exp1_holdout_confirmatory.py` | Confirmatory test of the rule on an independent hold-out set |
| E2 | `exp2_solver_scaling.py` | Solvers: SGD convergence and scaling in `n` (CPU vs GPU) |
| E3 | `exp3_graph_layout.py` | Graph layouts under two input distances (paths, resistance) |
| E4 | `exp4_temporal.py` | Temporal Sammon on dynamic networks, `lambda` grid |
| E4b | `exp4_neighbor_metrics.py` | Post-hoc neighbourhood-preservation metrics for E4 |
| E4c | `exp4_relative.py` | Relative change of stability and quality against `lambda=0` |
| E5 | `exp5_ablation.py` | Ablation: alpha × initialisation × `eps_D` quantile |
| E6 | `exp6_alpha_curves.py` | Alpha curves on a fine grid (the basis of the prediction rule) |
| E7 | `exp7_rank_weights.py` | Rank-based instead of distance-based weights (a negative result) |
| E8 | `exp8_prop2_check.py` | Empirical test of Proposition 2 over the stored embeddings |
| E9 | `exp9_metric_fidelity.py` | Metric fidelity against a known ground truth |

Analyses and helpers: `dataset_properties.py` (properties of the inputs,
including distance concentration), `fit_alpha_rule.py` (derivation of the
prediction rule), `pareto_analysis.py`, `stats.py` and `stats_holdout.py` (the
statistical protocol: Friedman, Nemenyi, Wilcoxon, permutation tests),
`power_analysis_regime.py`, `screen_regime_candidates.py`, `report_tables.py`
(the tables of the paper), `export_numbers.py` (**every number quoted in the
paper, as a LaTeX macro**), `exp_common.py` (the shared experiment skeleton),
`repair_csv.py`, `clean_error_rows.py`, `remove_rows.py` (maintenance of the
result CSVs).

### 5.6 `src/figures/` — figures

One script per figure. A CSV with the plotted data is written next to every PDF.
Output is vector PDF (`pdf.fonttype=42`) with colourblind-safe palettes
(Okabe–Ito); the shared setup lives in `fig_common.py`.

Main text: `fig_faithful_map`, `fig_pareto_front`, `fig_regime_map`,
`fig_alpha_gain_by_regime`, `fig_graph_layouts`, `fig_temporal_pareto`,
`fig_metric_fidelity_boxes`.
Supplement and diagnostics: `fig_alpha_curves`, `fig_alpha_strip`,
`fig_before_after`, `fig_cd_diagram`, `fig_gallery_methods`,
`fig_graph_layouts_all`, `fig_holdout_paired`, `fig_metric_correlations`,
`fig_metric_fidelity_gallery`, `fig_prop2_alpha_prediction`,
`fig_prop2_tightness`, `fig_rnx_curves`, `fig_runtime_scaling`,
`fig_sammon_demo`, `fig_sgd_convergence`, `fig_temporal_trajectories`.

### 5.7 `src/tools/`

`estimate_runtime_from_csv.py` — estimates how long a full run will take from
the measured `wall_time_sec` values of a previous run (a scheduling simulation
for N workers).

---

## 6. Running the experiments

Every experiment has its own launcher and **can be run on its own**:

```
src\run_exp1_dr_benchmark.bat full        ./src/run_exp1_dr_benchmark.sh full
src\run_exp3_graph_layout.bat full        ./src/run_exp3_graph_layout.sh full
```

**Modes.** Every launcher accepts `smoke`, `quick` or `full` (default `full`):

- `smoke` — the smallest possible grid, a check that the code runs (minutes),
- `quick` — a reduced grid for development,
- `full` — the grid used in the paper (hours).

Outputs of `smoke` and `quick` go **exclusively** into their own subdirectories
(`results/data/smoke/`, `results/figures/quick/`, …) and never overwrite full
results or files of the paper.

**Resume.** Long runs are unattended and resumable: after an interruption simply
run the same command again and the finished combinations are skipped. When a run
completes, a `<experiment>_DONE.txt` marker records the number of rows and
errors. The schema of the output CSV is checked **before** the workers start, so
a mismatch fails immediately rather than after hours of computation.

**Estimating the runtime** from the measured times of a previous run:

```
src\run_estimate_runtime.bat exp4_temporal "1,8,12"
```

### 6.1 Reproducing the paper from scratch

1. `src\run_list_datasets.bat` — downloads and verifies the datasets.
2. `src\run_dataset_properties.bat full` — properties of the inputs
   (distance concentration).
3. Experiments: `run_exp1_dr_benchmark`, `run_exp1_cluster_geometry`,
   `run_exp2_solver_scaling`, `run_exp3_graph_layout`, `run_exp4_temporal`,
   `run_exp4_neighbor_metrics`, `run_exp5_ablation`, `run_exp6_alpha_curves`,
   `run_exp7_rank_weights`, `run_exp8_prop2_check`, `run_exp9_metric_fidelity`
   (mode `full`).
4. `src\run_fit_alpha_rule.bat full` — derives the prediction rule for alpha.
5. `src\run_exp1_holdout_confirmatory.bat full` — confirmatory test on the
   hold-out set.
6. `src\run_main.bat full` — statistics, tables, figures and the macro file with
   every number quoted in the paper.

Chains of several phases are provided by `run_s1_all`, `run_s2_all` and
`run_all_experiments`.

---

## 7. Configuration

There are no magic constants in the code; everything lives in configuration
files:

- `src/common/config.yaml` — paths, parallelism, progress reporting, downloads,
  datasets, methods, metrics, figures, method parameters (`sammon.*`) and device
  selection (CPU/GPU).
- `src/experiments/config_experiments.yaml` — the grid of each experiment
  (datasets, methods, seeds, alpha and lambda grids) separately for `smoke`,
  `quick` and `full`, plus the reporting setup (which methods and metrics go
  into the main tables).

All seeds come from the configuration and the parallel map preserves order, so
repeating a run reproduces the same results. Determinism holds with a single
BLAS thread (`*_NUM_THREADS=1`), which the launchers set.

---

## 8. Tests

```
venv\python.exe -m pytest          # Windows
./venv/bin/python -m pytest        # Linux/macOS
```

`tests/` contains unit tests (stress and gradient against numerical
differentiation, weights, metrics, solvers), infrastructure tests (checkpoints,
resume, CSV schema, parallelism, logging) and regression tests of the
experiments on small data.

---

## 9. Citation and licence

If you use this code, please cite the paper (details will be added once it is
accepted) and the Zenodo archive.

The work builds on two earlier papers by the authors:

- M. Radvansky, M. Kudelka, Z. Horak, V. Snasel: *Network Layout Visualization
  Based on Sammon's Projection*, INCoS 2013, pp. 244–249.
  DOI [10.1109/INCoS.2013.43](https://doi.org/10.1109/INCoS.2013.43)
- M. Radvansky, M. Kudelka, Z. Horak, V. Snasel: *Visualization of Social Network
  Dynamics using Sammon's Projection*, CASoN 2013, pp. 56–61.
  DOI [10.1109/CASoN.2013.6622600](https://doi.org/10.1109/CASoN.2013.6622600)

The code is released under the **MIT** licence (see `LICENSE`). The archive of
data and results on Zenodo is released under **CC BY 4.0**. The datasets used in
the experiments remain under the licences of their original authors; this
repository does not redistribute them, it only downloads them from their
original sources.

---

## 10. Acknowledgements

This work was supported by the Student Grant Competition of VSB – Technical
University of Ostrava (SP2026/008). The authors thank the Department of Computer
Science, FEECS VSB-TUO, for the computing resources used in the experiments.
