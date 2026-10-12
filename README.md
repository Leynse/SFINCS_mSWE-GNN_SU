# Spatially Uniform Rainfall Dataset Preparation with SFINCS

This repository contains the supplied notebooks and Python modules used to prepare training and testing datasets for rainfall-driven flood modelling. Spatially uniform rainfall is called **Type A** in the original filenames. The workflow selects terrain, adds sloped terrain variants, runs SFINCS, exports simulation results to NPZ, builds multiscale graphs, and checks temporal samples for graph neural network training.

The six original Python modules and four notebooks are preserved **byte for byte**, including comments, notebook outputs, metadata, and existing local paths. The folder layout follows the supplied screenshots and the module import paths. The additions are documentation, YAML setup/reference files, a GitHub source-check workflow, ignore rules, folder placeholders, and source checksums. Model training and post-training analysis are outside this repository's scope.

## Repository contents

| Location | Purpose |
| --- | --- |
| `notebooks/type_a_temporal_uniform/` | Four original notebooks; run in the order below |
| `scripts/sfincs_dataset_creation/` | `dem.py`, `hyetograph.py`, `sfincs.py` |
| `scripts/sfincs_dataset_conversion/` | `pytorch_geometric.py`, `temporal_learning.py`, `type_a_data.py` |
| `data/dem/Raw_DEMs/DEM/` | Local input DEM text files; placeholder only |
| `data/rainfall/rainfall_events/` | Local input rainfall files; placeholder only |
| `data/Synthetic_DEMs/` | Generated terrain variants and selection CSVs |
| `exe/SFINCS_v2.3.0_mt_Faber_release_exe/` | Local SFINCS executable and its runtime files |
| `models/` | Generated SFINCS case folders |
| `outputs/` | Generated logs, NPZ datasets, split manifest and graphs |
| `environment.yml` | Conda setup template inferred from imports |
| `config/dataset_reference.yaml` | Human-readable reference to settings in the original code |
| `.github/workflows/source-checks.yml` | Python/notebook syntax checks; no simulations |


## Environment setup

The uploaded notebook metadata records Python **3.10.20**. An export of the original installed package versions was not provided. `environment.yml` therefore lists the dependencies inferred from the code; it is a setup template, not a validated or exact reproduction of the original environment. It includes HydroMT-SFINCS, NumPy, pandas, Matplotlib, xarray, NetCDF readers, PyTorch, PyTorch Geometric, tqdm and Jupyter.

From the extracted repository root:

```bash
conda env create -f environment.yml
conda activate sfincs-uniform-dataset
python -m ipykernel install --user --name sfincs-uniform-dataset --display-name "SFINCS uniform dataset"
jupyter lab
```

Select the `SFINCS uniform dataset` notebook kernel. The SFINCS executable is separate from the Python package `hydromt-sfincs`; installing this environment does not install `sfincs.exe`. The existing executable path targets a Windows installation of SFINCS v2.3.0. GPU inference/training dependencies are not configured here.

## Required local files and paths

The attachments contain code and screenshots, but no raw DEMs, rainfall forcing, SFINCS executable, generated simulations or exported datasets. Supply these locally:

1. Place the original `DEM_*.txt` files under `data/dem/Raw_DEMs/DEM/`. Each file contains `x`, `y`, and elevation columns for a complete grid.
2. Place the Type A `.prcp` rainfall files under `data/rainfall/rainfall_events/`. These contain elapsed time in seconds and rainfall intensity in mm/h. Split assignment uses return-period tokens such as `RP010` and `RP100` in filenames.
3. Place `sfincs.exe` and its required runtime files under `exe/SFINCS_v2.3.0_mt_Faber_release_exe/`.

The notebooks retain the original absolute project path:

```text
C:\Users\Prof. KN Jha\Desktop\SFINCS_mSWE-GNN_v3
```

To use the notebooks without modifying their contents, put the repository at that location on the original Windows machine, or arrange a local Windows directory junction at that path pointing to the extracted repository. For another project location, a user must adapt the `PROJECT_ROOT` and any literal `sys.path.insert` paths in their own working copy. No paths have been changed in this delivered repository.

**The reference YAML is not loaded by the original code.** Editing it does not update notebook paths, SFINCS settings, split logic or temporal sampling. There is no new YAML-driven command-line runner.

## Workflow

Run each notebook top to bottom, in this order.

| Order | Notebook | Main outputs |
| --- | --- | --- |
| 1 | `synthetic_dem_creation.ipynb` | Terrain selection CSVs, sloped DEM text files and `sloped_dem_summary.csv` |
| 2 | `type_A_sfincs_to_npz.ipynb` | SFINCS cases/logs, one NPZ per successful DEM–rainfall case, `run_summary.csv`, `dataset_split_type_A.json` |
| 3 | `type_A_prepare_training_data.ipynb` | NPZ/split checks and saved complete multiscale graphs per split |
| 4 | `type_A_test_temporal_samples.ipynb` | Shape, temporal-indexing and PyG batching checks for an example training event |

### Terrain selection

The synthetic-terrain notebook selects 20 base 64×64 DEMs using four relief groups and five samples per group with seed 42. It keeps all available 128×128 base DEMs. It adds a 5 m ramp to the first five selected 64×64 DEMs and a 10 m ramp to the next five; for 128×128 DEMs, the corresponding selections contain three DEMs each. Ramp directions use a random seed of 42.

The implemented ramp amplitudes are **5 m and 10 m**. The notebook's introductory text says “1% and 2%”; this original text is retained, but the implementation specifies total added ramp relief, not fixed percentage gradients. Added ramp relief is also distinct from the final terrain's total elevation range.

### Simulation and export

`sfincs.py` prepares case folders and ASCII elevation/mask files, writes the SFINCS configuration through HydroMT-SFINCS, copies rainfall forcing, runs the external executable, records runtime/logs, and saves NPZ output. Configured duration is 45 h with output every 1,800 s. Settings include Manning roughness 0.04, zero infiltration, wet/dry threshold 0.01 m, and EPSG 32643. Grid spacing is obtained from each DEM's coordinates.

Existing case directories are recreated when overwrite is enabled (the case-folder helper defaults to `overwrite=True`). A batch catches individual case failures and records them in `run_summary.csv`; only successful cases are listed in the split JSON. Check the run summary before treating the dataset as complete.

Each NPZ stores `dem`, `water_depth`, `model_time_seconds`, `rainfall_time_seconds`, `rain_mmhr` and JSON `metadata`.

### Dataset splits

The actual split function checks `RP100` first, then `RP010`, then the DEM's domain role. It does not use a configurable list of return periods.

| JSON key | Terrain/domain role | Rainfall filename |
| --- | --- | --- |
| `train` | Seen (`train`, 64×64 selection) | Contains neither `RP010` nor `RP100` |
| `validation` | Seen | Contains `RP010` |
| `test_unseen_rainfall` | Seen | Contains `RP100` |
| `test_unseen_domain` | Unseen (`test`, 128×128 selection) | Any non-`RP100`, including `RP010` |
| `test_unseen_domain_and_rainfall` | Unseen | Contains `RP100` |

Event counts depend on the input DEM/rainfall files and successful simulations. The authoritative split file is `outputs/batch_type_A_training_npz/dataset_split_type_A.json`. Its stored paths may be absolute paths to the original machine; moving generated datasets requires keeping those locations available or adapting a working copy of the manifest.

### Graphs and temporal samples

`pytorch_geometric.py` creates bidirectional four-neighbour grid graphs at fine, medium and coarse resolutions with edge distances and pooling maps. Terrain is pooled twice using 2×2 mean pooling, so each spatial dimension must be divisible by four. Rainfall is averaged over each model interval and repeated over fine-grid nodes. Complete graphs retain the full water-depth and rainfall sequences.

Default temporal inputs are `[h(t-2), h(t-1), h(t), P(t)]`, and the default six-step targets are `[h(t+1), ..., h(t+6)]`. Each sample additionally stores rainfall for the six rollout intervals and their durations. For 91 frames, these defaults produce 83 temporal samples per event. The test notebook checks temporal alignment and index offsets when batching two samples.

## Existing compatibility details

These observations are documented without altering the original sources:

- `type_a_data.py` functions `load_all_complete_graphs()` and `build_temporal_datasets()` expect a dictionary key named `test`. The notebooks generate the five keys above, which have no `test` entry. Calling these two helpers directly with the five-way manifest will raise a `KeyError`. The preparation notebook instead iterates through all split keys, and the test notebook reads a training event. For the five-way dataset, use the notebook's loading approach or call `load_complete_graphs()` separately for each named split.
- The graph reload demonstration in `type_A_prepare_training_data.ipynb` accesses training graph index 100. It requires at least 101 training graphs.
- Notebook visualisation examples use particular DEM/event filenames. Those cases must exist for those demonstration cells to run.
- Raw inputs, their provenance, the executable and original package versions were not supplied. This archive packages the supplied code; it does not demonstrate a newly reproduced dataset generation run.

## Checks and preservation

The ten original files were compared byte for byte against the uploaded copies. All six Python modules and all code cells in the four notebooks passed syntax checks. The added YAML files were parsed locally. No SFINCS simulations, dependency installation, graph generation or notebook execution were performed during packaging because the external runtime/data were not supplied.

The GitHub workflow repeats source syntax checks without executing notebook cells or installing simulation dependencies. It does not establish scientific or numerical correctness.

On a system with `sha256sum`, verify the preserved source files from the repository root:

```bash
sha256sum -c SOURCE_CHECKSUMS.sha256
```

The originals use their existing `scripts.*` imports. Their directories are left as Python namespace packages; no additional `__init__.py` files or modifications to modules are required for that layout when the repository root is on `sys.path`.

## Uploading to GitHub

Extract the ZIP and upload the **contents** of `SFINCS_mSWE-GNN_v3` to your repository root, including the dotfiles and `.github` directory. Alternatively, from the extracted root:

```bash
git init
git add .
git commit -m "Add spatially uniform SFINCS dataset preparation workflow"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository. The ignore rules keep local inputs, executables, generated cases and outputs out of Git; the archive includes folder placeholders. Existing notebook outputs and local path strings remain in the preserved notebooks and will be visible if you publish them.

No software licence has been assigned as part of this packaging. Set the intended licence and attribution in GitHub according to the original sources and their permissions.
