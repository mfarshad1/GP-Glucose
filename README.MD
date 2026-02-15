# Accelerating Discovery of Glucose Binders via Machine Learning and Molecular Simulations

This repository contains code and analysis workflows supporting the paper:

**“Accelerating Discovery of Glucose Binders via Machine Learning and Molecular Simulations”**  
Mohsen Farshad, Fathya Y. M. Salih, Yamil J. Colón, Jonathan K. Whitmer

## 🧠 Overview
This repository implements an integrated simulation–machine-learning workflow to analyze glucose binding in synthetic host molecules. The workflow combines umbrella-sampling MD free-energy calculations with structural descriptor extraction, principal component analysis (PCA), and Gaussian Process (GP) regression to learn structure–affinity relationships and quantify predictive uncertainty.

## 📈 Method Flow
- Compute binding free energies from umbrella sampling (PMFs) using WHAM.
- Extract structural descriptors for each host:
  - RDKit-based chemical/electronic/topological features (e.g., heteroatom counts, aromaticity, hydrogen-bond donors/acceptors, charge, electronegativity).
  - HOLE-based pore-geometry descriptors (average and median pore radius).
- Standardize descriptors and perform PCA for dimensionality reduction and interpretation.
- Train GP regression models in PCA space to predict binding free energies and generate paper figures/tables.

## 🗂️ Analysis pipeline (as used in this project)

**Environment**
- `conda activate rdkit`

**Data preparation and descriptor generation**
- Move/curate mol2 files into the working directory structure:
  - `bash mv_mol2.sh`
  - `bash delete-mol2.sh`
  - `bash mv_files.sh`
- Collect mol2 files into a single folder:
  - copy `host_*.mol2` into `all-host-mol2s/`
- Fix mol2 atom naming:
  - `./change_names_mol2.sh host_*.mol2 fixed_host_*.mol2`
- Extract RDKit descriptors from fixed mol2 files:
  - `python get-host-prop-v07.py all-host-mol2s/fixed_host_*.mol2`

**Pore geometry (HOLE-based)**
- Compute pore diameter profiles per host:
  - `for x in host-*; do (cd $x; python pore_diameter_single.py); done`
  - or submit all at once: `./run_python.sh`
- Combine pore geometry into host properties:
  - `python combine-all-props-v02.py host-*/diameter_profile*`

**Free-energy targets**
- Compute free energies from umbrella sampling / WHAM outputs:
  - `cd targets`
  - `./run_wham.sh`
  - `./run_python.sh`
  - `cd ..`

**Build datasets for ML**
- Combine host properties without targets:
  - `python combine-props-targets-v05.py`
- Combine host properties with targets:
  - `python combine-props-targets-v06.py`
- Generate paper host dataset (H1–H36):
  - `cd all-host-props-nopore`
  - `./delete_difference.sh`
  - `cd ..`

**PCA**
- `cd pca`
- `python pca-v03.py`
- `python pca-v04.py`
- `python pca-paper_fig.py`
- `cd ..`

**Gaussian Process regression**
- `python gpr-random_seed-paper.py` (paper GP figure)
- `python gpr-pca-active-learning-v03.py` (tables 2 and S4)

## Notes on external tools (WHAM / HOLE) and provided outputs

Some parts of the original workflow rely on external tools and/or simulation data that are not included in this repository.  
To keep the full ML/PCA/GP analysis reproducible, we provide the processed outputs those tools generate.

### WHAM (umbrella sampling free energies)
The original WHAM input data (umbrella windows / trajectories and WHAM intermediate files) are not included.  
Instead, we provide the final processed targets used for ML:

- `targets/output.txt` — binding free energies extracted from WHAM and used as the regression targets.

With `targets/output.txt` available, you can reproduce descriptor processing, PCA, GP regression, and the paper figures/tables without re-running WHAM.  
If you have access to the WHAM inputs, the scripts in `targets/` (e.g., `run_wham.sh`, `run_python.sh`) document how `output.txt` was generated.

### HOLE (pore geometry)
Computing pore-diameter profiles requires the HOLE program, which may not be available to all users.  
To support reproducibility without HOLE, we provide the precomputed pore-diameter outputs in each host folder (e.g., files matching `diameter_profile*`).

With these precomputed diameter profiles present, you can run the pore-geometry aggregation step directly:

```bash
python combine-all-props-v02.py host-*/diameter_profile*
