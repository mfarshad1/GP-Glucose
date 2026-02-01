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

## 📦 Outputs (per your scripts)
- PCA figures (variance explained, loadings, labeled/unlabeled PCA plots)
- GP parity/performance figures
- GP outputs used for paper tables (e.g., Tables 2 and S4)
