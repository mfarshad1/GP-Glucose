"""
Split dataset into training and testing using stratified k-folds.
Auto-runs PCA2–PCA10 without CLI arguments.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# =============================================================================
# Plot Configuration (LaTeX-like styling)
# =============================================================================

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize=18)
rc('xtick', labelsize=14)
rc('ytick', labelsize=14)
rc('legend', fontsize=14)
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams["savefig.pad_inches"] = 0.02

# =============================================================================
# Functions
# =============================================================================

def split_data(csv_path, target_col_name, n_splits, n_bins, out_dir,
               frac=1.0, suffix=""):
    """
    Split data into stratified K folds.
    If frac < 1.0, use a random subset of that fraction of the rows.
    Filenames get the provided suffix (e.g. '_3_4', '_1_2', '_1_4').
    """
    df = pd.read_csv(csv_path, index_col=0)

    # Optional subsampling to a fraction of the dataset
    if frac < 1.0:
        df = df.sample(frac=frac, random_state=42).sort_index()

    prop = df[target_col_name]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    prop_cut = pd.cut(prop, bins=n_bins, labels=range(n_bins))
    X = df.drop(columns=[target_col_name], axis=1)

    os.makedirs(out_dir, exist_ok=True)

    for k, (train_idx, test_idx) in enumerate(skf.split(X, prop_cut)):
        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]
        train_df.to_csv(f"{out_dir}/train_fold_{k+1}{suffix}.csv", index=True)
        test_df.to_csv(f"{out_dir}/test_fold_{k+1}{suffix}.csv", index=True)


def visualize_distns(target_col_name, n_splits, n_bins, out_dir,
                     suffix="", tag=""):
    """
    Visualize target distributions for each fold.
    suffix: filename suffix used for train/test files (e.g. '_3_4').
    tag:    extra tag to append to figure filenames 
            (e.g. '_3_4', '_1_2', '_1_4'; empty for full data).
    """
    # ========= Load all ΔG values to determine common axis ranges =========
    all_vals = []
    for k in range(n_splits):
        ftrain = pd.read_csv(f"{out_dir}/train_fold_{k+1}{suffix}.csv",
                             index_col=0)
        ftest  = pd.read_csv(f"{out_dir}/test_fold_{k+1}{suffix}.csv",
                             index_col=0)
        all_vals.append(ftrain[target_col_name].values)
        all_vals.append(ftest[target_col_name].values)
    all_vals = np.hstack(all_vals)
    xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.05 * (xmax - xmin + 1e-9)
    xmin -= pad
    xmax += pad

    # ========= Create figure (ONE ROW, n_splits COLUMNS) =========
    fig, axs = plt.subplots(
        nrows=1,
        ncols=n_splits,
        figsize=(8.5, 3),
        constrained_layout=True
    )

    if n_splits == 1:
        axs = [axs]  # ensure list-like

    for k in range(n_splits):
        train_df = pd.read_csv(f"{out_dir}/train_fold_{k+1}{suffix}.csv",
                               index_col=0)
        test_df  = pd.read_csv(f"{out_dir}/test_fold_{k+1}{suffix}.csv",
                               index_col=0)

        y_train = train_df[target_col_name]
        y_test  = test_df[target_col_name]

        ax = axs[k]

        # ---- minor ticks ----
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # ---- histograms ----
        ax.hist(y_train, bins=n_bins, alpha=0.5, color='tab:blue', label='Train')
        ax.hist(y_test,  bins=n_bins, alpha=0.5, color='tab:red',  label='Test')

        ax.set_xlim(xmin, xmax)

        ax.set_xlabel(r"$\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$")
        if k == 0:
            ax.set_ylabel("Frequency")

        if k == 0:
            ax.legend(frameon=False)

        ax.set_title(fr"Fold {k+1}", fontsize=16, fontweight='bold')

    # ========= Save PNG + PDF =========
    # keep original naming for full data (tag == "")
    base_name = "PropertyDistribution"
    png_path = f"{out_dir}/{base_name}{tag}.png"
    pdf_path = f"{out_dir}/{base_name}{tag}.pdf"

    fig.savefig(png_path, dpi=300)
    with PdfPages(pdf_path) as pp:
        pp.savefig(fig, bbox_inches='tight')

    plt.show()
    plt.close(fig)


# =============================================================================
# Main — AUTO RUN PCA2 → PCA10
# =============================================================================
if __name__ == "__main__":

    BASE = Path("/users/mfarshad/afs/Private/ML-new/traj")
    TARGET = "dG"
    FOLDS = 3
    BINS = 5

    for n in range(2, 11):  # pca2 ... pca10
        csv_path = BASE / f"pca/pca{n}.csv"
        out_dir  = BASE / "data-splits" / f"pca{n}"

        print(f"\n=== Processing PCA{n} ===")
        print(f"CSV: {csv_path}")
        print(f"OUT: {out_dir}")

        # --- Full data (original behavior, unchanged names) ---
        split_data(csv_path, TARGET, FOLDS, BINS, out_dir,
                   frac=1.0, suffix="")
        visualize_distns(TARGET, FOLDS, BINS, out_dir,
                         suffix="", tag="")

        # --- 3/4 of data ---
        print("  -> 3-fold on 3/4 of data")
        split_data(csv_path, TARGET, FOLDS, BINS, out_dir,
                   frac=0.75, suffix="_3_4")
        visualize_distns(TARGET, FOLDS, BINS, out_dir,
                         suffix="_3_4", tag="_3_4")

        # --- 1/2 of data ---
        print("  -> 3-fold on 1/2 of data")
        split_data(csv_path, TARGET, FOLDS, BINS, out_dir,
                   frac=0.5, suffix="_1_2")
        visualize_distns(TARGET, FOLDS, BINS, out_dir,
                         suffix="_1_2", tag="_1_2")

        # --- 1/4 of data ---
        print("  -> 3-fold on 1/4 of data")
        split_data(csv_path, TARGET, FOLDS, BINS, out_dir,
                   frac=0.3, suffix="_1_4")
        visualize_distns(TARGET, FOLDS, BINS, out_dir,
                         suffix="_1_4", tag="_1_4")
