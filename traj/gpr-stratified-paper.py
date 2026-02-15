#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpflow
import scipy
import matplotlib.ticker as ticker
import tensorflow as tf
from matplotlib import rc
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # <-- changed

# ================= Plot properties (LaTeX + serif like your other GP code) =================
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize=14)
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)

# ================= Small helper =================
def rsquared(x, y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return float(r_value**2)

# ================= Settings =================
n_folds = 3
n_pca_comp_list = list(range(2, 11))
len_pca = len(n_pca_comp_list)

# Leakage smoke-test flag (leave False for normal runs)
SHUFFLE_TRAIN_Y_FOR_SMOKE_TEST = False

# Fractions / suffix configs
fraction_configs = [
    ("full", "", ""),       # full data
]

# ================= Loop over data fractions =================
for frac_name, file_suffix, tag_suffix in fraction_configs:

    print(f"\n================== Data fraction: {frac_name} ==================\n")

    # ================= Overall stats container for this fraction =================
    all_r2_pca = pd.DataFrame(
        np.zeros((len_pca, 4)),
        columns=['train-avg', 'test-avg', 'train-std', 'test-std'],
        index=n_pca_comp_list
    )
    all_r2_pca.index.name = 'n_pca'

    # --- holders for PCA=2 data used in combined figure ---
    pca2_hist_data = []          # list of (y_train, y_test) per fold
    pca2_y_train_all = None
    pca2_y_train_pred_all = None
    pca2_y_test_all = None
    pca2_y_test_pred_all = None
    pca2_R2_train_all = None
    pca2_R2_test_all = None

    # ================= Main loop over PCA components =================
    for n_pca_comp in n_pca_comp_list:
        print(f"\nNumber of PCA components: {n_pca_comp}")

        # For a single parity plot per PCA (aggregate all folds)
        y_train_all, y_train_pred_all = [], []
        y_test_all,  y_test_pred_all  = [], []

        R2_results = pd.DataFrame(
            np.zeros((n_folds, 2)),
            columns=['train', 'test'],
            index=range(1, n_folds + 1)
        )

        # reset PCA2 fold hist holder when we hit n_pca_comp == 2
        if n_pca_comp == 2:
            pca2_hist_data = []

        for k in range(1, n_folds + 1):
            # ----- Load split -----
            train_path = f"data-splits/pca{n_pca_comp}/train_fold_{k}{file_suffix}.csv"
            test_path  = f"data-splits/pca{n_pca_comp}/test_fold_{k}{file_suffix}.csv"

            train_df = pd.read_csv(train_path, index_col=0)
            test_df  = pd.read_csv(test_path,  index_col=0)

            # ---------- assert no overlap of rows by index ----------
            overlap = train_df.index.intersection(test_df.index)
            if len(overlap) > 0:
                raise RuntimeError(
                    f"[LEAK] frac={frac_name} PCA={n_pca_comp} fold={k}: "
                    f"{len(overlap)} samples appear in BOTH train and test"
                )

            # Features and labels
            x_train = train_df.iloc[:, 1:1 + n_pca_comp].to_numpy()
            y_train = train_df["dG"].to_numpy().reshape(-1, 1)
            x_test  = test_df.iloc[:, 1:1 + n_pca_comp].to_numpy()
            y_test  = test_df["dG"].to_numpy().reshape(-1, 1)

            # store raw dG for PCA=2 histograms
            if n_pca_comp == 2:
                pca2_hist_data.append((y_train.copy(), y_test.copy()))

            # ----- Standardize (fit on train only) -----
            x_scaler = StandardScaler().fit(x_train)
            y_scaler = StandardScaler().fit(y_train)
            x_train_n = x_scaler.transform(x_train)
            y_train_n = y_scaler.transform(y_train)
            x_test_n  = x_scaler.transform(x_test)

            # ----- Optional smoke test: destroy signal in training labels -----
            if SHUFFLE_TRAIN_Y_FOR_SMOKE_TEST:
                rng = np.random.default_rng(2025)
                y_train_n = y_train_n.copy()
                rng.shuffle(y_train_n)

            # ----- GPR (same kernel, tiny white noise) -----
            kernel = gpflow.kernels.Matern32(lengthscales=1.0, variance=1.0) \
                   + gpflow.kernels.White(variance=1e-5)
            model = gpflow.models.GPR((x_train_n, y_train_n), kernel=kernel)
            opt = gpflow.optimizers.Scipy()
            opt.minimize(
                model.training_loss,
                model.trainable_variables,
                options=dict(maxiter=200, disp=False)
            )

            # ----- Predictions (normalized) -----
            y_train_pred_n, _ = model.predict_f(x_train_n)
            y_test_pred_n, _  = model.predict_f(x_test_n)

            # ----- Back-transform to original units -----
            y_train_pred = y_scaler.inverse_transform(y_train_pred_n)
            y_test_pred  = y_scaler.inverse_transform(y_test_pred_n)

            # ----- R^2 on original scale (per fold) -----
            R2_results.loc[k, 'train'] = rsquared(y_train, y_train_pred)
            R2_results.loc[k, 'test']  = rsquared(y_test,  y_test_pred)

            # Accumulate for a single parity plot per PCA (all folds together)
            y_train_all.append(y_train)
            y_train_pred_all.append(y_train_pred)
            y_test_all.append(y_test)
            y_test_pred_all.append(y_test_pred)

        # ================= Aggregated R^2 (used for "mean") =================
        y_train_all       = np.vstack(y_train_all)
        y_train_pred_all  = np.vstack(y_train_pred_all)
        y_test_all        = np.vstack(y_test_all)
        y_test_pred_all   = np.vstack(y_test_pred_all)

        R2_train_all = rsquared(y_train_all, y_train_pred_all)
        R2_test_all  = rsquared(y_test_all,  y_test_pred_all)

        # keep PCA=2 parity data for combined figure
        if n_pca_comp == 2:
            pca2_y_train_all      = y_train_all.copy()
            pca2_y_train_pred_all = y_train_pred_all.copy()
            pca2_y_test_all       = y_test_all.copy()
            pca2_y_test_pred_all  = y_test_pred_all.copy()
            pca2_R2_train_all     = R2_train_all
            pca2_R2_test_all      = R2_test_all

        # std over folds
        train_std = R2_results['train'].std()
        test_std  = R2_results['test'].std()

        # overwrite "mean" row with aggregated R^2
        R2_results.loc['mean', 'train'] = R2_train_all
        R2_results.loc['mean', 'test']  = R2_test_all
        R2_results.loc['std',  'train'] = train_std
        R2_results.loc['std',  'test']  = test_std

        perf_path = f"data-splits/pca{n_pca_comp}/performances{tag_suffix}.csv"
        R2_results.to_csv(perf_path)

        all_r2_pca.loc[n_pca_comp, 'train-avg'] = R2_train_all
        all_r2_pca.loc[n_pca_comp, 'test-avg']  = R2_test_all
        all_r2_pca.loc[n_pca_comp, 'train-std'] = train_std
        all_r2_pca.loc[n_pca_comp, 'test-std']  = test_std

        print(f"[{frac_name}] R2_train (agg) = {R2_train_all:.2f} ± {train_std:.2f}")
        print(f"[{frac_name}] R2_test  (agg) = {R2_test_all:.2f} ± {test_std:.2f}")

        # ================= Parity plot per PCA (train + test, clearly distinct) =================
        rng = np.random.default_rng(123)
        jscale = max(1e-6, 0.002 * float(np.std(np.vstack([y_train_all, y_test_all]))))
        y_train_pred_plot = y_train_pred_all + rng.normal(0.0, jscale, size=y_train_pred_all.shape)
        y_test_pred_plot  = y_test_pred_all  + rng.normal(0.0, jscale, size=y_test_pred_all.shape)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.scatter(
            y_train_all, y_train_pred_plot, s=22,
            facecolors="none", edgecolors="tab:blue", marker="o",
            linewidths=1.2, alpha=0.9,
            label=rf"Train ($R^2={R2_train_all:.2f}$)"
        )
        ax.scatter(
            y_test_all,  y_test_pred_plot,  s=26,
            facecolors="none", edgecolors="tab:red", marker="s",
            linewidths=1.4, alpha=0.9,
            label=rf"Test ($R^2={R2_test_all:.2f}$)"
        )

        # y = x line
        y_all = np.vstack([y_train_all, y_test_all]).ravel()
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        pad = 0.02 * (y_max - y_min + 1e-12)
        ax.plot(
            [y_min - pad, y_max + pad],
            [y_min - pad, y_max + pad],
            'k-', lw=1.1
        )

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        ax.set_xlabel(r"True $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$", fontsize=16, fontweight='bold')
        ax.set_ylabel(r"Predicted $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$", fontsize=16, fontweight='bold')
        ax.set_title(
            fr"{n_pca_comp} PCA components",
            fontsize=16, fontweight='bold'
        )

        ax.legend(
            frameon=False, fontsize=12, loc='upper left',
            labelspacing=0.4, handlelength=1.2, borderaxespad=0.2
        )

        fig.tight_layout()
        base_parity = f"data-splits/pca{n_pca_comp}/gpr_fit_scatter_allfolds{tag_suffix}"
        fig.savefig(base_parity + ".png", dpi=300)
        fig.savefig(base_parity + ".pdf", bbox_inches='tight')
        plt.show()

    # ================= Save the summary CSV for this fraction =================
    summary_csv = f"data-splits/r2_values_strat{tag_suffix}.csv"
    all_r2_pca.to_csv(summary_csv, index=True)

    # ================= Reload (optional) and find best PCA for this fraction =================
    all_r2_pca_loaded = pd.read_csv(summary_csv, index_col=0)
    best_n = all_r2_pca_loaded['test-avg'].idxmax()
    print(f"\n[{frac_name}] Best PCA components based on test: {best_n}")

    # ================= Summary error-bar plot for this fraction =================
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.errorbar(
        n_pca_comp_list,
        all_r2_pca_loaded.loc[n_pca_comp_list, 'train-avg'],
        yerr=all_r2_pca_loaded.loc[n_pca_comp_list, 'train-std'],
        fmt='-o', label='Train', capsize=4
    )
    ax.errorbar(
        n_pca_comp_list,
        all_r2_pca_loaded.loc[n_pca_comp_list, 'test-avg'],
        yerr=all_r2_pca_loaded.loc[n_pca_comp_list, 'test-std'],
        fmt='-x', label='Test', capsize=4
    )

    ax.set_xlabel(r'Number of PCA components', fontsize=16, fontweight='bold')
    ax.set_ylabel(r'$R^2$', fontsize=16, fontweight='bold')

    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(-0.2, 1.4)

    ax.legend(
        frameon=False, fontsize=12, loc='lower right',
        labelspacing=0.4, handlelength=1.2
    )
    fig.tight_layout()

    base_summary = f"data-splits/r2_vs_pca_strat{tag_suffix}"
    fig.savefig(base_summary + ".png", dpi=300)
    fig.savefig(base_summary + ".pdf", bbox_inches='tight')
    plt.show()

    # ================= Combined figure: histograms (folds), parity (PCA2), R2 vs PCA =========
    if (frac_name == "full") and (pca2_y_train_all is not None):

        # prepare common histogram bins over all PCA2 train/test dG
        all_vals = np.concatenate(
            [arr.ravel() for (yt, ys) in pca2_hist_data for arr in (yt, ys)]
        )
        bins = np.linspace(all_vals.min(), all_vals.max(), 7)

        fig = plt.figure(figsize=(8.0, 5.5))

        # ---- Outer grid: band 0 = histograms, band 1 = (parity + R2) ----
        outer = GridSpec(
            2, 1,
            height_ratios=[1.0, 1.3],   # top vs bottom height
            hspace=0.5                # gap between rows
        )

        # Top band: 1×3 histograms (Fold 1–3), equal widths
        top_gs = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[0], wspace=0.3
        )

        # Bottom band: 1×2 with 1.5:1 width ratio (parity : R2)
        bottom_gs = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[1],
            width_ratios=[1.02, 1.0],    # <-- this is the key
            wspace=0.3
        )

        # ---------------- top row: 3 hist panels ----------------
        hist_axes = [fig.add_subplot(top_gs[0, i]) for i in range(3)]
        colors = {"train": "tab:blue", "test": "tab:red"}

        for i, ax_h in enumerate(hist_axes):
            if i < len(pca2_hist_data):
                y_tr, y_te = pca2_hist_data[i]
                ax_h.hist(
                    y_tr.ravel(), bins=bins, alpha=0.7,
                    color=colors["train"], label="Train"
                )
                ax_h.hist(
                    y_te.ravel(), bins=bins, alpha=0.7,
                    color=colors["test"], label="Test"
                )
                ax_h.set_title(f"Fold {i+1}", fontsize=14, fontweight='bold')
                if i == 0:
                    ax_h.set_ylabel("Frequency", fontsize=14, labelpad= 12.5)
                    ax_h.legend(frameon=False, fontsize=10)
                ax_h.set_xlabel(r"$\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
                                fontsize=14)

        # ---------------- bottom left: parity (PCA2) ----------------
        ax_par = fig.add_subplot(bottom_gs[0, 0])
        ax_par.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_par.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        rng = np.random.default_rng(321)
        jscale = max(
            1e-6, 0.002 * float(
                np.std(np.vstack([pca2_y_train_all, pca2_y_test_all]))
            )
        )
        y_train_pred_plot = pca2_y_train_pred_all + rng.normal(
            0.0, jscale, size=pca2_y_train_pred_all.shape
        )
        y_test_pred_plot = pca2_y_test_pred_all + rng.normal(
            0.0, jscale, size=pca2_y_test_pred_all.shape
        )

        ax_par.scatter(
            pca2_y_train_all, y_train_pred_plot, s=22,
            facecolors="none", edgecolors="tab:blue", marker="o",
            linewidths=1.2, alpha=0.9,
            label=rf"Train ($R^2={pca2_R2_train_all:.2f}$)"
        )
        ax_par.scatter(
            pca2_y_test_all,  y_test_pred_plot,  s=26,
            facecolors="none", edgecolors="tab:red", marker="s",
            linewidths=1.4, alpha=0.9,
            label=rf"Test ($R^2={pca2_R2_test_all:.2f}$)"
        )

        y_all = np.vstack([pca2_y_train_all, pca2_y_test_all]).ravel()
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
        pad = 0.02 * (y_max - y_min + 1e-12)
        ax_par.plot(
            [y_min - pad, y_max + pad],
            [y_min - pad, y_max + pad],
            'k-', lw=1.1
        )

        ax_par.set_xlabel(
            r"True $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
            fontsize=14, fontweight='bold'
        )
        ax_par.set_ylabel(
            r"Predicted $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
            fontsize=14, fontweight='bold'
        )
        ax_par.set_title(r"2 PCA components", fontsize=14, fontweight='bold')
        ax_par.legend(frameon=False, fontsize=10, loc='upper left')

        # ---------------- bottom right: R^2 vs PCA ----------------
        ax_r2 = fig.add_subplot(bottom_gs[0, 1])
        ax_r2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax_r2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax_r2.errorbar(
            n_pca_comp_list,
            all_r2_pca_loaded.loc[n_pca_comp_list, 'train-avg'],
            yerr=all_r2_pca_loaded.loc[n_pca_comp_list, 'train-std'],
            fmt='-o', label='Train', capsize=4
        )
        ax_r2.errorbar(
            n_pca_comp_list,
            all_r2_pca_loaded.loc[n_pca_comp_list, 'test-avg'],
            yerr=all_r2_pca_loaded.loc[n_pca_comp_list, 'test-std'],
            fmt='-x', label='Test', capsize=4
        )

        ax_r2.set_xlabel(r'Number of PCA components', fontsize=14, fontweight='bold')
        ax_r2.set_ylabel(r'$R^2$', fontsize=14, fontweight='bold')
        ax_r2.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax_r2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax_r2.set_ylim(-0.2, 1.4)
        ax_r2.legend(frameon=False, fontsize=10, loc='lower right')

        fig.tight_layout()
        combo_base = f"data-splits/stratified_pca2_overview"
        fig.savefig(combo_base + ".png", dpi=300)
        fig.savefig(combo_base + ".pdf", bbox_inches='tight')
        plt.show()
