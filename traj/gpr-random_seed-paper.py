#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random-seed Gaussian Process Regression on PCA features.

- Uses the same GP algorithm and plotting style as the "pretty" random-seed code.
- Loops over seeds and PCA components (2..10).
- Saves per-(seed, PCA) parity plots and a summary R^2 vs PCA (mean ± std).

KEPT (original):
- The original 2-panel combined figure:
    (1) aggregated parity for BEST PCA (over seeds)
    (2) R^2 vs PCA components (mean ± std)

ADDED (in addition, not replacing):
- A 3-panel combined figure:
    (1) BEST seed parity for the BEST PCA predictor
    (2) WORST seed parity for the BEST PCA predictor
    (3) R^2 vs PCA components (mean ± std)

Also:
- Consistent naming: "PCA{n}" (e.g., PCA10), not "10 PCA".
"""

from collections import defaultdict
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpflow
import scipy
import matplotlib.ticker as ticker
import tensorflow as tf
from matplotlib import rc
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # for combined figure

# ================= Plot properties (LaTeX + serif, like k-fold code) =================
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
    """Compute R^2 using scipy linregress, on flattened arrays."""
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return float(r_value**2)

def parity_plot(ax, y_train, y_train_pred, y_test, y_test_pred,
                R2_train, R2_test, title, legend_loc='lower right'):
    """Parity plot styled like the per-(seed,PCA) parity plots."""
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.scatter(
        y_train, y_train_pred, s=22,
        facecolors="none", edgecolors="tab:blue", marker="o",
        linewidths=1.2, alpha=0.9,
        label=rf"Train ($R^2={R2_train:.2f}$)"
    )
    ax.scatter(
        y_test,  y_test_pred,  s=26,
        facecolors="none", edgecolors="tab:red", marker="s",
        linewidths=1.4, alpha=0.9,
        label=rf"Test ($R^2={R2_test:.2f}$)"
    )

    # y = x line
    y_all = np.vstack([y_train, y_test]).ravel()
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad = 0.02 * (y_max - y_min + 1e-12)
    ax.plot([y_min - pad, y_max + pad], [y_min - pad, y_max + pad], 'k-', lw=1.1)

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(r"True $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
                  fontsize=14, fontweight='bold')
    ax.set_ylabel(r"Predicted $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
                  fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(frameon=False, fontsize=10, loc=legend_loc,
              labelspacing=0.4, handlelength=0.8)

# ================= Settings =================
seed_list = list(range(10))          # 0..9
n_pca_comp_list = list(range(2, 11)) # 2..10

R2_train_all = defaultdict(list)     # {n_pca: [R2_train_seed0, ...]}
R2_test_all  = defaultdict(list)     # {n_pca: [R2_test_seed0,  ...]}

# For aggregated parity per PCA (so we can pick the best one later)
agg_y_train      = defaultdict(list)   # {n_pca: [y_train_seed0, ...]}
agg_y_train_pred = defaultdict(list)
agg_y_test       = defaultdict(list)
agg_y_test_pred  = defaultdict(list)

# NEW: store per-seed details so we can plot best/worst seed for best PCA later
perseed = defaultdict(dict)
# perseed[n_pca][seed] = dict(y_train, y_train_pred, y_test, y_test_pred, R2_train, R2_test)

# Make sure output directory exists
os.makedirs("pca/gpr-fit-results", exist_ok=True)

# ================= Main loops: over seeds, then PCA components =================
for seed in seed_list:
    print(f"\n===== Seed: {seed} =====")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    for n_pca_comp in n_pca_comp_list:
        print(f"\nNumber of PCA components: {n_pca_comp}")

        # ----- Load PCA data -----
        data = pd.read_csv(f"pca/pca{n_pca_comp}.csv")
        x_data = data.iloc[:, 1:1 + n_pca_comp].to_numpy()
        y_data = data["dG"].to_numpy().reshape(-1, 1)

        # ----- Train/test split -----
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=seed
        )

        # ----- GPR model -----
        kernel = (
            gpflow.kernels.Matern32(lengthscales=1.0, variance=1.0)
            # + gpflow.kernels.Polynomial()
            + gpflow.kernels.White()
        )

        model = gpflow.models.GPR((x_train, y_train), kernel=kernel)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

        # ----- Predictions -----
        y_train_pred, _ = model.predict_f(x_train)
        y_test_pred,  _ = model.predict_f(x_test)

        y_train_pred = y_train_pred.numpy()
        y_test_pred  = y_test_pred.numpy()

        # ----- R^2 (per seed, per PCA) -----
        R2_train = rsquared(y_train, y_train_pred)
        R2_test  = rsquared(y_test,  y_test_pred)
        print(f"R2_train = {R2_train:.2f}, R2_test = {R2_test:.2f}")

        R2_train_all[n_pca_comp].append(R2_train)
        R2_test_all[n_pca_comp].append(R2_test)

        # store for later aggregated parity per PCA
        agg_y_train[n_pca_comp].append(y_train)
        agg_y_train_pred[n_pca_comp].append(y_train_pred)
        agg_y_test[n_pca_comp].append(y_test)
        agg_y_test_pred[n_pca_comp].append(y_test_pred)

        # store for later best/worst seed selection
        perseed[n_pca_comp][seed] = dict(
            y_train=y_train, y_train_pred=y_train_pred,
            y_test=y_test, y_test_pred=y_test_pred,
            R2_train=R2_train, R2_test=R2_test
        )

        # ================= Parity plot for THIS seed & THIS PCA =================
        fig, ax = plt.subplots(figsize=(4, 3))
        parity_plot(
            ax,
            y_train, y_train_pred, y_test, y_test_pred,
            R2_train, R2_test,
            title=fr"PCA{n_pca_comp} (seed={seed})",
            legend_loc='lower right'
        )

        fig.tight_layout()
        base_parity = f"pca/gpr-fit-results/pca{n_pca_comp}_seed{seed}_scatter"
        fig.savefig(base_parity + ".png", dpi=300)
        fig.savefig(base_parity + ".pdf", bbox_inches='tight')
        plt.show()

# ================= Summary: mean ± std R² over seeds =================
mean_train = []
std_train  = []
mean_test  = []
std_test   = []

for n_pca_comp in n_pca_comp_list:
    train_vals = np.array(R2_train_all[n_pca_comp])
    test_vals  = np.array(R2_test_all[n_pca_comp])

    mean_train.append(np.mean(train_vals))
    std_train.append(np.std(train_vals))
    mean_test.append(np.mean(test_vals))
    std_test.append(np.std(test_vals))

# Save summary CSV
R2_avg_df = pd.DataFrame({
    'n_pca_comp': n_pca_comp_list,
    'R2_train_mean': mean_train,
    'R2_train_std': std_train,
    'R2_test_mean': mean_test,
    'R2_test_std': std_test
})
summary_csv = "pca/gpr-fit-results/r2_values_randomseed_mean_std.csv"
R2_avg_df.to_csv(summary_csv, index=False)
print("\nSaved summary R^2 table to:", summary_csv)

# ================= Standalone summary error-bar plot (unchanged) =================
fig, ax = plt.subplots(figsize=(4.0, 3.0))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax.errorbar(
    n_pca_comp_list,
    mean_train,
    yerr=std_train,
    fmt='-o', label='Train', capsize=4
)
ax.errorbar(
    n_pca_comp_list,
    mean_test,
    yerr=std_test,
    fmt='-x', label='Test', capsize=4
)

ax.set_xlabel(r'Number of PCA components', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$R^2$', fontsize=16, fontweight='bold')

ax.ticklabel_format(axis='y', style='plain', useOffset=False)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.set_ylim(-0.2, 1.4)

ax.legend(
    frameon=False, fontsize=12, loc='upper left',
    labelspacing=0.4, handlelength=0.8
)
fig.tight_layout()

base_summary = "pca/gpr-fit-results/r2_vs_pca_randomseed"
fig.savefig(base_summary + ".png", dpi=300)
fig.savefig(base_summary + ".pdf", bbox_inches='tight')
plt.show()

# ================= Find BEST PCA based on mean test R^2 =================
mean_test_arr = np.array(mean_test)
best_idx = int(np.argmax(mean_test_arr))
best_n_pca = n_pca_comp_list[best_idx]
print(f"\nBest PCA based on mean test R^2: PCA{best_n_pca}")

# ================= ORIGINAL 2-PANEL combined figure (KEEP THIS) =================
# Aggregate train/test data over seeds for this best PCA (for plotting)
ytr_all      = np.vstack(agg_y_train[best_n_pca])
ytr_pred_all = np.vstack(agg_y_train_pred[best_n_pca])
yte_all      = np.vstack(agg_y_test[best_n_pca])
yte_pred_all = np.vstack(agg_y_test_pred[best_n_pca])

# R^2 for legend: use mean over seeds (to match right panel),
# not the R^2 from concatenated data
R2_train_label = mean_train[best_idx]
R2_test_label  = mean_test[best_idx]

fig = plt.figure(figsize=(7.5, 3.0))
outer = GridSpec(1, 1)
inner = GridSpecFromSubplotSpec(
    1, 2,
    subplot_spec=outer[0],
    width_ratios=[1.0, 1.0],
    wspace=0.25
)

# ----- Left: aggregated parity for best PCA -----
ax_par = fig.add_subplot(inner[0, 0])
ax_par.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax_par.yaxis.set_minor_locator(ticker.AutoMinorLocator())

rng = np.random.default_rng(123)
jscale = max(
    1e-6,
    0.002 * float(np.std(np.vstack([ytr_all, yte_all])))
)
ytr_pred_plot = ytr_pred_all + rng.normal(0.0, jscale, size=ytr_pred_all.shape)
yte_pred_plot = yte_pred_all + rng.normal(0.0, jscale, size=yte_pred_all.shape)

ax_par.scatter(
    ytr_all, ytr_pred_plot, s=22,
    facecolors="none", edgecolors="tab:blue", marker="o",
    linewidths=1.2, alpha=0.9,
    label=rf"Train ($R^2={R2_train_label:.2f}$)"
)
ax_par.scatter(
    yte_all, yte_pred_plot, s=26,
    facecolors="none", edgecolors="tab:red", marker="s",
    linewidths=1.4, alpha=0.9,
    label=rf"Test ($R^2={R2_test_label:.2f}$)"
)

y_all = np.vstack([ytr_all, yte_all]).ravel()
y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
pad = 0.02 * (y_max - y_min + 1e-12)
ax_par.plot(
    [y_min - pad, y_max + pad],
    [y_min - pad, y_max + pad],
    'k-', lw=1.1
)

ax_par.xaxis.set_major_formatter(formatter)
ax_par.yaxis.set_major_formatter(formatter)

ax_par.set_xlabel(
    r"True $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
    fontsize=14, fontweight='bold'
)
ax_par.set_ylabel(
    r"Predicted $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
    fontsize=14, fontweight='bold'
)
ax_par.set_title(
    rf"PCA{best_n_pca}",
    fontsize=14, fontweight='bold'
)
ax_par.legend(frameon=False, fontsize=10, loc='upper left')

# ----- Right: R^2 vs PCA components (mean ± std) -----
ax_r2 = fig.add_subplot(inner[0, 1])
ax_r2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax_r2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax_r2.errorbar(
    n_pca_comp_list,
    mean_train,
    yerr=std_train,
    fmt='-o', label='Train', capsize=4
)
ax_r2.errorbar(
    n_pca_comp_list,
    mean_test,
    yerr=std_test,
    fmt='-x', label='Test', capsize=4
)

ax_r2.set_xlabel(r'Number of PCA components', fontsize=14, fontweight='bold')
ax_r2.set_ylabel(r'$R^2$', fontsize=14, fontweight='bold')
ax_r2.ticklabel_format(axis='y', style='plain', useOffset=False)
ax_r2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax_r2.set_ylim(-0.2, 1.4)
ax_r2.legend(frameon=False, fontsize=10, loc='lower right')

fig.tight_layout()
combo_base_2panel = "pca/gpr-fit-results/randomseed_bestPCA_parity_plus_r2"
fig.savefig(combo_base_2panel + ".png", dpi=300)
fig.savefig(combo_base_2panel + ".pdf", bbox_inches='tight')
plt.show()

# ================= ADDED 3-PANEL figure (BEST seed + WORST seed + R2 vs PCA) =================
# Within BEST PCA, find BEST and WORST seeds by test R^2
seed_to_R2test = {s: perseed[best_n_pca][s]['R2_test'] for s in seed_list}
best_seed  = max(seed_to_R2test, key=seed_to_R2test.get)
worst_seed = min(seed_to_R2test, key=seed_to_R2test.get)

print(f"Best seed for PCA{best_n_pca} by test R^2:  seed={best_seed}  (R2_test={seed_to_R2test[best_seed]:.2f})")
print(f"Worst seed for PCA{best_n_pca} by test R^2: seed={worst_seed} (R2_test={seed_to_R2test[worst_seed]:.2f})")

d_best  = perseed[best_n_pca][best_seed]
d_worst = perseed[best_n_pca][worst_seed]

fig = plt.figure(figsize=(11.2, 3.0))
outer = GridSpec(1, 1)
inner = GridSpecFromSubplotSpec(
    1, 3,
    subplot_spec=outer[0],
    width_ratios=[1.0, 1.0, 1.0],
    wspace=0.28
)

# Panel 1: BEST seed parity
ax1 = fig.add_subplot(inner[0, 0])
parity_plot(
    ax1,
    d_best['y_train'], d_best['y_train_pred'],
    d_best['y_test'],  d_best['y_test_pred'],
    d_best['R2_train'], d_best['R2_test'],
    title=rf"Best seed={best_seed} (PCA{best_n_pca})",
    legend_loc='lower right'
)

# Panel 2: WORST seed parity
ax2 = fig.add_subplot(inner[0, 1])
parity_plot(
    ax2,
    d_worst['y_train'], d_worst['y_train_pred'],
    d_worst['y_test'],  d_worst['y_test_pred'],
    d_worst['R2_train'], d_worst['R2_test'],
    title=rf"Worst seed={worst_seed} (PCA{best_n_pca})",
    legend_loc='lower right'
)

# Panel 3: R^2 vs PCA components (mean ± std) [same as before]
ax3 = fig.add_subplot(inner[0, 2])
ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax3.yaxis.set_minor_locator(ticker.AutoMinorLocator())

ax3.errorbar(
    n_pca_comp_list,
    mean_train,
    yerr=std_train,
    fmt='-o', label='Train', capsize=4
)
ax3.errorbar(
    n_pca_comp_list,
    mean_test,
    yerr=std_test,
    fmt='-x', label='Test', capsize=4
)

ax3.set_xlabel(r'Number of PCA components', fontsize=14, fontweight='bold')
ax3.set_ylabel(r'$R^2$', fontsize=14, fontweight='bold')
ax3.ticklabel_format(axis='y', style='plain', useOffset=False)
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax3.set_ylim(-0.2, 1.4)
ax3.legend(frameon=False, fontsize=10, loc='lower right')

fig.tight_layout()
combo_base_3panel = "pca/gpr-fit-results/randomseed_bestPCA_bestWorstSeed_plus_r2"
fig.savefig(combo_base_3panel + ".png", dpi=300)
fig.savefig(combo_base_3panel + ".pdf", bbox_inches='tight')
plt.show()
