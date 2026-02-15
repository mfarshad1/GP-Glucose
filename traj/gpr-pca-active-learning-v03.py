#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random-seed GP on PCA features + ranking of large set

Created on Tue Jul 29 15:45:23 2025
@author: mfarshad
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpflow
import scipy
import matplotlib.ticker as ticker
from matplotlib import rc
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ================= Plot properties (LaTeX-like, pretty) =================
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
# set data split random seed list
seeds = list(range(10))  # 0..9

# Number of PCA components to use: 2..10
n_pca_comp_list = list(range(2, 11))

# initialize list to store R^2 values averaged over seeds
R2_test_list = []
R2_train_list = []


# ================= Main loop over number of PCA components =================
for n_pca_comp in n_pca_comp_list:
    print(f"\nNumber of PCA components: {n_pca_comp}")

    # read PCA data
    data = pd.read_csv(f"pca/pca{n_pca_comp}.csv")

    # split into inputs and targets
    x_data = data.iloc[:, 1:1 + n_pca_comp].to_numpy()
    y_data = data["dG"].to_numpy().reshape(-1, 1)

    R2_test_seeds = []
    R2_train_seeds = []

    # placeholders for plotting from the last seed
    y_train_plot = None
    y_train_pred_plot = None
    y_test_plot = None
    y_test_pred_plot = None
    x_train_plot = None
    x_test_plot = None
    model_plot = None

    # ---------- loop over seeds ----------
    for seed in seeds:
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # split data into training and testing (80/20 split as requested)
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=seed
        )

        # ---------- build GP model (Matern32 + White, zero-mean) ----------
        kernel = (
            gpflow.kernels.Matern32(lengthscales=1.0, variance=1.0)
            # + gpflow.kernels.Polynomial()
            + gpflow.kernels.White()
        )

        model = gpflow.models.GPR((x_train, y_train), kernel=kernel)

        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            model.training_loss,
            model.trainable_variables
        )

        # ---------- predictions ----------
        fit_results = model.predict_f(x_train)
        y_train_pred = fit_results[0].numpy().flatten()
        sig_train = fit_results[1].numpy().flatten()

        fit_results = model.predict_f(x_test)
        y_test_pred = fit_results[0].numpy().flatten()
        sig_test = fit_results[1].numpy().flatten()

        # ---------- R^2 ----------
        R2_test = rsquared(y_test, y_test_pred)
        R2_train = rsquared(y_train, y_train_pred)

        print(f"[seed={seed}] R2_test = {R2_test:.3f}")
        print(f"[seed={seed}] R2_train = {R2_train:.3f}")

        R2_test_seeds.append(R2_test)
        R2_train_seeds.append(R2_train)

        # store for plotting from the last seed
        if seed == seeds[-1]:
            y_train_plot = y_train
            y_train_pred_plot = y_train_pred
            y_test_plot = y_test
            y_test_pred_plot = y_test_pred
            x_train_plot = x_train
            x_test_plot = x_test
            model_plot = model

    # ---------- seed-averaged R^2 ----------
    R2_test_mean = float(np.mean(R2_test_seeds))
    R2_train_mean = float(np.mean(R2_train_seeds))
    print(f"Mean over seeds -> R2_test = {R2_test_mean:.3f}")
    print(f"Mean over seeds -> R2_train = {R2_train_mean:.3f}")

    # store R^2 values
    R2_test_list.append(R2_test_mean)
    R2_train_list.append(R2_train_mean)

    # ================= Pretty parity plot (uses last-seed scatter, mean R² in labels) =================
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # hollow markers, like k-fold style
    ax.scatter(
        y_train_plot, y_train_pred_plot, s=22,
        facecolors="none", edgecolors="tab:blue", marker="o",
        linewidths=1.2, alpha=0.9,
        label=rf"Train ($R^2={R2_train_mean:.2f}$)"
    )
    ax.scatter(
        y_test_plot, y_test_pred_plot, s=26,
        facecolors="none", edgecolors="tab:red", marker="s",
        linewidths=1.4, alpha=0.9,
        label=rf"Test ($R^2={R2_test_mean:.2f}$)"
    )

    # y = x line
    y_all = np.concatenate([y_train_plot.flatten(), y_test_plot.flatten()])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad = 0.02 * (y_max - y_min + 1e-12)
    ax.plot(
        [y_min - pad, y_max + pad],
        [y_min - pad, y_max + pad],
        'k-', lw=1.1, label=r"$y=x$"
    )

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(r"True $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
                  fontsize=16, fontweight='bold')
    ax.set_ylabel(r"Predicted $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
                  fontsize=16, fontweight='bold')
    ax.set_title(
        fr"{n_pca_comp} PCA components (seed-averaged $R^2$)",
        fontsize=16, fontweight='bold'
    )

    ax.legend(
        frameon=False, fontsize=12, loc='upper left',
        labelspacing=0.4, handlelength=0.8
    )

    fig.tight_layout()
    base_parity = f"pca/gpr-fit-results/pca{n_pca_comp}_r2"
    fig.savefig(base_parity + ".png", dpi=300)
    fig.savefig(base_parity + ".pdf", bbox_inches='tight')
    plt.show()

    # ================= PC-wise model/data plots (kept, just slightly prettied) =================
    ncols = 3  # Fixed to 3 columns
    nrows = (n_pca_comp + ncols - 1) // ncols
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                            figsize=(ncols * 4.0, nrows * 3.5))
    axs = axs.flatten()

    for i in range(n_pca_comp):
        idx_sorted = np.argsort(x_data[:, i])
        x_sorted = x_data[idx_sorted, :]
        fit_results = model_plot.predict_f(x_sorted)
        y_sorted = fit_results[0].numpy().flatten()
        sig_sorted = fit_results[1].numpy().flatten()
        x_pc = x_sorted[:, i]

        ax = axs[i]
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # GP mean and 95% band
        ax.plot(x_pc, y_sorted, color='tab:blue', label='GPR mean')
        ax.fill_between(
            x_pc,
            y_sorted - 2 * sig_sorted,
            y_sorted + 2 * sig_sorted,
            alpha=0.2
        )

        # Data points for that PC
        ax.scatter(
            x_train_plot[:, i], y_train_plot,
            s=18, facecolors="none", edgecolors="tab:blue", linewidths=1.0,
            label='Train'
        )
        ax.scatter(
            x_test_plot[:, i], y_test_plot,
            s=20, facecolors="none", edgecolors="tab:red", linewidths=1.0,
            label='Test'
        )

        ax.set_xlabel(fr"PC {i+1}", fontsize=14, fontweight='bold')
        ax.set_ylabel(r'$\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$',
                      fontsize=14, fontweight='bold')
        ax.set_title(fr'Fit along PC {i+1}', fontsize=14, fontweight='bold')

        # Only put legend on first subplot to avoid clutter
        if i == 0:
            ax.legend(frameon=False, fontsize=10, loc='best')

    # Hide any unused subplots
    for j in range(n_pca_comp, len(axs)):
        axs[j].axis('off')

    fig.tight_layout()
    fig.savefig(f"pca/gpr-fit-results/pca{n_pca_comp}_gpr.png", dpi=300)
    plt.show()


# ================= R^2 vs PCA components summary =================
plt.figure(figsize=(4.0, 3.0))
plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.plot(n_pca_comp_list, R2_train_list, '-o', label='Train')
plt.plot(n_pca_comp_list, R2_test_list, '-x', label='Test')

plt.xlabel(r'Number of PCA components', fontsize=16, fontweight='bold')
plt.ylabel(r'$R^2$', fontsize=16, fontweight='bold')
plt.title(r'GPR $R^2$ vs. PCA components', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, frameon=False, loc='best')

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.tight_layout()
plt.savefig("pca/gpr-fit-results/r2_vs_pca.png", dpi=300)
plt.savefig("pca/gpr-fit-results/r2_vs_pca.pdf", bbox_inches='tight')
plt.show()

# save R^2 values to a csv
R2_df = pd.DataFrame({
    'n_pca_comp': n_pca_comp_list,
    'R2_train': R2_train_list,
    'R2_test': R2_test_list
})
R2_df.to_csv("pca/gpr-fit-results/r2_values.csv", index=False)

# --------------------------------------------------------------------
# NEW: choose PCA size with best mean test R^2 for ranking GP
# --------------------------------------------------------------------
best_idx = int(np.argmax(R2_test_list))
n_pca_rank = n_pca_comp_list[best_idx]
print(f"\nUsing n_pca_comp = {n_pca_rank} for ranking GP (best mean test R^2).")

# --------------------------------------------------------------------
# Perform "active-learning-style" ranking with a more flexible GP
# on the optimal PCA dimension
# --------------------------------------------------------------------

# 1) Load pca{n_pca_rank} data for ranking GP (trained on full labeled set)
data_rank = pd.read_csv(f"pca/pca{n_pca_rank}.csv")
x_rank = data_rank.iloc[:, 1:1 + n_pca_rank].to_numpy()
y_rank = data_rank["dG"].to_numpy().reshape(-1, 1)

D = x_rank.shape[1]
Q = 1

print("\nRanking GP: training label range:",
      float(y_rank.min()), "to", float(y_rank.max()))

# ---- Flexible GP: Matern32 with White noise ----
kernel_rank = (
    gpflow.kernels.Matern32(lengthscales=1.0, variance=1.0)
    # + gpflow.kernels.Polynomial()
    + gpflow.kernels.White()
)

# Keep noise tiny and fixed to interpolate extremes more strongly
kernel_rank.kernels[1].trainable = False  # White noise term

mean_fn_rank = gpflow.mean_functions.Linear(
    A=np.zeros((D, Q), dtype=gpflow.default_float()),
    b=np.array([y_rank.mean()], dtype=gpflow.default_float()),
)

model_rank = gpflow.models.GPR(
    (x_rank, y_rank),
    kernel=kernel_rank,
    mean_function=mean_fn_rank,
)

opt_rank = gpflow.optimizers.Scipy()
opt_rank.minimize(
    model_rank.training_loss,
    model_rank.trainable_variables,
)

# 2) Read large-set PCA data (same PC definition as chosen PCA: large-set-pca{n_pca_rank}.csv)
data = pd.read_csv(f"pca/large-set-pca{n_pca_rank}.csv")

x_data = data.iloc[:, 1:1 + n_pca_rank].to_numpy()
x_test = x_data

# 3) Predict on full large-set data using ranking model
fit_results = model_rank.predict_f(x_test)
y_test_pred = fit_results[0].numpy().flatten()
sig_test = fit_results[1].numpy().flatten()

print("\nRanking GP: prediction range on large set:",
      float(y_test_pred.min()), "to", float(y_test_pred.max()))

# get indices of data points with highest uncertainty
idx_sorted_asc = np.argsort(sig_test)
idx_sorted_desc = idx_sorted_asc[::-1]
x_sorted = x_test[idx_sorted_desc, :]
y_sorted = y_test_pred[idx_sorted_desc]
sig_sorted = sig_test[idx_sorted_desc]

print("\nTop 10 most uncertain data points:")
for i in range(10):
    print(f"Data point {i}, host {data['Real_Name'][idx_sorted_desc[i]]}:")
    print(f"  Predicted dG: {y_sorted[i]:.2f} +/- {sig_sorted[i]:.2f} kcal/mol")

# get indices of data points with lowest predicted dG (strongest binders)
idx_sorted = np.argsort(y_test_pred)
x_sorted = x_test[idx_sorted, :]
y_sorted = y_test_pred[idx_sorted]
sig_sorted = sig_test[idx_sorted]

print("\nBottom 10 dG predicted data points (strongest binders):")
for i in range(10):
    print(f"Data point {i}, host {data['Real_Name'][idx_sorted[i]]}:")
    print(f"  Predicted dG: {y_sorted[i]:.2f} +/- {sig_sorted[i]:.2f} kcal/mol")

print("\nAll Predictions (sorted by ΔG):")
all_predictions = pd.DataFrame({
    'Host': data['Real_Name'],
    'Predicted_dG': y_test_pred,
    'Uncertainty': sig_test
})

# Sort by ΔG (most negative first = strongest binders)
all_predictions_sorted = all_predictions.sort_values(by='Predicted_dG')

# Print all rows
pd.set_option('display.max_rows', None)
print(all_predictions_sorted.to_string(index=False))

# Save to CSV
all_predictions_sorted.to_csv("pca/gpr-fit-results/all_predictions.csv", index=False)
print("\nSaved all predictions to: pca/gpr-fit-results/all_predictions.csv")
