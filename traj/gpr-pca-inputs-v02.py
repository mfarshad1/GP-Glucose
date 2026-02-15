#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpflow
import scipy
import matplotlib.ticker as ticker
import tensorflow as tf
from matplotlib import rc
from sklearn.model_selection import train_test_split

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
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return float(r_value**2)

# ================= Settings =================
seed_list = list(range(0, 9, 1))      # same as original
n_pca_comp_list = list(range(2, 11))  # 2..10

R2_train_all = defaultdict(list)  # {n_pca: [R2_train_seed0, R2_train_seed1, ...]}
R2_test_all  = defaultdict(list)  # {n_pca: [R2_test_seed0,  R2_test_seed1,  ...]}

# ================= Main loops: over seeds, then PCA components =================
for seed in seed_list:
    print(f"\n===== Seed: {seed} =====")
    np.random.seed(seed)
    tf.random.set_seed(seed)

    for n_pca_comp in n_pca_comp_list:
        print(f"\nNumber of PCA components: {n_pca_comp}")

        # ----- Load PCA data (exactly as original) -----
        data = pd.read_csv(f"pca/pca{n_pca_comp}.csv")
        x_data = data.iloc[:, 1:1 + n_pca_comp].to_numpy()
        y_data = data["dG"].to_numpy().reshape(-1, 1)

        # ----- Train/test split: SAME as original script -----
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3, random_state=seed
        )

        # ----- GPR: SAME algorithm as original random-seed code -----
        kernel = (
            gpflow.kernels.Matern52(lengthscales=1.0, variance=1.0) \
            + gpflow.kernels.Polynomial()
            + gpflow.kernels.White()
        )

        model = gpflow.models.GPR((x_train, y_train), kernel=kernel)

        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            model.training_loss,
            model.trainable_variables
        )

        # ----- Predictions -----
        y_train_pred, _ = model.predict_f(x_train)
        y_test_pred,  _ = model.predict_f(x_test)

        y_train_pred = y_train_pred.numpy()
        y_test_pred  = y_test_pred.numpy()

        # ----- R^2 (per seed, per PCA) -----
        R2_train = rsquared(y_train, y_train_pred)
        R2_test  = rsquared(y_test,  y_test_pred)
        print(f"R2_train = {R2_train:.3f}, R2_test = {R2_test:.3f}")

        R2_train_all[n_pca_comp].append(R2_train)
        R2_test_all[n_pca_comp].append(R2_test)

        # ================= Parity plot for THIS seed & THIS PCA =================
        fig, ax = plt.subplots(figsize=(4, 3))
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
        ax.plot(
            [y_min - pad, y_max + pad],
            [y_min - pad, y_max + pad],
            'k-', lw=1.1
        )

        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        ax.set_xlabel(r"True $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
                      fontsize=16, fontweight='bold')
        ax.set_ylabel(r"Predicted $\Delta G\;(\mathrm{kcal\cdot mol^{-1}})$",
                      fontsize=16, fontweight='bold')
        ax.set_title(
            fr"{n_pca_comp} PCA components (seed={seed})",
            fontsize=16, fontweight='bold'
        )

        ax.legend(
            frameon=False, fontsize=12, loc='upper left',
            labelspacing=0.4, handlelength=0.8
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

# ================= Summary error-bar plot (styled like k-fold) =================
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
# ax.set_title(
#     r'GPR $R^2$ vs. PCA components (random seeds, Mean $\pm$ SD)',
#     fontsize=16, fontweight='bold'
# )

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
