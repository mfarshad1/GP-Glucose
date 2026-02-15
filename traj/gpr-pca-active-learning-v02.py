#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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

# plot properties
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('axes', labelsize='14')
rc('xtick', labelsize='12')
rc('ytick', labelsize='12')

# set data split random seed
np.random.seed(0)
# Number of PCA components to use
n_pca_comp_list = list(range(2, 11))  # Changed to go from 2 to 10

# initialize list to store R^2 values
R2_test_list = []
R2_train_list = []

seeds = list(range(10))  # 0..9

# loop over number of PCA components
for n_pca_comp in n_pca_comp_list:
    print(f"\nNumber of PCA components: {n_pca_comp}")

    # read PCA data
    data = pd.read_csv(f"pca/pca{n_pca_comp}.csv")
    # split into inputs and targets
    x_data = data.iloc[:,1:1+n_pca_comp].to_numpy()
    y_data = data["dG"].to_numpy()

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

    for seed in seeds:
        np.random.seed(seed)

        # split data into training and testing
        N_train = int(round(0.8 * len(y_data)))
        train_idx = np.random.choice(range(len(y_data)), replace=False, size=N_train)
        x_train = x_data[train_idx,:]
        y_train = np.reshape(y_data[train_idx], (-1,1))

        test_idx = np.setdiff1d(range(len(y_data)), train_idx)
        x_test = x_data[test_idx,:]
        y_test = np.reshape(y_data[test_idx], (-1,1))

        # build GP model
        model = gpflow.models.GPR(
            (x_train, y_train),
            kernel=gpflow.kernels.Matern32() + gpflow.kernels.Polynomial() + gpflow.kernels.White() , 
            mean_function=None)
        
        # optimize model
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

        D = x_train.shape[1]
        Q = 1
        
        mean_fn = gpflow.mean_functions.Linear(
            A=np.zeros((D, Q), dtype=gpflow.default_float()),
            b=np.array([y_train.mean()], dtype=gpflow.default_float( )),  # shape (1,)
        )
        
        kernel = (gpflow.kernels.Matern32() + gpflow.kernels.Polynomial()) + gpflow.kernels.White()# drop White; use likelihood noise
        model = gpflow.models.GPR((x_train, y_train), kernel=kernel, mean_function=mean_fn)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

        # predict on training data
        fit_results = model.predict_f(x_train)
        y_train_pred = fit_results[0].numpy().flatten()
        sig_train = fit_results[1].numpy().flatten()

        # predict on testing data
        fit_results = model.predict_f(x_test)
        y_test_pred = fit_results[0].numpy().flatten()
        sig_test = fit_results[1].numpy().flatten()

        # Calculate prediction R^2
        def rsquared(x, y):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x.flatten(), y.flatten())
            return r_value**2
        R2_test = rsquared(y_test, y_test_pred)
        R2_train = rsquared(y_train, y_train_pred)
        
        print(f"[seed={seed}] R2_test = {R2_test}")
        print(f"[seed={seed}] R2_train = {R2_train}")

        R2_test_seeds.append(R2_test)
        R2_train_seeds.append(R2_train)

        if seed == seeds[-1]:
            # store for plotting
            y_train_plot = y_train
            y_train_pred_plot = y_train_pred
            y_test_plot = y_test
            y_test_pred_plot = y_test_pred
            x_train_plot = x_train
            x_test_plot = x_test
            model_plot = model

    # seed-averaged R^2
    R2_test = float(np.mean(R2_test_seeds))
    R2_train = float(np.mean(R2_train_seeds))
    print(f"Mean over seeds -> R2_test = {R2_test}")
    print(f"Mean over seeds -> R2_train = {R2_train}")

    # store R^2 values
    R2_test_list.append(R2_test)
    R2_train_list.append(R2_train)

    # plot R2 results (uses last-seed scatter; labels show seed-averaged R^2)
    plt.figure()
    plt.scatter(y_train_plot, y_train_pred_plot, color='blue', label=f'$R_{{train}}^2 = {round(R2_train, 2)}$')
    plt.scatter(y_test_plot, y_test_pred_plot, color='red', label=f'$R_{{test}}^2 = {round(R2_test, 2)}$')
    plt.plot(y_train_plot, y_train_plot, color='black', linestyle='--')
    plt.xlabel('True dG', fontsize=16, fontweight='bold')
    plt.ylabel('Predicted dG', fontsize=16, fontweight='bold')
    plt.title(f'{n_pca_comp} PCA components', fontsize=14, fontweight='bold')
    plt.legend(fontsize=14, loc='best')
    plt.savefig(f"pca/gpr-fit-results/pca{n_pca_comp}_r2.png")
    plt.show()

    # plot x-y model and data on a subplot
    ncols = 3  # Fixed to 3 columns for better visualization
    nrows = (n_pca_comp + ncols - 1) // ncols  # Calculate needed rows
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*5, nrows*5))
    axs = axs.flatten()
    for i in range(n_pca_comp):
        idx_sorted = np.argsort(x_data[:,i])
        x_sorted = x_data[idx_sorted,:]
        fit_results = model_plot.predict_f(x_sorted)
        y_sorted = fit_results[0].numpy().flatten()
        sig_sorted = fit_results[1].numpy().flatten()
        x_sorted = x_sorted[:,i]
        ax = axs[i]
        ax.plot(x_sorted, y_sorted, color='blue', label='GPR Model')
        ax.fill_between(x_sorted, y_sorted-2*sig_sorted, y_sorted+2*sig_sorted, color='blue', alpha=0.2)
        ax.scatter(x_train_plot[:,i], y_train_plot, color='blue', label=f'$R_{{train}}^2 = {round(R2_train, 2)}$')
        ax.scatter(x_test_plot[:,i], y_test_plot, color='red', label=f'$R_{{test}}^2 = {round(R2_test, 2)}$')
        ax.set_xlabel(f"PC {i+1}", fontsize=14, fontweight='bold')
        ax.set_ylabel('dG', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_title(f'Fit to PC {i+1}', fontsize=14, fontweight='bold')
    # Hide any unused subplots
    for j in range(n_pca_comp, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    plt.savefig(f"pca/gpr-fit-results/pca{n_pca_comp}_gpr.png")
    plt.show()

# plot R^2 values
plt.figure()
plt.plot(n_pca_comp_list, R2_train_list, '-o', color='blue', label='Train')
plt.plot(n_pca_comp_list, R2_test_list, '-x', color='red', label='Test')
plt.xlabel('Number of PCA components', fontsize=16, fontweight='bold')
plt.ylabel('$R^2$', fontsize=16, fontweight='bold')
plt.title('GPR $R^2$ vs. PCA components', fontsize=16, fontweight='bold')
plt.legend(fontsize=14, loc='best')
plt.savefig("pca/gpr-fit-results/r2_vs_pca.png")
plt.show()

# save R^2 values to a csv
R2_df = pd.DataFrame({'n_pca_comp': n_pca_comp_list, 'R2_train': R2_train_list, 'R2_test': R2_test_list})
R2_df.to_csv("pca/gpr-fit-results/r2_values.csv", index=False)


# --------------------------------------------------------------------
# Perform Active Learning Using Last GP Model (now uses n_pca_comp=10)
# --------------------------------------------------------------------

# read full dataset PCA data
data = pd.read_csv(f"pca/large-set-pca{n_pca_comp_list[-1]}.csv")  # Uses last component (10)

# split into inputs and targets
x_data = data.iloc[:,1:1+n_pca_comp_list[-1]].to_numpy()
x_test = x_data

# predict on testing data
fit_results = model.predict_f(x_test)
y_test_pred = fit_results[0].numpy().flatten()
sig_test = fit_results[1].numpy().flatten()

# get indices of data points with highest uncertainty
idx_sorted_asc = np.argsort(sig_test)
idx_sorted_desc = idx_sorted_asc[::-1]
x_sorted = x_test[idx_sorted_desc,:]
y_sorted = y_test_pred[idx_sorted_desc]
sig_sorted = sig_test[idx_sorted_desc]

# print top 10 most uncertain data points
print("\nTop 10 most uncertain data points:")
for i in range(10):
    print(f"Data point {i}, host {data['Real_Name'][idx_sorted_desc[i]]}:")
    print(f"  Predicted dG: {y_sorted[i]:.2f} +/- {sig_sorted[i]:.2f} kcal/mol")

# get indices of data points with lowest y_predicted
idx_sorted = np.argsort(y_test_pred)
x_sorted = x_test[idx_sorted,:]
y_sorted = y_test_pred[idx_sorted]
sig_sorted = sig_test[idx_sorted]

# print bottom 10 dG predicted data points
print("\nBottom 10 dG predicted data points:")
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

# Print all rows (set pandas to display all rows)
pd.set_option('display.max_rows', None)
print(all_predictions_sorted.to_string(index=False))

# Save to CSV
all_predictions_sorted.to_csv("pca/gpr-fit-results/all_predictions.csv", index=False)
print("\nSaved all predictions to: pca/gpr-fit-results/all_predictions.csv")
