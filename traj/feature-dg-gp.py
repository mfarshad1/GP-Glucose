#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 22:50:37 2025

@author: mfarshad
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import gpflow
import scipy
import matplotlib.ticker as ticker
from matplotlib import rc
from sklearn.preprocessing import StandardScaler

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
np.random.seed(12345)

# Load original data with all features
data = pd.read_csv('/afs/crc.nd.edu/user/m/mfarshad/Private/ML/traj/all-host-props-nopore/all_host_props_training.csv')

# Extract original features (skip 'Real_Name', 'dG')
features = [col for col in data.columns if col not in ['Real_Name', 'dG']]
x_data = data[features].values
x_data = StandardScaler().fit_transform(x_data)

y_data = data["dG"].values

# split data into training and testing
N_train = int(round(0.7 * len(y_data)))
train_idx = np.random.choice(range(len(y_data)), replace=False, size=N_train)
test_idx = np.setdiff1d(range(len(y_data)), train_idx)

x_train = x_data[train_idx, :]
y_train = np.reshape(y_data[train_idx], (-1, 1))
x_test = x_data[test_idx, :]
y_test = np.reshape(y_data[test_idx], (-1, 1))

# build GP model
model = gpflow.models.GPR(
    (x_train, y_train),
        # kernel=gpflow.kernels.Matern12() * gpflow.kernels.Polynomial() * gpflow.kernels.Linear(),
        # kernel=gpflow.kernels.SquaredExponential(), # radial basis function
        kernel =gpflow.kernels.Matern32() + gpflow.kernels.White(0.1)
    )

# optimize model
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)

# predict on training and testing data
def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x.flatten(), y.flatten())
    return r_value**2

# Train predictions
fit_results = model.predict_f(x_train)
y_train_pred = fit_results[0].numpy().flatten()
sig_train = fit_results[1].numpy().flatten()
R2_train = rsquared(y_train, y_train_pred)

# Test predictions
fit_results = model.predict_f(x_test)
y_test_pred = fit_results[0].numpy().flatten()
sig_test = fit_results[1].numpy().flatten()
R2_test = rsquared(y_test, y_test_pred)

print(f"\nGPR using original features:")
print(f"R2_train = {R2_train}")
print(f"R2_test = {R2_test}")

# plot results
plt.figure()
plt.scatter(y_train, y_train_pred, color='blue', label=f'$R_{{train}}^2 = {round(R2_train, 2)}$')
plt.scatter(y_test, y_test_pred, color='red', label=f'$R_{{test}}^2 = {round(R2_test, 2)}$')
plt.plot(y_train, y_train, color='black', linestyle='--')
plt.xlabel('True dG', fontsize=16, fontweight='bold')
plt.ylabel('Predicted dG', fontsize=16, fontweight='bold')
plt.title('GPR on Original Features', fontsize=14, fontweight='bold')
plt.legend(fontsize=14, loc='best')
plt.tight_layout()
plt.savefig("pca/gpr-fit-results/gpr_original_features.png")
plt.show()
