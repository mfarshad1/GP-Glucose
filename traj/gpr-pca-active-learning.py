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
rc('font',**{'family':'serif','serif':['Computer Moder Roman']})
# rc('font',**{'family':'sans-serif'})
rc('axes', labelsize='14')
rc('xtick', labelsize='12')
rc('ytick', labelsize='12')

# set data split random seed
np.random.seed(4)
# Number of PCA components to use
# n_pca_comp_list = [2, 3, 4, 5]
n_pca_comp_list = [9]

# initialize list to store R^2 values
R2_test_list = []
R2_train_list = []

# loop over number of PCA components
for n_pca_comp in n_pca_comp_list:
    print(f"\nNumber of PCA components: {n_pca_comp}")

    # read PCA data
    data = pd.read_csv(f"pca/pca{n_pca_comp}.csv")
    #print(len(data))
    # split into inputs and targets
    x_data = data.iloc[:,1:1+n_pca_comp].to_numpy()
    y_data = data["dG"].to_numpy()

    # split data into training and testing
    N_train = int(round(0.75 * len(y_data)))
    # get random indices for training data with the specified random seed
    train_idx = np.random.choice(range(len(y_data)), replace=False, size=N_train)
    x_train = x_data[train_idx,:]
    y_train = np.reshape(y_data[train_idx], (-1,1))

    test_idx = np.setdiff1d(range(len(y_data)), train_idx)
    x_test = x_data[test_idx,:]
    y_test = np.reshape(y_data[test_idx], (-1,1))

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
        """ Return R^2 where x and y are array-like."""

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x.flatten(), y.flatten())
        return r_value**2
    R2_test = rsquared(y_test, y_test_pred)
    R2_train = rsquared(y_train, y_train_pred)
    
    print(f"R2_test = {R2_test}")
    print(f"R2_train = {R2_train}")

    # store R^2 values
    R2_test_list.append(R2_test)
    R2_train_list.append(R2_train)

    # plot R2 results
    plt.figure()
    plt.scatter(y_train, y_train_pred, color='blue', label=f'$R_{{train}}^2 = {round(R2_train, 2)}$')
    plt.scatter(y_test, y_test_pred, color='red', label=f'$R_{{test}}^2 = {round(R2_test, 2)}$')
    plt.plot(y_train, y_train, color='black', linestyle='--')
    plt.xlabel('True dG', fontsize=16, fontweight='bold')
    plt.ylabel('Predicted dG', fontsize=16, fontweight='bold')
    plt.title(f'{n_pca_comp} PCA components', fontsize=14, fontweight='bold')
    plt.legend(fontsize=14, loc='best')
    plt.savefig(f"pca/gpr-fit-results/pca{n_pca_comp}_r2.png")
    plt.show()

    # plot x-y model and data on a subplot
    ncols = (n_pca_comp if n_pca_comp < 4 else 3)
    nrows = (1 if n_pca_comp < 4 else 2)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*5, nrows*5))
    axs = axs.flatten()
    for i in range(n_pca_comp):
        # get sorted indices for plotting based on column i
        idx_sorted = np.argsort(x_data[:,i])
        x_sorted = x_data[idx_sorted,:]
        # get model predictions on entire dataset
        fit_results = model.predict_f(x_sorted)
        y_sorted = fit_results[0].numpy().flatten()
        sig_sorted = fit_results[1].numpy().flatten()
        # redefine x_sorted to just PC_i
        x_sorted = x_sorted[:,i]
        # plot effect of PCA_i on dG
        ax = axs[i]
        ax.plot(x_sorted, y_sorted, color='blue', label='GPR Model')
        ax.fill_between(x_sorted, y_sorted-2*sig_sorted, y_sorted+2*sig_sorted, color='blue', alpha=0.2)
        ax.scatter(x_train[:,i], y_train, color='blue', label=f'$R_{{train}}^2 = {round(R2_train, 2)}$')
        ax.scatter(x_test[:,i], y_test, color='red', label=f'$R_{{test}}^2 = {round(R2_test, 2)}$')
        ax.set_xlabel(f"PC {i+1}", fontsize=14, fontweight='bold')
        ax.set_ylabel('dG', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.set_title(f'Fit to PC {i+1}', fontsize=14, fontweight='bold')
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
# Perform Active Learning Using Last GP Model
# --------------------------------------------------------------------

# read full dataset PCA data
data = pd.read_csv(f"pca/large-set-pca{n_pca_comp}.csv")

# split into inputs and targets
x_data = data.iloc[:,1:1+n_pca_comp].to_numpy()
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
    # print(f"  PC: {x_sorted[i,:]}")
    print(f"  Predicted dG: {y_sorted[i]:.2f} +/- {sig_sorted[i]:.2f} kcal/mol")
    # print(f"  sigma: {sig_sorted[i]}")

# get indices of data points with lowest y_predicted
idx_sorted = np.argsort(y_test_pred)
x_sorted = x_test[idx_sorted,:]
y_sorted = y_test_pred[idx_sorted]
sig_sorted = sig_test[idx_sorted]

# print bottom 10 dG predicted data points
print("\nBottom 10 dG predicted data points:")
for i in range(10):
    print(f"Data point {i}, host {data['Real_Name'][idx_sorted[i]]}:")
    # print(f"  PC: {x_sorted[i,:]}")
    print(f"  Predicted dG: {y_sorted[i]:.2f} +/- {sig_sorted[i]:.2f} kcal/mol")
    # print(f"  sigma: {sig_sorted[i]}")