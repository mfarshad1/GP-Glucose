import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import gpflow
import scipy
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker
from matplotlib import rc

# plot properties
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font',**{'family':'serif','serif':['Computer Moder Roman']})
rc('axes', labelsize='14')
rc('xtick', labelsize='12')
rc('ytick', labelsize='12')

# Number of PCA components and seeds
n_pca_comp_list = list(range(2, 11))
seed_list = list(range(0, 10,1))

# Initialize R² lists
R2_test_list = []
R2_train_list = []

for n_pca_comp in n_pca_comp_list:
    print(f"\nNumber of PCA components: {n_pca_comp}")

    # Load and normalize data
    data = pd.read_csv(f"pca/pca{n_pca_comp}.csv")
    x_data = data.iloc[:,1:1+n_pca_comp].to_numpy()
    y_data = data["dG"].to_numpy()
    
    x_scaler = StandardScaler()
    x_data = x_scaler.fit_transform(x_data)

    R2_train_seeds = []
    R2_test_seeds = []

    for seed in seed_list:
        np.random.seed(seed)

        N_train = int(round(0.8 * len(y_data)))
        train_idx = np.random.choice(range(len(y_data)), replace=False, size=N_train)
        x_train = x_data[train_idx,:]
        y_train = np.reshape(y_data[train_idx], (-1,1))
        y_train += 1e-4 * np.random.randn(*y_train.shape)  # noise regularization

        test_idx = np.setdiff1d(range(len(y_data)), train_idx)
        x_test = x_data[test_idx,:]
        y_test = np.reshape(y_data[test_idx], (-1,1))

        # ARD-enabled RBF kernel + noise
        kernel=gpflow.kernels.RBF() + (gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=3000000)) + \
                        gpflow.kernels.White()
        model = gpflow.models.GPR((x_train, y_train), kernel=kernel)

        # Force minimal noise
        model.kernel.kernels[-1].variance.assign(max(model.kernel.kernels[-1].variance.numpy(), 1e-2))

        # Optimize
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

        # Predict
        y_train_pred = model.predict_f(x_train)[0].numpy().flatten()
        y_test_pred = model.predict_f(x_test)[0].numpy().flatten()

        def rsquared(x, y):
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x.flatten(), y.flatten())
            return r_value**2

        R2_train_seeds.append(rsquared(y_train, y_train_pred))
        R2_test_seeds.append(rsquared(y_test, y_test_pred))

    # Average over seeds
    R2_train = np.mean(R2_train_seeds)
    R2_test = np.mean(R2_test_seeds)

    print(f"Avg R2_test = {R2_test:.3f}, Avg R2_train = {R2_train:.3f}")
    R2_train_list.append(R2_train)
    R2_test_list.append(R2_test)

# Plot R² vs PCA components
plt.figure()
plt.plot(n_pca_comp_list, R2_train_list, '-o', color='blue', label='Train')
plt.plot(n_pca_comp_list, R2_test_list, '-x', color='red', label='Test')
plt.xlabel('Number of PCA components', fontsize=16, fontweight='bold')
plt.ylabel('$R^2$', fontsize=16, fontweight='bold')
plt.title('GPR $R^2$ vs. PCA components', fontsize=16, fontweight='bold')
plt.legend(fontsize=14, loc='best')
plt.savefig("pca/gpr-fit-results/r2_vs_pca.png")
plt.show()

# Save to CSV
R2_df = pd.DataFrame({'n_pca_comp': n_pca_comp_list, 'R2_train': R2_train_list, 'R2_test': R2_test_list})
R2_df.to_csv("pca/gpr-fit-results/r2_values.csv", index=False)
