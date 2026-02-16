import numpy as np
import glob
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
# importing Statistics module
import statistics
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from matplotlib import pyplot
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec  # <-- minimal new import for layout

# ================== User knob: number of PCs ==================
MAX_PCS = 10  # <-- change to 5, 6, 7, ... if you want fewer PCs

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font',**{'family':'serif','serif':['Computer Moder Roman']})
# rc('font',**{'family':'sans-serif'})
rc('axes', labelsize='18')
rc('xtick', labelsize='14')
rc('ytick', labelsize='14')

# Read the CSV file into a DataFrame
data = pd.read_csv('/afs/crc.nd.edu/user/m/mfarshad/Private/ML-new2/traj/all-host-props-nopore/all_host_props_training.csv')

# Use the first row (excluding the first column) as DataFrame df
df = data.iloc[:, 2:25:1]

# Explicitly exclude dG and some unused features (same as before)
features = [col for col in df.columns if col != 'dG']
features = [col for col in features if col != 'mean_ring_planarity' and col != 'mean_ring_planarity' and col != 'mean_ring_planarity']
# features = [col for col in features if col != 'num_sulfurs' and col != 'num_sulfurs' and col != 'num_sulfurs']
# %%
features = [col for col in features if col != 'num_electroneg_1_3_neighbors' and col != 'pore_std' and col != 'pore_std']


# Reference numbers for features (1..N) for plotting and LaTeX table
feature_ids = list(range(1, len(features) + 1))

# Standardizing ONLY the features (not dG)
x = StandardScaler().fit_transform(df[features].values)

# ================== PCA with loop over number of components ==================
pca_dict = {}
pc_data_dict = {}

# Fit PCA for k = 2..MAX_PCS
for k in range(2, MAX_PCS + 1):
    pca = PCA(n_components=k)
    pc_data = pca.fit_transform(x)
    pca_dict[k] = pca
    pc_data_dict[k] = pc_data

# Build DataFrames and save pca{k}.csv
for k in range(2, MAX_PCS + 1):
    cols = [f'principal component {i}' for i in range(1, k + 1)]
    principalDf = pd.DataFrame(data=pc_data_dict[k], columns=cols)
    principalDf.insert(0, "Real_Name", data["Real_Name"])

    finalDf = pd.concat([principalDf, df[['dG']]], axis=1)
    finalDf.to_csv(f'pca{k}.csv', index=False)

# Print explained variance ratios
for k in range(2, MAX_PCS + 1):
    print(f"pca{k} explained_variance_ratio_:", pca_dict[k].explained_variance_ratio_)

# For plots, use the last PCA (with MAX_PCS components)
pca_last = pca_dict[MAX_PCS]

pp = PdfPages('pca-explained-variance.pdf')

# ------------ PCA + loadings figure with tight GridSpec layout ------------
# width scales with MAX_PCS so bottom row has enough room
fig = plt.figure(figsize=(25, 5))
gs = GridSpec(2, MAX_PCS, height_ratios=[1.2, 1.0], hspace=0.5, wspace=0.4)

# Top row: variance (left half) and cumulative variance (right half)
mid = MAX_PCS // 2 if MAX_PCS > 2 else 1  # at least 1 column on left
ax_var = fig.add_subplot(gs[0, 0:mid])
ax_cum = fig.add_subplot(gs[0, mid:MAX_PCS])

ax_var.plot(
    range(1, MAX_PCS + 1),
    pca_last.explained_variance_ratio_[:MAX_PCS],
    '-o', color='blue'
)
ax_var.set_xlabel('Number of PCA components', fontsize=14, fontweight='bold')
ax_var.set_ylabel('Variance', fontsize=14, fontweight='bold', labelpad=5)

ax_cum.plot(
    range(1, MAX_PCS + 1),
    np.cumsum(pca_last.explained_variance_ratio_[:MAX_PCS]),
    '-o', color='blue'
)
ax_cum.set_xlabel('Number of PCA components', fontsize=14, fontweight='bold')
ax_cum.set_ylabel('Cumulative variance', fontsize=14, fontweight='bold', labelpad=5)
ax_cum.set_ylim(None, 1)

# Bottom row: MAX_PCS columns of loadings (numeric feature refs)
feature_weights = pca_last.components_  # shape: (MAX_PCS, n_features)
custom_colors = [
    'blue', 'orange', 'green', 'red', 'purple',
    'brown', 'pink', 'gray', 'olive', 'cyan',
    'magenta', 'yellow'
]

for i in range(MAX_PCS):
    ax = fig.add_subplot(gs[1, i])
    ax.bar(range(len(features)), feature_weights[i][:len(features)], color=custom_colors)
    ax.set_xticks(range(len(features)))
    # use numeric reference IDs instead of feature names
    ax.set_xticklabels(feature_ids, rotation=0, ha='center', fontsize=8)  # <-- changed
    ax.set_ylim(-0.8, 0.8)
    if i >= 1:
        ax.set_yticks([])
    else:
        ax.set_ylabel('Weight', fontsize=12, fontweight='bold', labelpad=-5)

fig.tight_layout()
pp.savefig(fig, bbox_inches='tight')
plt.show()

# ------------ Feature-importance figure (unchanged, but tight) ------------
# Get PCA results (already fitted using PCA(n_components=MAX_PCS))
explained_var = pca_last.explained_variance_ratio_  # shape: (MAX_PCS,)
loadings = pca_last.components_  # shape: (MAX_PCS, num_features)

# Compute feature importance as variance-weighted squared loadings
feature_importance = np.sum((explained_var[:, np.newaxis]) * (loadings ** 2), axis=0)

# Create DataFrame for sorting and labeling
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Map features to reference numbers for the importance plot
ref_map = {feat: idx+1 for idx, feat in enumerate(features)}
importance_df['Ref'] = importance_df['Feature'].map(ref_map)

# Plotting feature importance with numeric x-ticks
plt.figure(figsize=(12, 6))
bars = plt.bar(importance_df['Ref'], importance_df['Importance'], color='skyblue')
plt.xticks(
    importance_df['Ref'],
    importance_df['Ref'],
    rotation=45,
    ha='center',          # <-- changed here too
    fontsize=24
)
plt.ylabel('Variance-weighted squared loading', fontsize=12)
plt.xlabel('Feature reference number', fontsize=12)
plt.title('PCA Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

print(features)

plt.show()
pp.close()
