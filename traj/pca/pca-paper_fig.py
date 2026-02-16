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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import colorsys
from matplotlib.ticker import MultipleLocator

# ================== User knob: number of PCs ==================
MAX_PCS = 6  # fixed to 6 PCA components

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))

rc('text', usetex=True)
rc('ps', usedistiller='xpdf')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('axes', labelsize=24)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)

# Read the CSV file into a DataFrame
data = pd.read_csv(
    '/afs/crc.nd.edu/user/m/mfarshad/Private/ML-new2/traj/all-host-props-nopore/all_host_props_training.csv'
)

# Use the first row (excluding the first column) as DataFrame df
df = data.iloc[:, 1:25:1]

# Explicitly exclude dG and some unused features (same as before)
features = [col for col in df.columns if col != 'dG']
features = [col for col in features if col != 'mean_ring_planarity' and col != 'mean_ring_planarity' and col != 'mean_ring_planarity']
# features = [col for col in features if col != 'num_sulfurs' and col != 'num_sulfurs' and col != 'num_sulfurs']
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

# ============================================================
# 1) PCA variance + loadings figure (aligned rows)
# ============================================================
fig1 = plt.figure(figsize=(15, 8))

# Outer grid: band 0 = top row, band 1 = block containing rows 2+3
outer = GridSpec(
    2, 1,
    height_ratios=[1.0, 1.6],  # relative heights for top vs bottom block
    hspace=0.25             # gap between row 1 and the loadings block
)

# ----- Top band: 1×2 grid (variance, cumulative variance) -----
top_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.2)

ax_var = fig1.add_subplot(top_gs[0, 0])
ax_cum = fig1.add_subplot(top_gs[0, 1])

ax_var.plot(
    range(1, MAX_PCS + 1),
    pca_last.explained_variance_ratio_[:MAX_PCS],
    '-o'
)
ax_var.set_xlabel('Number of PCA components', fontsize=20)
ax_var.set_ylabel('Variance', fontsize=20, labelpad=10)

# ----- MINIMAL CHANGE: auto y-lim for variance (better than hard-coded) -----
v = pca_last.explained_variance_ratio_[:MAX_PCS]
vmax = float(np.max(v))
ax_var.set_ylim(0.0, vmax * 1.10)  # 10% headroom

ax_cum.plot(
    range(1, MAX_PCS + 1),
    np.cumsum(pca_last.explained_variance_ratio_[:MAX_PCS]),
    '-o'
)
ax_cum.set_xlabel('Number of PCA components', fontsize=20)
ax_cum.set_ylabel('Cumulative variance', fontsize=20, labelpad=5)
ax_cum.set_ylim(0.2, 1.0)

# x-ticks at 1,2,3,4,5,6
ax_var.xaxis.set_major_locator(MultipleLocator(1))
ax_cum.xaxis.set_major_locator(MultipleLocator(1))

# y-ticks
ax_var.yaxis.set_major_locator(MultipleLocator(0.05))
ax_cum.yaxis.set_major_locator(MultipleLocator(0.2))

# ----- Second block: rows 2 and 3 inside outer[1] -----
# inner 2×1 grid controls ONLY the gap between rows 2 and 3
inner = GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=outer[1],
    hspace=0.0      # small gap between second and third rows
)

mid_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=inner[0], wspace=0.0)
bot_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=inner[1], wspace=0.0)

feature_weights = pca_last.components_  # shape: (MAX_PCS, n_features)

# Color per FEATURE (consistent across all PCs)
cmap = plt.get_cmap('tab20')
feature_colors = [cmap(i % cmap.N) for i in range(len(features))]

# Row 2: PC1–PC3
for pc_index in range(3):
    ax = fig1.add_subplot(mid_gs[0, pc_index])
    ax.bar(
        range(len(features)),
        feature_weights[pc_index][:len(features)],
        color=feature_colors
    )
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(feature_ids, rotation=0, ha='center', fontsize=10)
    ax.set_ylim(-0.9, 0.9)

    # ----- MINIMAL CHANGE: PC label slightly lower -----
    ax.text(
        0.02, 0.92, f"PC{pc_index + 1}",  # was 0.98
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=14, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
    )

    if pc_index > 0:
        ax.set_yticks([])
    else:
        ax.set_ylabel('Weight', fontsize=20, fontweight='bold', labelpad=5)

# Row 3: PC4–PC6
for pc_index in range(3, 6):
    ax = fig1.add_subplot(bot_gs[0, pc_index - 3])
    ax.bar(
        range(len(features)),
        feature_weights[pc_index][:len(features)],
        color=feature_colors
    )
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(feature_ids, rotation=0, ha='center', fontsize=10)
    ax.set_ylim(-0.9, 0.9)

    # ----- MINIMAL CHANGE: PC label slightly lower -----
    ax.text(
        0.02, 0.92, f"PC{pc_index + 1}",  # was 0.98
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=14, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)
    )

    if pc_index > 3:
        ax.set_yticks([])
    else:
        ax.set_ylabel('Weight', fontsize=20, fontweight='bold', labelpad=5)

    # bottom row gets the x-label
    ax.set_xlabel('Feature reference number', fontsize=20)

fig1.tight_layout()
pp.savefig(fig1, bbox_inches='tight')
plt.show()

# ============================================================
# 2) Feature-importance figure
# ============================================================
explained_var = pca_last.explained_variance_ratio_
loadings = pca_last.components_

# variance-weighted squared loadings
feature_importance = np.sum(
    (explained_var[:, np.newaxis]) * (loadings ** 2),
    axis=0
)

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

ref_map = {feat: idx + 1 for idx, feat in enumerate(features)}
importance_df['Ref'] = importance_df['Feature'].map(ref_map)

fig2 = plt.figure(figsize=(12, 6))
plt.bar(importance_df['Ref'], importance_df['Importance'], color='skyblue')
plt.xticks(
    importance_df['Ref'],
    importance_df['Ref'],
    rotation=0,
    ha='center',
    fontsize=14
)
plt.ylabel('Variance-weighted squared loading', fontsize=20)
plt.xlabel('Feature reference number', fontsize=20)
plt.tight_layout()

print(features)

plt.show()
pp.savefig(fig2, bbox_inches='tight')
pp.close()
