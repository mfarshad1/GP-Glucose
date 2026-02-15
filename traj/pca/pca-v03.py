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

# Read the CSV file into a DataFrame
data = pd.read_csv('/afs/crc.nd.edu/user/m/mfarshad/Private/ML-new2/traj/all-host-props-nopore/all_host_props_filtered.csv')

# Use the first row (excluding the first column) as DataFrame df
df = data.iloc[:, 2:25:1]

# Explicitly exclude dG from features
features = [col for col in df.columns if col != 'dG' and col != 'dG']  # This ensures dG is excluded
features = [col for col in features if col != 'mean_ring_planarity' and col != 'mean_ring_planarity' and col != 'mean_ring_planarity']
# features = [col for col in features if col != 'num_sulfurs' and col != 'num_sulfurs' and col != 'num_sulfurs']
features = [col for col in features if col != 'num_electroneg_1_3_neighbors' and col != 'pore_std' and col != 'pore_std']


# Standardizing ONLY the features (not dG)
x = StandardScaler().fit_transform(df[features].values)  # Changed this line


pca2 = PCA(n_components=2)
principalComponents2 = pca2.fit_transform(x)

pca3 = PCA(n_components=3)
principalComponents3 = pca3.fit_transform(x)

pca4 = PCA(n_components=4)
principalComponents4 = pca4.fit_transform(x)

pca5 = PCA(n_components=5)
principalComponents5 = pca5.fit_transform(x)

# New PCA components 6-10
pca6 = PCA(n_components=6)
principalComponents6 = pca6.fit_transform(x)

pca7 = PCA(n_components=7)
principalComponents7 = pca7.fit_transform(x)

pca8 = PCA(n_components=8)
principalComponents8 = pca8.fit_transform(x)

pca9 = PCA(n_components=9)
principalComponents9 = pca9.fit_transform(x)

pca10 = PCA(n_components=10)
principalComponents10 = pca10.fit_transform(x)

principalDf2 = pd.DataFrame(data=principalComponents2, columns=['principal component 1', 
                'principal component 2'])
principalDf2.insert(0, "Real_Name", data["Real_Name"])

principalDf3 = pd.DataFrame(data=principalComponents3, columns=['principal component 1', 
                'principal component 2','principal component 3'])
principalDf3.insert(0, "Real_Name", data["Real_Name"])

principalDf4 = pd.DataFrame(data=principalComponents4, columns=['principal component 1', 
                'principal component 2','principal component 3','principal component 4'])
principalDf4.insert(0, "Real_Name", data["Real_Name"])

principalDf5 = pd.DataFrame(data=principalComponents5, columns=['principal component 1', 
                'principal component 2','principal component 3','principal component 4','principal component 5'])
principalDf5.insert(0, "Real_Name", data["Real_Name"])

# New DataFrames for components 6-10
principalDf6 = pd.DataFrame(data=principalComponents6, columns=['principal component 1', 
                'principal component 2','principal component 3','principal component 4','principal component 5',
                'principal component 6'])
principalDf6.insert(0, "Real_Name", data["Real_Name"])

principalDf7 = pd.DataFrame(data=principalComponents7, columns=['principal component 1', 
                'principal component 2','principal component 3','principal component 4','principal component 5',
                'principal component 6','principal component 7'])
principalDf7.insert(0, "Real_Name", data["Real_Name"])

principalDf8 = pd.DataFrame(data=principalComponents8, columns=['principal component 1', 
                'principal component 2','principal component 3','principal component 4','principal component 5',
                'principal component 6','principal component 7','principal component 8'])
principalDf8.insert(0, "Real_Name", data["Real_Name"])

principalDf9 = pd.DataFrame(data=principalComponents9, columns=['principal component 1', 
                'principal component 2','principal component 3','principal component 4','principal component 5',
                'principal component 6','principal component 7','principal component 8','principal component 9'])
principalDf9.insert(0, "Real_Name", data["Real_Name"])

principalDf10 = pd.DataFrame(data=principalComponents10, columns=['principal component 1', 
                'principal component 2','principal component 3','principal component 4','principal component 5',
                'principal component 6','principal component 7','principal component 8','principal component 9',
                'principal component 10'])
principalDf10.insert(0, "Real_Name", data["Real_Name"])

# Write to csv
principalDf2.to_csv('large-set-pca2.csv', index=False)
principalDf3.to_csv('large-set-pca3.csv', index=False)
principalDf4.to_csv('large-set-pca4.csv', index=False)
principalDf5.to_csv('large-set-pca5.csv', index=False)
principalDf6.to_csv('large-set-pca6.csv', index=False)
principalDf7.to_csv('large-set-pca7.csv', index=False)
principalDf8.to_csv('large-set-pca8.csv', index=False)
principalDf9.to_csv('large-set-pca9.csv', index=False)
principalDf10.to_csv('large-set-pca10.csv', index=False)

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
# ax.set_title('2 component PCA', fontsize=20)

# target = df.iloc[:, -1]
# targets = target.unique()  # Extract unique targets
# colors = sns.color_palette("bright", len(targets))

# for target, color in zip(targets, colors):
#     indicesToKeep = principalDf['dG'] == target
#     ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
#                , principalDf.loc[indicesToKeep, 'principal component 2']
#                , c=color
#                , s=50)
# ax.legend(targets)
# ax.grid()

print(pca2.explained_variance_ratio_)
print(pca3.explained_variance_ratio_)
print(pca4.explained_variance_ratio_)
print(pca5.explained_variance_ratio_)
print(pca6.explained_variance_ratio_)
print(pca7.explained_variance_ratio_)
print(pca8.explained_variance_ratio_)
print(pca9.explained_variance_ratio_)
print(pca10.explained_variance_ratio_)

pp = PdfPages('pca-explained-variance.pdf')
fig, axes = plt.subplots(2, 2, figsize=(40, 7))

# First row: one column
axes[0,0].plot(range(1, 11), pca10.explained_variance_ratio_[:10], '-o', color='blue')
axes[0,0].set_xlabel('Number of PCA components', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Variance', fontsize=14, fontweight='bold', labelpad=5)
axes[0,1].plot(range(1, 11), np.cumsum(pca10.explained_variance_ratio_[:10]), '-o', color='blue')
axes[0,1].set_xlabel('Number of PCA components', fontsize=14, fontweight='bold')
axes[0,1].set_ylabel('Cumulative variance', fontsize=14, fontweight='bold', labelpad=5)
axes[0,1].set_ylim(None,1)

axes[1,0].set_xticks([])
axes[1,0].set_yticks([])
axes[1,0].axis('off')
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])
axes[1,1].axis('off')

# Define custom colors using named colors
custom_colors = [
    'blue', 'orange', 'green', 'red', 'purple', 
    'brown', 'pink', 'gray', 'olive', 'cyan', 
    'magenta', 'yellow', 'lime'  # Add more custom colors as needed
]

# Second row: ten columns
feature_weights = pca10.components_
for i in range(10):
    colors = sns.color_palette("tab10", len(feature_weights[i]))
    ax = fig.add_subplot(2, 10, i+11)  # Add subplots to the second row
    # Use the same number of bars and labels (12 in this case)
    ax.bar(range(len(features)), feature_weights[i][:len(features)], color=custom_colors)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(-0.8,0.8)
    plt.subplots_adjust(hspace=0.3,wspace=0.1)
    if i >= 1:
        ax.set_yticks([])
    else:
        ax.set_ylabel('Weight', fontsize=14, fontweight='bold', labelpad=-5)

# Get PCA results (already fitted using PCA(n_components=10))
explained_var = pca10.explained_variance_ratio_  # shape: (10,)
loadings = pca10.components_  # shape: (10, num_features)z
