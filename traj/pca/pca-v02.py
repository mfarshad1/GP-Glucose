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

# [All previous imports and setup remain exactly the same...]

# Read the CSV file into a DataFrame
data = pd.read_csv('/afs/crc.nd.edu/user/m/mfarshad/Private/ML-new/traj/all-host-props-nopore/all_host_props_training.csv')

# Use the first row (excluding the first column) as DataFrame df
df = data.iloc[:, 1:18:1]

# Explicitly exclude dG from features
features = [col for col in df.columns if col != 'dG']  # This ensures dG is excluded
features = features[0:16]  # Remove the first feature

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

finalDf2 = pd.concat([principalDf2, df[['dG']]], axis=1)
finalDf3 = pd.concat([principalDf3, df[['dG']]], axis=1)
finalDf4 = pd.concat([principalDf4, df[['dG']]], axis=1)
finalDf5 = pd.concat([principalDf5, df[['dG']]], axis=1)

finalDf2.to_csv('pca2.csv', index=False)  # ✅
finalDf3.to_csv('pca3.csv', index=False)  # ✅
finalDf4.to_csv('pca4.csv', index=False)  # ✅
finalDf5.to_csv('pca5.csv', index=False)  # ✅

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
# ax.set_xlabel('Principal Component 1', fontsize=15)
# ax.set_ylabel('Principal Component 2', fontsize=15)
# ax.set_title('2 component PCA', fontsize=20)

# target = df.iloc[:, -1]
# targets = target.unique()  # Extract unique targets
# colors = sns.color_palette("bright", len(targets))

# for target, color in zip(targets, colors):
#     indicesToKeep = finalDf['dG'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c=color
#                , s=50)
# ax.legend(targets)
# ax.grid()

print(pca2.explained_variance_ratio_)
print(pca3.explained_variance_ratio_)
print(pca4.explained_variance_ratio_)
print(pca5.explained_variance_ratio_)

pp = PdfPages('pca-explained-variance.pdf')
fig, axes = plt.subplots(2, 2, figsize=(20, 7))

# First row: one column
axes[0,0].plot([1, 2, 3, 4, 5], pca5.explained_variance_ratio_, '-o', color='blue')
axes[0,0].set_xlabel('Number of PCA components', fontsize=14, fontweight='bold')
axes[0,0].set_ylabel('Variance', fontsize=14, fontweight='bold', labelpad=5)
axes[0,1].plot([1, 2, 3, 4, 5], np.cumsum(pca5.explained_variance_ratio_), '-o', color='blue')
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
    'magenta', 'yellow'  # Add more custom colors as needed
]

# Second row: five columns
feature_weights = pca5.components_
for i in range(5):
    colors = sns.color_palette("tab10", len(feature_weights[i]))
    ax = fig.add_subplot(2, 5, i+6)  # Add subplots to the second row
    # Use the same number of bars and labels (12 in this case)
    ax.bar(range(len(features)), feature_weights[i][:len(features)], color=custom_colors)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(-0.6,0.6)
    plt.subplots_adjust(hspace=0.3,wspace=0.1)
    if i >= 1:
        ax.set_yticks([])
    else:
        ax.set_ylabel('Weight', fontsize=14, fontweight='bold', labelpad=-5)
        
plt.tight_layout()
plt.show()
pp.savefig(fig, bbox_inches='tight')
pp.close()
