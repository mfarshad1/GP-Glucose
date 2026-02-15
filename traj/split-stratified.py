"""
Split dataset into training and testing using using stratified k-folds.
The input csv file, name of target column and number of folds are user-
provided arguments.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import argparse, os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import rc

# =============================================================================
# Plot Configuration
# =============================================================================

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
plt.rcParams["font.family"] = "Serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
rc('axes', labelsize='16')
rc('xtick', labelsize='14')
rc('ytick', labelsize='14')
rc('legend', fontsize='12')
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams["savefig.pad_inches"]=0.02

# =============================================================================
# Functions
# =============================================================================

def split_data(csv_path, target_col_name, n_splits, n_bins, out_dir):
    # read the property data
    df = pd.read_csv(csv_path,index_col=0)
    prop = df[target_col_name]

    # Initialize a stratified k-fold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cut the property data into discrete bins
    prop_cut = pd.cut(prop, bins=n_bins, labels=range(n_bins))

    # Create a placeholder input array
    X = df.drop(columns=[target_col_name], axis=1)
    
    os.makedirs(out_dir, exist_ok=True)

    # Loop over the training and testing indices
    for k, (train_idx, test_idx) in enumerate(skf.split(X, prop_cut)):
    
        # Save the training and testing property data to a csv file
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        train_df.to_csv(f"{out_dir}/train_fold_{k+1}.csv", index=True)
        test_df.to_csv(f"{out_dir}/test_fold_{k+1}.csv", index=True)

def visualize_distns(target_col_name, n_splits, n_bins,  out_dir):
    # Initialize a plot to visualize the distribution of the property data
    nrows = 2
    ncols = int(np.ceil(n_splits/2))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    axs = axs.flatten()
    
    # Loop over the training and testing indices
    for k in range(n_splits):
        # Read the training and testing property data to a csv file
        train_df = pd.read_csv(f"{out_dir}/train_fold_{k+1}.csv", index_col=0)
        test_df = pd.read_csv(f"{out_dir}/test_fold_{k+1}.csv", index_col=0)
        y_train = train_df[target_col_name]
        y_test = test_df[target_col_name]

        # Plot the distribution of the property data
        ax = axs[k]
        ax.hist(y_train, bins=n_bins, alpha=0.5, color='b', label='Training')
        ax.hist(y_test, bins=n_bins, alpha=0.5, color='r', label='Testing')
        ax.set_xlabel(target_col_name)
        ax.set_ylabel('Frequency')
        if k == 0: ax.legend(loc='upper left')
        ax.set_title(f'Fold {k+1}')
    # Add plot attributes for unnormalized property data
    fig.suptitle(f'Target Distribution for Training and Testing Sets', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/PropertyDistribution.png')
    # plt.show()
    # fig.close()

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    split_data(args.csv, args.target, args.folds, args.bins, args.dir)
    visualize_distns(args.target, args.folds, args.bins, args.dir)