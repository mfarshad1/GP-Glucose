'''
Combine data from different csvs
'''

import pandas as pd
import numpy as np
import os
import sys

# read pore size data
host_path = sys.argv[1]  # e.g., "host-mol2s/fixed_host_11.mol2"

# First get the filename without path or extension
#filename = host_path.split('/')[-1].split('.')[0]  # "fixed_host_11"

# Then split on '_' and take the last part
#host_name = filename.split('-')[-1]  # "11" or "asb11"
filename = os.path.basename(host_path)  # "diameter_profile_urea+cage5"
host_name = filename.split('diameter_profile_')[-1].split('.')[0]  # "urea+cage5"

print('name', host_name)  # Prints: name 11

# read csv and skip first line
pore_prof = pd.read_csv(host_path, skiprows=1, header=None).to_numpy().flatten()

# get average and std of pore size
pore_avg = np.mean(pore_prof)
#pore_var = np.std(pore_prof)
pore_median = np.median(pore_prof)

# Calculate the mean while ignoring NaNs
pore_avg = np.nanmean(pore_prof)
# Calculate the standard deviation while ignoring NaNs
pore_var = np.nanstd(pore_prof)

# read data from other csv
rdkit_prop = pd.read_csv(f'all-host-props/host_{host_name}_props.csv')

# combine data
rdkit_prop['pore_avg'] = pore_avg
rdkit_prop['pore_median'] = pore_var

# write to csv
rdkit_prop.to_csv(f'all-host-props-nopore/host_{host_name}_all_props.csv', index=False)
