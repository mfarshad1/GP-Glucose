'''
Combine data from different csvs
'''

import pandas as pd
import numpy as np
import os
import sys

# read pore size data
host_path = sys.argv[1]  # e.g., "host-32/diameter_profile_32.csv"

# Get the filename without path
filename = os.path.basename(host_path)  # e.g., "diameter_profile_32.csv" or "diameter_profile_host-12.csv"
base = filename.split('.')[0]

# 1) remove "diameter_profile_" prefix if present
if base.startswith('diameter_profile_'):
    tmp = base[len('diameter_profile_'):]
else:
    tmp = base

# 2) if what's left starts with "host-", strip that too
if tmp.startswith('host-'):
    host_name = tmp[len('host-'):]
else:
    host_name = tmp

print('name', host_name)

# directory where the HOLE CSVs live (same dir as diameter_profile)
host_dir = os.path.dirname(host_path)

# ==== use the TRUE distribution over all radii ====
dist_path = os.path.join(host_dir, f'diameter_distribution_{host_name}.csv')

# read diameter_all (all radii) and skip header
pore_prof = pd.read_csv(dist_path, skiprows=1, header=None).to_numpy().flatten()

# true global stats over all radii (matches HOLE)
pore_avg    = np.nanmean(pore_prof)
pore_std    = np.nanstd(pore_prof)
pore_median = np.nanmedian(pore_prof)

# read data from other csv
rdkit_prop = pd.read_csv(f'all-host-props/host_{host_name}_props.csv')

# combine data
rdkit_prop['pore_avg']    = pore_avg
rdkit_prop['pore_median'] = pore_median
rdkit_prop['pore_std']    = pore_std

# write to csv
os.makedirs('all-host-props-nopore', exist_ok=True)
rdkit_prop.to_csv(f'all-host-props-nopore/host_{host_name}_all_props.csv', index=False)

