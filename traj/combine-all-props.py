'''
Combine data from different csvs
'''

import pandas as pd
import numpy as np
import os
import sys

# read pore size data
host_path = sys.argv[1]
host_name = host_path.split('/')[-1].split('.')[0].split('_')[-1]
# read csv and skip first line
pore_prof = pd.read_csv(host_path, skiprows=1, header=None).to_numpy().flatten()
# Read pore size data

# Read CSV and skip the first line
pore_prof = pd.read_csv(host_path, skiprows=1, header=None).dropna().to_numpy().flatten()

# get average and std of pore size
pore_avg = np.mean(pore_prof)
pore_var = np.std(pore_prof)

# read data from other csv
rdkit_prop = pd.read_csv(f'host-props/host_{host_name}_props.csv')

# combine data
rdkit_prop['pore_avg'] = pore_avg
rdkit_prop['pore_var'] = pore_var

# write to csv
rdkit_prop.to_csv(f'host-props/host_{host_name}_all_props.csv', index=False)
