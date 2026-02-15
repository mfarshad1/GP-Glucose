'''
Combine property and target data 
'''

import pandas as pd
import numpy as np
import os
import sys

# Read target data
target = pd.read_csv('targets/output.txt', delimiter=' ', header=0, index_col=0)

# select column to extract (column 1)
select_dG = 2
dG = target.iloc[:, select_dG]

# initialize empty dataframe with same index as target
all_data = pd.DataFrame(index=dG.index)

# get host names from index column
host_names = dG.index
host_files = [f for f in os.listdir('.') if f.startswith('host-')]
host_names = [f.split('-')[1] for f in host_files]

for h, host_name in enumerate(host_names):
    # Read property data
    all_prop = pd.read_csv(f'all-host-props-nopore/host_{host_name}_all_props.csv', delimiter=',', header=0, index_col=0)
 
    # list all columns in all_prop
    cols = all_prop.columns

    # initialize empty columns in all_data
    if h == 0:
        for col in cols:
            all_data[col] = np.nan
        # add a column for the target
        all_data['dG'] = np.nan
    
    #loop through columns and add to all_data
    for col in cols:
        all_data[col][host_name] = all_prop[col][host_name]

    # add target value
    all_data['dG'][host_name] = dG[host_name]

# write to csv
all_data.to_csv(f'host-props-nopore/all_host_props.csv', index=True)
