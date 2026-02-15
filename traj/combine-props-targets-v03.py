'''
Combine properties for all hosts, without targets
'''

import pandas as pd
import numpy as np
import os
import sys

# get all host names from dircetory names - list all directories starting with host-
host_files = [f for f in os.listdir('.') if f.startswith('host-')]
host_names = [f.split('-')[1] for f in host_files]
print('name', host_names, host_files)
# initialize empty dataframe 
all_data = pd.DataFrame()

for h, host_name in enumerate(host_names):
    # Read property data
    try:
        all_prop = pd.read_csv(f'all-host-props-nopore/host_{host_name}_all_props.csv', delimiter=',', header=0, index_col=0)
    except FileNotFoundError:
        print(f'Skipping host: {host_name}')
        continue

    # list all columns in all_prop
    cols = all_prop.columns

    # initialize empty columns in all_data
    if h == 0:
        for col in cols:
            all_data[col] = np.nan
        # Change name of first column to Real_Name
        all_data.rename(columns={all_data.columns[0]: 'Real_Name'}, inplace=True)
    
    # make df of just host name
    host_name_df = pd.DataFrame([host_name], columns=['Real_Name'], index=[host_name])

    # add host properties to all_data
    all_data.loc[host_name] = pd.concat([host_name_df.iloc[0], all_prop.iloc[0]])

    # for col in cols:
    #     all_data[col] = all_prop[col]

print(f'Extracted properties from {len(all_data)} hosts out of {len(host_names)}')

# write to csv
all_data.to_csv(f'all-host-props-nopore/all_host_props.csv', index=True)
