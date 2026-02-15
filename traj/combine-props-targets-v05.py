"""
Combine properties for all hosts (keeping targets),
and FORCE column order so that:
    Real_Name, paper_name, dG, ...
"""

import pandas as pd
import numpy as np
import os

# get all host names from directory names - list all directories starting with host-
host_files = [f for f in os.listdir('.') if f.startswith('host-')]
host_names = [f.split('-', 1)[1] for f in host_files]
print('name', host_names, host_files)

# initialize empty dataframe
all_data = pd.DataFrame()

for h, host_name in enumerate(host_names):
    # Read property data
    try:
        all_prop = pd.read_csv(
            f'all-host-props-nopore/host_{host_name}_all_props.csv',
            delimiter=',',
            header=0
        )
    except FileNotFoundError:
        print(f'Skipping host: {host_name}')
        continue

    # ---- NEW: drop redundant identifier cols (prevents extra host_name column) ----
    drop_cols = [c for c in ['host_name', 'Host', 'host', 'name'] if c in all_prop.columns]
    if drop_cols:
        all_prop.drop(columns=drop_cols, inplace=True)

    # Standardize name columns
    if 'real_name' in all_prop.columns:
        all_prop.rename(columns={'real_name': 'Real_Name'}, inplace=True)
    elif 'Real_Name' not in all_prop.columns:
        # fallback if not present for some files
        all_prop.insert(0, 'Real_Name', host_name)

    # Ensure paper_name exists (fallback)
    if 'paper_name' not in all_prop.columns:
        all_prop.insert(1, 'paper_name', np.nan)

    # Append (each host file should be 1 row)
    all_data = pd.concat([all_data, all_prop], ignore_index=True)

print(f'Extracted properties from {len(all_data)} hosts out of {len(host_names)}')

# ---- FORCE the desired column order: Real_Name, paper_name, target, rest ----
cols = list(all_data.columns)

# Normalize: if you happened to have 'Paper_Name' instead of 'paper_name'
if 'Paper_Name' in cols and 'paper_name' not in cols:
    all_data.rename(columns={'Paper_Name': 'paper_name'}, inplace=True)
    cols = list(all_data.columns)

# Identify target column name(s) you might be using
target_candidates = ['dG', 'DG', 'deltaG', 'DeltaG', 'target', 'Target']
target_col = next((c for c in target_candidates if c in cols), None)

# Build ordered list
ordered = []
for c in ['Real_Name', 'paper_name']:
    if c in cols:
        ordered.append(c)
if target_col is not None:
    ordered.append(target_col)

# Add everything else, preserving existing order
ordered += [c for c in cols if c not in ordered]

# Reorder dataframe
all_data = all_data[ordered]

# write to csv
os.makedirs('all-host-props-nopore', exist_ok=True)
all_data.to_csv('all-host-props-nopore/all_host_props.csv', index=False)
print('Wrote: all-host-props-nopore/all_host_props.csv')

