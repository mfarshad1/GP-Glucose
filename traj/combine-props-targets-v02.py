import pandas as pd
import numpy as np

# Read target data
target = pd.read_csv('targets/output.txt', delimiter=' ', header=0, index_col=0)

# Select column to extract (column 2)
select_dG = 2
dG = target.iloc[:, select_dG]

# Convert index to string to ensure consistency
dG.index = dG.index.astype(str)

# Initialize empty dataframe with same index as target
all_data = pd.DataFrame(index=dG.index)

# Initialize columns for properties and dG target
initialized_columns = False

# Get host names from index column
dg_host_names = dG.index

for h, host_name in enumerate(dg_host_names):
    try:
        # Read property data, ensure indices are strings
        all_prop = pd.read_csv(f'all-host-props-nopore/host_{host_name}_all_props.csv', delimiter=',', header=0, index_col=0)
        all_prop.index = all_prop.index.astype(str)

        if not initialized_columns:
            cols = all_prop.columns
            for col in cols:
                all_data[col] = np.nan
            all_data['dG'] = np.nan
            initialized_columns = True

        if host_name not in all_prop.index:
            print(f'Host {host_name} not found in property data. Skipping.')
            continue

        # Assign property values to all_data
        for col in cols:
            all_data.at[host_name, col] = all_prop.at[host_name, col]

        # Add dG target value
        all_data.at[host_name, 'dG'] = dG[host_name]

    except FileNotFoundError:
        print(f'Property file not found. Skipping host: {host_name}')

# Remove rows with NaN values (if any)
all_data = all_data.dropna()

# Write to CSV
all_data.to_csv('all-host-props-nopore/all_host_props_training.csv', index=True)
