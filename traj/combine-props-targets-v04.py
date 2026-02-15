import pandas as pd
import numpy as np

# Read target data
target = pd.read_csv('targets/output.txt', delimiter=' ', header=0, index_col=0)

# Select column to extract (column 2)
select_dG = 2
dG = target.iloc[:, select_dG]

# Ensure that indices in dG are strings for consistent comparison
dG.index = dG.index.astype(str)

# Initialize empty dataframe with same index as target
all_data = pd.DataFrame(index=dG.index)

# Initialize empty columns and target column
all_data['dG'] = np.nan

# Get host names from index column
dg_host_names = dG.index

for host_name in dg_host_names:
    # Read property data
    try:
        all_prop = pd.read_csv(
            f'all-host-props-nopore/host_{host_name}_all_props.csv', 
            delimiter=',', 
            header=0, 
            index_col=0
        )
        
        # Convert indices in all_prop to strings for consistent comparison
        all_prop.index = all_prop.index.astype(str)
        
        # Debug print
        print(f"Processing Host: {host_name}")
        print(f"Available indices in all_prop: {all_prop.index.tolist()}")
        
        # Check if the host_name is actually present
        if host_name not in all_prop.index:
            print(f'Host {host_name} not found in property data. Skipping.')
            continue

        # List all columns in all_prop
        cols = all_prop.columns

        # Initialize columns in all_data based on all_prop columns if not already initialized
        if len(all_data.columns) - 1 < len(cols):  # Excluding the 'dG' column
            for col in cols:
                if col not in all_data.columns:
                    all_data[col] = np.nan

        # Prepare concatenated data for the host
        host_props = all_prop.loc[host_name]
        
        # Ensure the data length matches the columns in all_data
        concatenated_data = pd.Series(host_props.values, index=cols)

        # Add data to all_data
        all_data.loc[host_name, cols] = concatenated_data
        all_data.loc[host_name, 'dG'] = dG[host_name]

    except FileNotFoundError:
        print(f'Property file not found. Skipping host: {host_name}')

# Remove rows with NaN values (if any)
all_data = all_data.dropna()

# Write to CSV
all_data.to_csv('all-host-props-nopore/all_host_props_training.csv', index=True)
