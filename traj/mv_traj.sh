#!/bin/bash

# Specify the base directory
base_directory="Data-MF-05/ML/traj"

cd /afs/crc/group/whitmer/

# Loop from 1 to 4
for i in {6..8}; do
    # Use find to locate all umbrella.trr files
    while IFS= read -r -d '' umbrella_trr; do
        # Extract the real name of the folder
        real_name=$(dirname "$umbrella_trr" | awk -F'/' '{print $(NF-3)}')
        
	# Create specified directories
        mkdir ${real_name}

        # Construct the destination path
        destination_path="${base_directory}/${real_name}/final_nvt0_${real_name}.trr"
         
        # Copy the file to the current directory
        cp "$umbrella_trr" "$destination_path"

        echo "Copied $umbrella_trr to $destination_path"
    done < <(find "Data-MF-0${i}" -type f -path "*/amber/*/Umbrella_sampling/umbrella/run_long_1/final_nvt0.trr" -print0)
done

