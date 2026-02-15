#!/bin/bash

# Specify the base directory
base_directory="/afs/crc.nd.edu/user/m/mfarshad/Private/ML/traj"

cd /afs/crc/group/whitmer/

# Loop from 1 to 4
for i in {1..4}; do
    # Use find to locate all hub.pdb files
    while IFS= read -r -d '' hub_pdb; do
        # Extract the real name of the folder
        real_name=$(dirname "$hub_pdb" | awk -F'/' '{print $(NF-1)}')

        # Construct the destination path
        destination_path="${base_directory}/host_${real_name}.pdb"

        # Copy the file to the current directory
        cp "$hub_pdb" "$destination_path"

        echo "Copied $hub_pdb to $destination_path"
    done < <(find "Data-MF-0${i}" -type f -path "*/amber/*/cyc.pdb" -print0)
done

