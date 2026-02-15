#!/bin/bash

# Source directory
source_directory="/afs/crc/group/whitmer/Data-MF-05/ML/traj/"

# Destination base directory
destination_base_directory="/afs/crc.nd.edu/user/m/mfarshad/Private/ML-new/traj/"

# Find all host*.mol2 and host*.trr files in the source directory
find "$source_directory" \( -name "host*.mol2" -o -name "host*.trr" -o -name "host*.gro" -o -name "host*.xyz"  \) -print0 |
while IFS= read -r -d '' file; do
    # Extract the filename without extension and prefix
    filename=$(basename "$file" | sed 's/^host_//' | sed 's/\.[^.]*$//')

    # Construct the destination directory
    destination_directory="${destination_base_directory}host-${filename}"
    #destination_directory="${destination_base_directory}${filename}"

    # Create the destination directory if it doesn't exist
    mkdir -p "$destination_directory"

    # Copy the file to the destination directory
    cp "$file" "$destination_directory/"

    echo "Copied $file to $destination_directory/"
done

