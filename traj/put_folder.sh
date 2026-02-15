#!/bin/bash

# Iterate over all files matching the pattern host_{specific name}.trr
for file in host_*.trr; do
    # Extract the specific name from the file name
    specific_name=$(echo "$file" | cut -d'_' -f2 | cut -d'.' -f1)
    
    # Create a folder with the specific name if it doesn't exist
    mkdir -p "$specific_name"
    
    # Move the file into the folder
    mv "$file" "$specific_name/"
done
