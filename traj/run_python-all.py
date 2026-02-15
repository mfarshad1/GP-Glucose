#!/bin/bash

# Define the Python script file you want to copy and execute
python_script="pore_diameter_all.py"

# Store the current directory
current_dir=$(pwd)

# Find all subfolders
subfolders=$(find . -mindepth 1 -type d)

# Iterate through each subfolder
for folder in $subfolders; do
    # Extract the folder name without leading './'
    folder_name=$(basename "$folder")
    
    # Navigate into the subfolder
    cd "$folder" || continue
    
    # Copy the Python script to the subfolder
    cp "$current_dir/$python_script" .
    
    # Execute the Python script with the folder name as argument
    python "$python_script" "$folder_name"
    
    # Return to the original directory
    cd "$current_dir" || exit
done

