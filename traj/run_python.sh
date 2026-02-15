#!/bin/bash

# Define the Python script file you want to copy and execute
# python_script="pore_diameter_all.py"
python_script="pore_diameter.py"
pbs_script="run.pbs"

# Store the current directory
current_dir=$(pwd)

# Find all subfolders
subfolders=$(find host-* -mindepth 0 -type d)
#subfolders=host-asb1

# Iterate through each subfolder
for folder in $subfolders; do
    # Extract the folder name without leading './'
    # folder_name=$(basename "$folder")
    
    # Navigate into the subfolder
    cd "$folder" || continue
    rm *pore_diameter.o*
    #rm *csv
    rm *hole*
    #rm *pdf
    # Copy the Python script and pbs to the subfolder
    cp "$current_dir/$python_script" .
    cp "$current_dir/$pbs_script" .

    # Execute the Python script with the folder name as argument
    #python "$python_script" "$folder_name"
    qsub "$pbs_script" $folder

    # Return to the original directory
    cd "$current_dir" || exit
done

