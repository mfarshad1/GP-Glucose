#!/bin/bash

# Define the Python script file you want to copy and execute
python_script="pore_diameter.py"
pbs_script="run.pbs"

# Find all subfolders
subfolders=$(find . -mindepth 1 -type d)

# Copy the Python script to each subfolder and execute it
for folder in $subfolders; do 
#    cp $python_script "$folder"
#    cp $pbs_script "$folder"
    cd "$folder"
    # python "$python_script"
    #qsub "$pbs_script"
    rm hole*
    rm *.o*
    rm *.rad*
    rm *.pdf
    rm *.vmd
    rm *.csv
    cd ..
done

