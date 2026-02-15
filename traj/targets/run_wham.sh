#!/bin/bash

# Define the base directory pattern
base_dir="/afs/crc/group/whitmer/Data-MF-07/amber/*/Umbrella_sampling/umbrella"

# Find all "umbrella" directories and navigate inside
find $base_dir -type d -exec sh -c '
    # Navigate inside the "umbrella" directory
    cd "$0"
    
    # Check if files_wham.sh exists and execute it
    if [ -f "files_wham.sh" ]; then
	echo "Running files_wham.sh in directory: $PWD"
        bash files_wham.sh
    fi
    # Remove the last line from pullf-files.dat and tpr-files.dat
    sed -i '$d' run_long_1/pullf-files.dat
    sed -i '$d' run_long_1/tpr-files.dat
    # Check if wham.sh exists and execute it
    if [ -f "wham.sh" ]; then
        echo "Running wham.sh in directory: $PWD"
        bash wham.sh
    fi
' {} \;

