#!/bin/bash

module load gromacs

# Input files pattern
input_pattern="final_nvt0_*.trr"

# Loop through matching files
for input_trr in $input_pattern; do
    # Extract the residue number from the input file name
    residue_name=$(echo $input_trr | sed 's/final_nvt0_\(.*\)\.trr/\1/')

    # Output TRR file corresponding to the current input file
    output_trr="host_${residue_name}.trr"

    # Output GRO file corresponding to the current input file
    output_gro="host_${residue_name}.gro"

    # Input TRR file corresponding to the current input file
    input_trr="final_nvt0_${residue_name}.trr"

    # Input GRO file corresponding to the current input file
    input_gro="final_nvt0_${residue_name}.gro"
    
    # Create index file
    echo -e "del 0-30\nr 1\nq" | gmx make_ndx -f $input_gro -o index_${residue_name}

    # Delete specific residue from the trajectory
    gmx trjconv -f $input_gro -s $input_gro -o $output_gro -pbc nojump -n index_${residue_name}.ndx
    gmx trjconv -f $input_trr -s $input_gro -o $output_trr -pbc nojump -n index_${residue_name}.ndx
done

rm *#*#*
