#!/bin/bash
module load vmd
# Set the path to the VMD executable
vmd_executable=vmd

# Loop through each .trr file
for trr_file in host_*.trr; do
    # Extract the unique identifier from the filename
    unique_identifier=$(echo "$trr_file" | sed 's/host_\(.*\)\.trr/\1/')

    # Set the corresponding .gro file
    gro_file="host_${unique_identifier}.gro"

    # Set the output XYZ file
    xyz_file="host_${unique_identifier}.xyz"

    # Run VMD with the Tcl script to convert the trajectory to XYZ
    "$vmd_executable" -dispdev text <<EOF
mol new $gro_file waitfor all
mol addfile $trr_file type trr waitfor all
set outfile [open $xyz_file w]
set num_frames [molinfo top get numframes]
for {set frame 0} {\$frame < \$num_frames} {incr frame} {
    animate goto \$frame
    animate wait
    animate write xyz output
    }
animate write xyz $xyz_file
}
exit
EOF

    echo "Processing completed for $trr_file"
done
rm *#*#*
