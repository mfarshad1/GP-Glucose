#!/bin/bash

# Define a function to substitute atom names
substitute_names() {
    # Use sed to perform the substitution
    sed -E 's/\b(h[[:alnum:]]*)\b/H /g;
            s/\b(c[[:alnum:]]*)\b/C /g;
            s/\b(o[[:alnum:]]*)\b/O /g;
            s/\b(s[[:alnum:]]*)\b/S /g;
            s/\b(n[[:alnum:]]*)\b/N /g'
}

# Define the Bash script file you want to copy and execute
bash_script="change_names_mol2.sh"

# Store the current directory
current_dir=$(pwd)

# Find all subfolders
subfolders=$(find host-* -mindepth 0 -type d)

# Iterate through each subfolder
for folder in $subfolders; do
    # Extract the folder name without leading './'
    folder_name=$(basename "$folder")
    host_name=$(basename "$folder" | sed 's/^host-//')

    # Navigate into the subfolder
    cd "$folder" || continue
    rm *pore_diameter.o*
    rm fixed*
    # Copy the Bash script to the subfolder
    cp "$current_dir/$bash_script" .

    # Define input and output filenames based on the folder name
    input_name="host_$host_name.mol2"
    output_name="fixed_host_$host_name.mol2"

    # Perform substitution and save the result to the output file
    substitute_names < "$input_name" > "$output_name"

    # Display a message indicating successful substitution for this folder
    echo "Atom names starting with 'h', 'c', 'o', 's' and 'n' substituted for $input_name. Output saved to $output_name"

    # Return to the original directory
    cd "$current_dir" || exit
done

