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

# Input file
input_file=$1

# Output file
output_file=$2

# Perform substitution and save the result to the output file
substitute_names < "$input_file" > "$output_file"

# Display a message indicating successful substitution
echo "Atom names starting with 'h', 'c', 'o', 's' and 'n' substituted. Output saved to $output_file"

