#!/bin/bash

# Check if file is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

file=$1

# Check if file exists
if [ ! -f "$file" ]; then
    echo "Error: File '$file' not found"
    exit 1
fi

# Use awk to process the file
awk '
{
    # Store the line and its 4th column
    lines[NR] = $0
    cols[NR] = $4
}

END {
    # Create a frequency count of 4th column values
    for (i = 1; i <= NR; i++) {
        count[cols[i]]++
    }
    
    # Find which values appear more than once
    for (val in count) {
        if (count[val] > 1) {
            duplicates[val] = 1
        }
    }
    
    # Print all lines that have duplicate 4th columns
    for (i = 1; i <= NR; i++) {
        if (cols[i] in duplicates) {
            print lines[i]
        }
    }
}
' "$file"
