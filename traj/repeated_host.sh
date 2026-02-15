#!/bin/bash

# Search recursively for all *.mol2 files
find . -type f -name "*.mol2" | while read -r file; do
    if grep -Fxq "DAV" "$file" && grep -Fxq "  304   314     1     0     0" "$file"; then
        dir=$(dirname "$file")
        echo "❌ Deleting folder: $dir (matched $file)"
        rm -rf "$dir"
    fi
done

