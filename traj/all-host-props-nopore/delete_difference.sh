#!/usr/bin/env bash
set -euo pipefail

# Keep ONLY rows from all_host_props.csv whose 2nd column (paper_name)
# is exactly H1..H36, and save to all_host_props_filtered.csv

props="all_host_props.csv"
out="all_host_props_filtered.csv"
tmp="${out}.tmp"

awk -F',' '
  NR==1 {print; next}
  $2 ~ /^H([1-9]|[12][0-9]|3[0-6])$/ {print}
' "$props" > "$tmp" && mv "$tmp" "$out"

echo "Wrote: $out"

