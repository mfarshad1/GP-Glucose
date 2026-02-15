#!/bin/bash
#$ -q long,hpc
#$ -pe smp 12
#$ -N Host-Features
#$ -t 1-32

conda activate rdkit-env #runs on the same rdkit versioni mentioned in the paper (2025.03.2)

real_names=(107 133 171 172 174 176 183 185 189 3 41 48 87 97 83charge 83 95 12 13 19 20 22 32 33 40 45 46 47 4 52 56)

i="${SGE_TASK_ID-1}"
x="${real_names[$i]}"

echo -e "Processing host-$x"

echo -e "\n\nExtracting rdkit features"
python get-host-prop-v07-reviewed.py all-host-mol2s/fixed_host_$x.mol2 

echo -e "\n\nCombining pore size and rkdit features"
python combine-all-props-v03.py host-$x/diameter_profile_host-$x.csv

echo -e "\n\nCombining host features with target values for all hosts (should fail for all tasks, except the last)"
python combine-props-targets-v07.py
