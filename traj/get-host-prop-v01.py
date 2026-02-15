'''
Find properties of a certain host molecule from its pdb file

## Properties to find
# 1. Number of atoms
# 2. Number of aromatic rings
# 2. Number of cycles (all rings)
# 3. Number of hydrogen bond donors
# 4. Number of hydrogen bond acceptors
# 5. Number of nitrogen atoms

To run for all host molecules:
 for x in host-mol2s/fixed_host_*.mol2; do python get-host-prop-v01.py $x; done
'''
# import libraries
import sys
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

def get_charges_from_mol2(mol2_file):
    """Extract atomic charges from MOL2 file and calculate total charge"""
    charges = []
    with open(mol2_file, 'r') as f:
        in_atom_section = False
        for line in f:
            line = line.strip()
            if line.startswith('@<TRIPOS>ATOM'):
                in_atom_section = True
                continue
            elif line.startswith('@<TRIPOS>'):
                in_atom_section = False
                continue
            
            if in_atom_section and line:
                parts = line.split()
                try:
                    # Charge is typically the last column in ATOM section
                    charge = float(parts[-1])
                    charges.append(charge)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Couldn't parse charge from line: {line}")
                    continue
    
    if not charges:
        print("Warning: No charges found in MOL2 file")
        return 0.0
    
    total_charge = sum(charges)
    # Round to nearest integer (molecular charges are usually integer)
    return round(total_charge)

# write a function to detect pr rings
def detect_pr_rings(mol):
    # find the number of rings
    all_rings = Chem.GetSSSR(mol)
    # loop over all_rings
    num_pr_aromatic_rings = 0
    for ring in all_rings:
        # if ring size is 6 and one of the atoms is nitrogen
        if len(ring) == 6 and any([mol.GetAtomWithIdx(atom).GetAtomicNum() == 7 for atom in ring]):
            num_pr_aromatic_rings += 1
    return num_pr_aromatic_rings

# write a function to detect composition of large cycles
def detect_large_cycles(mol):
    # find the number of rings
    all_rings = Chem.GetSSSR(mol)
    # get cycles with size > 6
    large_cycles = [ring if len(ring) > 6 else None for ring in all_rings]
    all_cycle_atoms = []
    for ring in large_cycles:
        if ring is not None:
            # add ring atom idxs to all_cycle_atoms
            all_cycle_atoms += ring
    # remove repeats from all_cycle_atoms
    unique_cycle_atoms = np.unique(all_cycle_atoms)
    # find number of atoms in the cycle
    num_all_atoms = len(unique_cycle_atoms)
    ## find the composition of unique_cycle_atoms
    # find index of nitrogen atoms
    nitrogen_idxs_in_cycle = np.where([mol.GetAtomWithIdx(int(atom)).GetAtomicNum() == 7 for atom in unique_cycle_atoms])[0]
    nitrogen_idxs = unique_cycle_atoms[nitrogen_idxs_in_cycle]
    num_nitrogens = len(nitrogen_idxs)
    ## get indexes of 1-3 neighbors of nitrogen atoms
    nitrogen_neighbors = [list(mol.GetAtomWithIdx(int(atom)).GetNeighbors()) for atom in nitrogen_idxs]
    # unravel the list
    nitrogen_neighbors = [item for sublist in nitrogen_neighbors for item in sublist]
    # convert atom objects to atom idx
    nitrogen_neighbors_idx = [atom.GetIdx() for atom in nitrogen_neighbors]
    ## get the neighbors of nitrogen neighbors - should include oxygen atoms
    nitrogen2_neighbors = [list(mol.GetAtomWithIdx(int(atom)).GetNeighbors()) for atom in nitrogen_neighbors_idx]
    # unravel the list
    nitrogen2_neighbors = [item for sublist in nitrogen2_neighbors for item in sublist]
    # convert atom objects to atom idx
    nitrogen2_neighbors_idx = [atom.GetIdx() for atom in nitrogen2_neighbors]
    # get the unique oxygen atoms in the cycle
    possible_oxygen_idxs = np.copy(nitrogen2_neighbors_idx)
    possible_oxygen_idxs_unique = np.unique(possible_oxygen_idxs)

    num_oxygens = sum([(1 if mol.GetAtomWithIdx(int(atom)).GetAtomicNum() == 8 else 0) for atom in possible_oxygen_idxs_unique])
    num_sulfurs = sum([(1 if mol.GetAtomWithIdx(int(atom)).GetAtomicNum() == 16 else 0) for atom in unique_cycle_atoms])

    return num_all_atoms, num_nitrogens, num_oxygens, num_sulfurs

# read the mol2 file
host_path = "/users/mfarshad/afs/Private/ML-new/traj/all-host-mol2s/fixed_host_16.mol2"
host_name = host_path.split('/')[-1].split('.')[0].split('_')[-1]

# Read molecule and calculate properties
mol = Chem.MolFromMol2File(host_path, removeHs=False)
if mol is None:
    raise ValueError(f"Could not read molecule from {host_path}")

# Calculate all properties
props = {
    'host_name': host_name,
    'num_atoms': mol.GetNumAtoms(),
    'num_bz_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
    'num_pr_aromatic_rings': detect_pr_rings(mol),
    'num_hbd': Descriptors.NumHDonors(mol),
    'num_hba': Descriptors.NumHAcceptors(mol),
    'num_nitrogens': Descriptors.NHOHCount(mol),
    'rdkit_formal_charge': Chem.GetFormalCharge(mol),
}

# Add cycle composition
cycle_props = detect_large_cycles(mol)
props.update({
    'num_all_cycle_atoms': cycle_props[0],
    'num_cycle_nitrogens': cycle_props[1],
    'num_cycle_oxygens': cycle_props[2],
    'num_cycle_sulfurs': cycle_props[3],
})

# Calculate charge from MOL2 file (most reliable)
mol2_charge = get_charges_from_mol2(host_path)
props['mol2_total_charge'] = mol2_charge

# Validate charges
if abs(mol2_charge - props['rdkit_formal_charge']) > 0.5:
    print(f"Warning: Charge discrepancy for {host_name}: MOL2 charge={mol2_charge}, RDKit formal charge={props['rdkit_formal_charge']}")

# Save the properties in a dataframe
host_props = pd.DataFrame(props, index=[0])

# Save the dataframe to a csv file
prop_folder = 'all-host-props'
host_props.to_csv(f'{prop_folder}/host_{host_name}_props.csv', index=False)

print(f"Processed {host_name}: Total charge = {mol2_charge}")
