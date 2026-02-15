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

    # # visualize the N neighbrs
    # AllChem.EmbedMolecule(mol)
    # mol.GetConformer()
    # # get the 3D coordinates
    # coords = mol.GetConformer().GetPositions()
    # # get atom symbols
    # symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    # # get the coordinates of nitrogen atoms and 1-2 an 1-3 nitrogen_neighbors
    # nitrogen_coords = [coords[int(atom)] for atom in nitrogen_idxs]
    # nitrogen1_coords = [coords[int(atom)] for atom in nitrogen_neighbors_idx]
    # nitrogen2_coords = [coords[int(atom)] for atom in possible_oxygen_idxs_unique]
    # # save the coordinates to xyz files
    # with open('nitrogen_coords.xyz', 'w') as f:
    #     f.write(f'{len(nitrogen_coords)}\n\n')
    #     for c, coord in enumerate(nitrogen_coords):
    #         f.write(f'{symbols[nitrogen_idxs[c]]} {coord[0]} {coord[1]} {coord[2]}\n')
    # with open('nitrogen1_coords.xyz', 'w') as f:
    #     f.write(f'{len(nitrogen1_coords)}\n\n')
    #     for c, coord in enumerate(nitrogen1_coords):
    #         f.write(f'{symbols[nitrogen_neighbors_idx[c]]} {coord[0]} {coord[1]} {coord[2]}\n')
    # with open('nitrogen2_coords.xyz', 'w') as f:
    #     f.write(f'{len(nitrogen2_coords)}\n\n')
    #     for c, coord in enumerate(nitrogen2_coords):
    #         f.write(f'{symbols[possible_oxygen_idxs_unique[c]]} {coord[0]} {coord[1]} {coord[2]}\n')
    # with open('mol_coords.xyz', 'w') as f:
    #     f.write(f'{len(coords)}\n\n')
    #     for c, coord in enumerate(coords):
    #         f.write(f'{symbols[c]} {coord[0]} {coord[1]} {coord[2]}\n')

    num_oxygens = sum([(1 if mol.GetAtomWithIdx(int(atom)).GetAtomicNum() == 8 else 0) for atom in possible_oxygen_idxs_unique])
    num_sulfurs = sum([(1 if mol.GetAtomWithIdx(int(atom)).GetAtomicNum() == 16 else 0) for atom in unique_cycle_atoms])

    return num_all_atoms, num_nitrogens, num_oxygens, num_sulfurs

# read the mol2 file
mol2_folder = 'host-mol2s'
# host_name = f'mc_davis'
# host_path = f'{mol2_folder}/fixed_host_{host_name}.mol2'
host_path = "/users/mfarshad/afs/Private/ML-new/traj/all-host-mol2s/fixed_host_16.mol2"
host_name = host_path.split('/')[-1].split('.')[0].split('_')[-1]
mol = Chem.MolFromMol2File(host_path, removeHs=False)

# 1. Number of atoms
num_atoms = mol.GetNumAtoms()
# 2. Number of aromatic rings
num_benzene_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
num_pr_aromatic_rings = detect_pr_rings(mol)
# 3. Number of hydrogen bond donors
num_hbd = Descriptors.NumHDonors(mol)
# 4. Number of hydrogen bond acceptors
num_hba = Descriptors.NumHAcceptors(mol)
# 5. Number of nitrogen atoms
num_nitrogens = Descriptors.NHOHCount(mol)
# 6. Composition of large cycles
num_all_cycle_atoms, num_cycle_nitrogens, num_cycle_oxygens, num_cycle_sulfurs = detect_large_cycles(mol)

# Save the properties in a dataframe
host_props = pd.DataFrame({
    'host_name': host_name,
    'num_atoms': num_atoms,
    'num_bz_aromatic_rings': num_benzene_aromatic_rings,
    'num_pr_aromatic_rings': num_pr_aromatic_rings,
    'num_hbd': num_hbd,
    'num_hba': num_hba,
    'num_nitrogens': num_nitrogens,
    'num_all_cycle_atoms': num_all_cycle_atoms,
    'num_cycle_nitrogens': num_cycle_nitrogens,
    'num_cycle_oxygens': num_cycle_oxygens,
    'num_cycle_sulfurs': num_cycle_sulfurs
}, index=[0])

# Save the dataframe to a csv file
prop_folder = 'all-host-props'
host_props.to_csv(f'{prop_folder}/host_{host_name}_props.csv', index=False)
