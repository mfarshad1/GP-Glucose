'''
Find properties of a certain host molecule from its pdb file

## Properties to find
# 1. Number of atoms
# 2. Number of aromatic rings
# 2. Number of cycles (all rings)
# 3. Number of hydrogen bond donors
# 4. Number of hydrogen bond acceptors
# 5. Number of nitrogen atoms

'''

# import libraries
import sys
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

# read the pdb file
pdb_folder = 'host-pdbs'
# abs_num = sys.argv[1]
# host_name = f'asb{abs_num}'
# host_name = f'asb1'
# host_name = sys.argv[1]
# host_path = f'{pdb_folder}/host_{host_name}.pdb'
# host_path = f'{pdb_folder}/hub.mol2'
# mol = Chem.MolFromPDBFile(host_path)
mol = Chem.MolFromMol2File('host_test_mod.mol2')

# 1. Number of atoms
num_atoms = mol.GetNumAtoms()
# 2. Number of aromatic rings
num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
# 3. Number of hydrogen bond donors
num_hbd = Descriptors.NumHDonors(mol)
# 4. Number of hydrogen bond acceptors
num_hba = Descriptors.NumHAcceptors(mol)
# 5. Number of nitrogen atoms
num_nitrogens = Descriptors.NHOHCount(mol)


# Save the properties in a dataframe
host_props = pd.DataFrame({
    'host_name': host_name,
    'num_atoms': num_atoms,
    'num_aromatic_rings': num_aromatic_rings,
    'num_hbd': num_hbd,
    'num_hba': num_hba,
    'num_nitrogens': num_nitrogens
}, index=[0])

# Save the dataframe to a csv file
prop_folder = 'host-props'
host_props.to_csv(f'{prop_folder}/host_{host_name}_props.csv', index=False)