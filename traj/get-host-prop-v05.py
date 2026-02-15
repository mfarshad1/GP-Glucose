"""
Find properties of a certain host molecule from its PDB file

## Properties to find
# 1. Number of atoms
# 2. Number of aromatic rings
# 3. Number of cycles (all rings)
# 4. Number of hydrogen bond donors
# 5. Number of hydrogen bond acceptors
# 6. Number of nitrogen atoms
# 7. Number of atoms in large cycles
# 8. Number of electronegative atoms in those cycles
# 9. Planarity score of aromatic atoms
# 10. Number of facing aromatic ring pairs ("temple" structure)

To run for all host molecules:
 for x in host-mol2s/fixed_host_*.mol2; do python get-host-prop-v01.py $x; done
"""

import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import os
from joblib import Parallel, delayed
from numba import njit

@njit
def compute_centroid(coords):
    return np.mean(coords, axis=0)

@njit
def center_coords(coords, centroid):
    return coords - centroid

@njit
def project_onto_plane(coords_centered, normal):
    projected = np.empty_like(coords_centered)
    for i in range(coords_centered.shape[0]):
        v = coords_centered[i]
        proj = v - np.dot(v, normal) * normal
        projected[i] = proj
    return projected

def flatten_ring(coords):
    centroid = compute_centroid(coords)
    coords_centered = center_coords(coords, centroid)
    _, _, vh = np.linalg.svd(coords_centered)
    normal = vh[2]
    projected = project_onto_plane(coords_centered, normal)
    return projected + centroid

def detect_large_cycles(mol):
    ri = mol.GetRingInfo()
    large_cycles = [ring for ring in ri.AtomRings() if len(ring) >= 8]
    all_cycle_atoms = []
    for ring in large_cycles:
        all_cycle_atoms += ring
    unique_cycle_atoms = np.unique(all_cycle_atoms)

    nitrogen_idxs = [int(i) for i in unique_cycle_atoms if mol.GetAtomWithIdx(int(i)).GetAtomicNum() == 7]
    num_nitrogens = len(nitrogen_idxs)

    neighbor_set = set()
    for ni in nitrogen_idxs:
        level1 = list(mol.GetAtomWithIdx(ni).GetNeighbors())
        level2 = [n for a in level1 for n in list(a.GetNeighbors())]
        level3 = [n for a in level2 for n in list(a.GetNeighbors())]
        for a in level1 + level2 + level3:
            neighbor_set.add(a.GetIdx())

    electronegative = [i for i in neighbor_set if mol.GetAtomWithIdx(i).GetAtomicNum() in [7, 8, 9, 16]]
    num_oxygens = sum(1 for i in unique_cycle_atoms if mol.GetAtomWithIdx(int(i)).GetAtomicNum() == 8)
    num_sulfurs = sum(1 for i in unique_cycle_atoms if mol.GetAtomWithIdx(int(i)).GetAtomicNum() == 16)

    return len(unique_cycle_atoms), num_nitrogens, num_oxygens, num_sulfurs, len(electronegative)

# host_path = sys.argv[1]
host_path = sys.argv[1]
host_name = host_path.split('/')[-1].split('.')[0].split('_')[-1]
mol = Chem.MolFromMol2File(host_path, removeHs=False)
Chem.SanitizeMol(mol)

num_atoms = mol.GetNumAtoms()
num_hbd = Descriptors.NumHDonors(mol)
num_hba = Descriptors.NumHAcceptors(mol)
num_nitrogens = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
num_cycle_atoms, num_cycle_n, num_cycle_o, num_cycle_s, num_electroneg = detect_large_cycles(mol)

carbon_aromatic_like = 0
nitrogen_hetero_aromatic_like = 0

for ring in mol.GetRingInfo().AtomRings():
    if len(ring) >= 8:
        continue
    atom_types = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring]
    has_nitrogen = any(num == 7 for num in atom_types)
    all_carbon = all(num == 6 for num in atom_types)

    if all_carbon:
        max_h = max(
            sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 1)
            for i in ring
        )
        if max_h <= 1:
            carbon_aromatic_like += 1
    elif has_nitrogen:
        nitrogen_hetero_aromatic_like += 1

net_partial_charge = 0.0
inside_atom_block = False

with open(host_path, 'r') as f:
    for line in f:
        if line.startswith("@<TRIPOS>ATOM"):
            inside_atom_block = True
            continue
        if line.startswith("@<TRIPOS>"):
            inside_atom_block = False
        if inside_atom_block and line.strip():
            fields = line.strip().split()
            try:
                charge = float(fields[-1])
                net_partial_charge += charge
            except ValueError:
                continue

net_partial_charge = round(net_partial_charge, 4)

host_props = pd.DataFrame({
    'host_name': host_name,
    'num_atoms': num_atoms,
    'num_carbon_aromatic_rings': carbon_aromatic_like,
    'num_nitrogen_aromatic_rings': nitrogen_hetero_aromatic_like,
    'num_hbd': num_hbd,
    'num_hba': num_hba,
    'num_nitrogens': num_nitrogens,
    'num_all_cycle_atoms': num_cycle_atoms,
    'num_cycle_nitrogens': num_cycle_n,
    'num_cycle_oxygens': num_cycle_o,
    'num_cycle_sulfurs': num_cycle_s,
    'num_electroneg_1_3_neighbors': num_electroneg,
    'Net partial charge': net_partial_charge
}, index=[0])

# Skip conformer generation and optimization for speed
# Uncomment and reduce numConfs if needed later
"""
mol = Chem.MolFromMol2File(host_path, removeHs=False)
mol = Chem.RenumberAtoms(mol, list(range(mol.GetNumAtoms())))

params = AllChem.ETKDGv3()
params.randomSeed = 42
params.numThreads = 8
params.pruneRmsThresh = 0.25

conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=10000, params=params)
if not conf_ids:
    raise ValueError(f"❌ No conformers generated for {host_name}")

for cid in conf_ids:
    conf = mol.GetConformer(cid)
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) >= 8:
            continue
        atom_types = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in ring]
        has_n = any(x == 7 for x in atom_types)
        all_c = all(x == 6 for x in atom_types)
        if all_c:
            max_h = max(
                sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetAtomicNum() == 1)
                for i in ring
            )
            if max_h > 1:
                continue
        if all_c and max_h <= 1 or has_n:
            coords = np.array([conf.GetAtomPosition(i) for i in ring])
            flattened_coords = flatten_ring(coords)
            for idx, i in enumerate(ring):
                conf.SetAtomPosition(i, list(flattened_coords[idx]))

def optimize_conf(cid):
    try:
        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return None
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        ff.Minimize(maxIts=1000)
        e = ff.CalcEnergy()
        return (cid, e) if abs(e) < 1e5 else None
    except:
        return None

results = Parallel(n_jobs=8)(delayed(optimize_conf)(cid) for cid in conf_ids)
energies = [r for r in results if r]

if not energies:
    raise ValueError(f"❌ MMFF optimization failed for all conformers of {host_name}")

best_conf_id = min(energies, key=lambda x: x[1])[0]
best_mol = Chem.Mol(mol)
best_mol.RemoveAllConformers()
best_mol.AddConformer(mol.GetConformer(best_conf_id), assignId=True)

output_path = f'host-{host_name}/best_ETKDG_MMFF_flattened.pdb'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
Chem.MolToPDBFile(best_mol, output_path)
print(f"✅ Saved best conformer for {host_name} to {output_path}")
"""

prop_folder = 'all-host-props'
os.makedirs(prop_folder, exist_ok=True)
host_props.to_csv(f'{prop_folder}/host_{host_name}_props.csv', index=False)
