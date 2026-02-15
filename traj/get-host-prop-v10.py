#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from numba import jit, njit

# ================== Host name mapping (paper_name <-> real_name) ==================
# paper_name is H1..H36 used in the paper
# real_name is the comment in the LaTeX table WITHOUT the leading "H"
PAPER_TO_REAL = {
    "H1":  "3",
    "H2":  "4",
    "H3":  "12",
    "H4":  "13",
    "H5":  "19",
    "H6":  "20",
    "H7":  "22",
    "H8":  "32",
    "H9":  "33",
    "H10": "40",
    "H11": "41",
    "H12": "45",
    "H13": "46",
    "H14": "47",
    "H15": "48",
    "H16": "52",
    "H17": "56",
    "H18": "83",
    "H19": "83charge",
    "H20": "87",
    "H21": "95",
    "H22": "97",
    "H23": "107",
    "H24": "133",
    "H25": "171",
    "H26": "172",
    "H27": "174",
    "H28": "176",
    "H29": "183",
    "H30": "185",
    "H31": "189",
    "H32": "roelen1water",
    "H33": "davis2water",
    "H34": "stoddart",
    "H35": "davis3water",
    "H36": "hub4",
}
REAL_TO_PAPER = {real: paper for paper, real in PAPER_TO_REAL.items()}

def _canon_for_mapping(tag: str) -> str:
    """
    Normalize the filename tag for mapping.
    - '107'  -> '107'
    - 'H107' -> '107'
    - 'h107' -> '107'
    - '83charge' stays '83charge'
    """
    t = str(tag).strip()
    if len(t) >= 2 and (t[0] in ("h", "H")) and t[1:].isdigit():
        return t[1:]  # drop leading H
    return t

# ================== Numba-accelerated helpers ==================
@jit(nopython=True)
def _count_atoms_fast(atomic_nums, target_num):
    count = 0
    for num in atomic_nums:
        if num == target_num:
            count += 1
    return count

@njit
def compute_centroid(coords):
    n, d = coords.shape
    centroid = np.zeros(d)
    for i in range(n):
        for j in range(d):
            centroid[j] += coords[i, j]
    for j in range(d):
        centroid[j] /= n
    return centroid

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

def compute_ring_planarity(mol):
    conf = mol.GetConformer()
    planarity_scores = []
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) < 4:
            continue
        coords = np.array([conf.GetAtomPosition(i) for i in ring], dtype=np.float64)
        flattened = flatten_ring(coords)
        rmsd = np.sqrt(np.mean(np.sum((coords - flattened) ** 2, axis=1)))
        planarity_scores.append(rmsd)
    return float(np.mean(planarity_scores)) if planarity_scores else 0.0

# ====== Additional Feature Functions ======
def electronegativity_sum(mol):
    electronegativity = {
        1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
        15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
    }
    total = 0.0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        total += electronegativity.get(z, 0.0)
    return round(total, 4)

def estimate_lone_pairs(mol):
    valence_electrons = {
        1: 1, 6: 4, 7: 5, 8: 6, 9: 7,
        15: 5, 16: 6, 17: 7, 35: 7, 53: 7
    }
    lone_pair_count = 0.0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        ve = valence_electrons.get(z, 0)
        bonds = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])
        formal_charge = atom.GetFormalCharge()
        lp = (ve - bonds - formal_charge) / 2.0
        if lp > 0:
            lone_pair_count += lp
    return round(float(lone_pair_count), 2)

def count_nco_groups(mol):
    pattern = Chem.MolFromSmarts('[NX3][CX3]=[OX1]')
    return len(mol.GetSubstructMatches(pattern))

# ================== Cycle features (THIS is what you asked to match) ==================
def detect_large_cycles(mol):
    """
    Cycle atoms are defined as the UNION of atoms in any ring with len(ring) >= 8.
    Then count element types among those unique atoms.

    Returns:
      (num_all_cycle_atoms, num_cycle_carbons, num_cycle_nitrogens,
       num_cycle_oxygens, num_cycle_sulfurs, electronegative_neighbor_proxy)
    """
    ri = mol.GetRingInfo()
    large_cycles = [list(ring) for ring in ri.AtomRings() if len(ring) >= 8]

    all_cycle_atoms = np.array(sum(large_cycles, []), dtype=np.uint32) if large_cycles else np.array([], dtype=np.uint32)
    unique_cycle_atoms = np.unique(all_cycle_atoms)

    atomic_nums = np.array([mol.GetAtomWithIdx(int(i)).GetAtomicNum() for i in unique_cycle_atoms], dtype=np.int32)
    num_carbons   = int(_count_atoms_fast(atomic_nums, 6))
    num_nitrogens = int(_count_atoms_fast(atomic_nums, 7))
    num_oxygens   = int(_count_atoms_fast(atomic_nums, 8))
    num_sulfurs   = int(_count_atoms_fast(atomic_nums, 16))

    neighbor_set = set()
    for ni in (int(i) for i in unique_cycle_atoms if mol.GetAtomWithIdx(int(i)).GetAtomicNum() == 7):
        for neighbor in mol.GetAtomWithIdx(ni).GetNeighbors():
            neighbor_set.add(int(neighbor.GetIdx()))
    electronegative = sum(
        1 for i in neighbor_set
        if mol.GetAtomWithIdx(i).GetAtomicNum() in {7, 8, 9, 16}
    )

    return int(len(unique_cycle_atoms)), num_carbons, num_nitrogens, num_oxygens, num_sulfurs, int(electronegative)

# ================== Aromatic-like counts ==================
def count_aromatic_like_rings(mol):
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
    return int(carbon_aromatic_like), int(nitrogen_hetero_aromatic_like)

def count_aromatic_like_within_main_cycles(mol):
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if not atom_rings:
        return 0, 0
    largest_size = max(len(r) for r in atom_rings)
    main_cycles = [r for r in atom_rings if len(r) == largest_size]
    main_atoms = set().union(*main_cycles)

    carbon_aromatic_like = 0
    nitrogen_hetero_aromatic_like = 0
    for ring in atom_rings:
        if not any(i in main_atoms for i in ring):
            continue
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
    return int(carbon_aromatic_like), int(nitrogen_hetero_aromatic_like)

# ================== Main ==================
if len(sys.argv) < 2:
    raise SystemExit("Usage: extract_host_props.py <host.mol2>")

host_path = sys.argv[1]
host_tag_raw = os.path.splitext(os.path.basename(host_path))[0].split('_')[-1].strip()

real_name = _canon_for_mapping(host_tag_raw)
paper_name = REAL_TO_PAPER.get(real_name, real_name)

mol = Chem.MolFromMol2File(host_path, removeHs=False)
if mol is None:
    raise RuntimeError(f"RDKit failed to read mol2: {host_path}")
Chem.SanitizeMol(mol)

# N-C=O groups
nco_count = int(count_nco_groups(mol))

# Atom counts
atomic_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
num_nitrogens = int(_count_atoms_fast(atomic_nums, 7))
num_oxygens   = int(_count_atoms_fast(atomic_nums, 8))
num_sulfurs   = int(_count_atoms_fast(atomic_nums, 16))

# HBD/HBA
num_hbd = int(Descriptors.NumHDonors(mol))
num_hba = int(Descriptors.NumHAcceptors(mol))

# Cycle features EXACTLY as detect_large_cycles()
num_cycle_atoms, num_cycle_c, num_cycle_n, num_cycle_o, num_cycle_s, num_electroneg = detect_large_cycles(mol)

# Aromatic counts (keep the "within main cycles" version, matching your code intent)
carbon_aromatic, nitrogen_aromatic = count_aromatic_like_within_main_cycles(mol)

# Planarity
planarity_score = float(compute_ring_planarity(mol))

# Electronegativity + lone pairs
electronegativity_total = electronegativity_sum(mol)
lone_pairs_total = estimate_lone_pairs(mol)

# Partial charge (from MOL2)
net_partial_charge = 0.0
with open(host_path, 'r') as f:
    inside_atom_block = False
    for line in f:
        if line.startswith("@<TRIPOS>ATOM"):
            inside_atom_block = True
            continue
        if line.startswith("@<TRIPOS>"):
            inside_atom_block = False
        if inside_atom_block and line.strip():
            fields = line.strip().split()
            try:
                net_partial_charge += float(fields[-1])
            except ValueError:
                continue

host_props = pd.DataFrame({
    'real_name': real_name,
    'paper_name': paper_name,
    'EstimatedLonePairs': lone_pairs_total,
    'num_electroneg_1_3_neighbors': int(num_electroneg),
    'mean_ring_planarity': round(planarity_score, 4),
    'ElectronegativitySum': electronegativity_total,
    'Net partial charge': round(net_partial_charge, 4),
    'NCO_count': nco_count,
    'num_nitrogens': num_nitrogens,
    'num_oxygens': num_oxygens,
    'num_sulfurs': num_sulfurs,
    'num_hbd': num_hbd,
    'num_hba': num_hba,
    'num_carbon_aromatic_rings': carbon_aromatic,
    'num_nitrogen_aromatic_rings': nitrogen_aromatic,
    'num_all_cycle_atoms': int(num_cycle_atoms),
    'num_cycle_nitrogens': int(num_cycle_n),
    'num_cycle_oxygens': int(num_cycle_o),
    'num_cycle_sulfurs': int(num_cycle_s),
    'num_cycle_carbons': int(num_cycle_c),
}, index=[0])

prop_folder = 'all-host-props'
os.makedirs(prop_folder, exist_ok=True)
host_props.to_csv(f'{prop_folder}/host_{real_name}_props.csv', index=False)
print(f"[OK] Wrote {prop_folder}/host_{real_name}_props.csv")

