#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract molecular features/properties of host molecules.

Two main-cycle definitions are used:
  (1) NON-inclusive largest loop only (excludes one side of each benzene ring)
      -> used as a "pocket size" proxy (not necessarily written to CSV unless you want)
  (2) INCLUSIVE main cycle: largest loop + connected aromatic cluster atoms
      -> used for ALL cycle element counts:
           num_all_cycle_atoms, num_cycle_carbons, num_cycle_nitrogens,
           num_cycle_oxygens, num_cycle_sulfurs

Also computes 1-2/1-3 neighbor electronegative counts around the inclusive main cycle.

MINIMAL-CHANGE FIX:
  - DO NOT mix cycle definitions.
  - REMOVE override of num_cycle_carbons via detect_large_cycles (ring len>=8 heuristic).
  - All num_cycle_* counts come from the inclusive main cycle (same as before logic).
"""

import sys, copy
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
from numba import jit, njit
from itertools import chain
from rdkit.Chem import Draw, rdDepictor  # for debugging

# ================== Host name mapping (paper_name <-> real_name) ==================
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
    Normalize filename tag for mapping.
    - '107'  -> '107'
    - 'H107' -> '107'
    - 'h107' -> '107'
    - '83charge' stays '83charge'
    """
    t = str(tag).strip()
    if len(t) >= 2 and (t[0] in ("h", "H")) and t[1:].isdigit():
        return t[1:]
    return t


# ================== Numba-accelerated functions ==================
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


def get_tail_ends(mol, n3_idx, n2_idx):
    """Find remaining atoms in a tail starting from atom n3 (excluding n2)."""
    tail_ends = []
    visited = {n2_idx, n3_idx}

    atom_3_neighbors = mol.GetAtomWithIdx(n3_idx).GetNeighbors()
    atom_3_neighbors = [a.GetIdx() for a in atom_3_neighbors if a.GetIdx() != n2_idx]

    tail_ends.extend(atom_3_neighbors)
    visited.update(atom_3_neighbors)

    current_neighs = list(atom_3_neighbors)
    while len(current_neighs) > 0:
        next_neighs = []
        for atom_idx in current_neighs:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    next_neighs.append(neighbor_idx)
                    tail_ends.append(neighbor_idx)
        current_neighs = next_neighs

    return tail_ends


def get_tail_atoms(mol, ref_atom, large_cycle_atoms_idx, beyond_13=False):
    """
    Find 1-2 and 1-3 neighbors of a ref_atom that are NOT in large_cycle_atoms_idx.
    Optionally extend beyond 1-3.
    """
    tail_atoms = []

    neigh_12 = [a.GetIdx() for a in ref_atom.GetNeighbors()]
    neigh_12 = list(set(neigh_12) - set(large_cycle_atoms_idx))
    if len(neigh_12) == 0:
        return set()

    for n2_idx in neigh_12:
        tail_atoms.append(n2_idx)
        neigh_12_atom = mol.GetAtomWithIdx(n2_idx)

        neigh_13 = [a.GetIdx() for a in neigh_12_atom.GetNeighbors()]
        neigh_13 = list(set(neigh_13) - set(large_cycle_atoms_idx))

        for n3_idx in neigh_13:
            tail_atoms.append(n3_idx)
            if beyond_13:
                tail_atoms.extend(get_tail_ends(mol, n3_idx, n2_idx))

    return set(tail_atoms)


def draw_mol_with_highlights(
    mol,
    highlights=None,
    outpath='0test.png',
    size=(800, 800),
    highlight_color=(1.0, 0.2, 0.2)
):
    """
    Draw a molecule with atom indices and optional highlighted atoms (debugging).
    """
    mol = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(mol, clearConfs=True)

    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", str(atom.GetIdx()))

    dopts = Draw.MolDrawOptions()
    dopts.baseFontSize = 0.6

    highlight_atom_colors = None
    if highlights is not None:
        highlights2 = []
        for i in highlights:
            try:
                ii = int(i)
                if 0 <= ii < mol.GetNumAtoms():
                    highlights2.append(ii)
            except Exception:
                pass
        highlight_atom_colors = {idx: highlight_color for idx in highlights2}

    img = Draw.MolToImage(
        mol,
        size=size,
        options=dopts,
        highlightAtoms=highlights if highlights is not None else None,
        highlightAtomColors=highlight_atom_colors,
    )
    img.save(outpath)
    return img


def find_main_ring_inclusive(mol, host_name):
    """
    Inclusive main-cycle definition:
      - start with largest ring(s)
      - expand to include connected ring atoms (aromatic clusters)
    Returns:
      (inclusive_main_cycle_atoms_set, non_inclusive_main_cycle_atoms_set)
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if not atom_rings:
        return set(), set()

    largest_size = max(len(r) for r in atom_rings)
    main_cycles = [r for r in atom_rings if len(r) == largest_size]
    main_cycle_atoms = set().union(*main_cycles)
    non_inclusive_main_cycle = copy.copy(main_cycle_atoms)

    # (optional debug)
    # os.makedirs(f'host-{host_name}', exist_ok=True)
    # draw_mol_with_highlights(mol, highlights=list(main_cycle_atoms),
    #                          outpath=f'host-{host_name}/main_cycle_atoms.png')

    for _ in range(2):
        for ring in atom_rings:
            if ring in main_cycles:
                continue
            for atom_idx in ring:
                if atom_idx in main_cycle_atoms:
                    continue
                atom = mol.GetAtomWithIdx(atom_idx)
                atom_neighbors = [na.GetIdx() for na in atom.GetNeighbors()]
                if len(set(atom_neighbors).intersection(main_cycle_atoms)) > 0:
                    main_cycle_atoms.add(atom_idx)

    # (optional debug)
    # draw_mol_with_highlights(mol, highlights=list(main_cycle_atoms),
    #                          outpath=f'host-{host_name}/main_cycle_atoms_inclusive.png')

    return main_cycle_atoms, non_inclusive_main_cycle


def get_cycle_props(mol, host_name):
    """
    Returns consistent cycle properties.

    non_inclusive_main_cycle_size: size of the largest loop only (pocket proxy)
    num_all_cycle_atoms: number of atoms in inclusive main cycle
    num_cycle_c/o/n/s: element counts within inclusive main cycle
    num_o/n/s_neighbors: counts of O/N/S within 1-2 or 1-3 neighbors of inclusive cycle atoms (tails)
    """
    target_symbols = ['O', 'N', 'S']

    main_cycle_atoms, non_inclusive_main_cycle = find_main_ring_inclusive(mol, host_name)

    if not main_cycle_atoms:
        # No rings: return zeros
        return (0, 0, 0, 0, 0, 0, 0, 0, 0)

    cycle_neighbors_13 = set()
    n_atoms_total = mol.GetNumAtoms()

    for ref_idx in main_cycle_atoms:
        ref_neighbors = get_tail_atoms(mol, mol.GetAtomWithIdx(ref_idx), main_cycle_atoms, beyond_13=False)
        cycle_neighbors_13 |= set(ref_neighbors)
        if len(cycle_neighbors_13 | set(main_cycle_atoms)) == n_atoms_total:
            break

    cycle_neighbors_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in cycle_neighbors_13]
    O_neighbors = cycle_neighbors_symbols.count('O')
    N_neighbors = cycle_neighbors_symbols.count('N')
    S_neighbors = cycle_neighbors_symbols.count('S')

    main_cycle_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in main_cycle_atoms]
    C_main_cycle = main_cycle_symbols.count('C')
    O_main_cycle = main_cycle_symbols.count('O')
    N_main_cycle = main_cycle_symbols.count('N')
    S_main_cycle = main_cycle_symbols.count('S')

    non_inclusive_size = len(non_inclusive_main_cycle)
    inclusive_size = len(main_cycle_atoms)

    return (
        non_inclusive_size,
        inclusive_size,
        C_main_cycle,
        O_main_cycle,
        N_main_cycle,
        S_main_cycle,
        O_neighbors,
        N_neighbors,
        S_neighbors,
    )


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
    return carbon_aromatic_like, nitrogen_hetero_aromatic_like


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
    return carbon_aromatic_like, nitrogen_hetero_aromatic_like


# ================== Main script ==================
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
nco_count = count_nco_groups(mol)

# Global atom counts
atomic_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.int32)
num_nitrogens = int(_count_atoms_fast(atomic_nums, 7))
num_oxygens = int(_count_atoms_fast(atomic_nums, 8))
num_sulfurs = int(_count_atoms_fast(atomic_nums, 16))

# HBD/HBA
num_hbd = int(Descriptors.NumHDonors(mol))
num_hba = int(Descriptors.NumHAcceptors(mol))

# Cycle properties (CONSISTENT, from inclusive main cycle)
(
    pocket_cycle_size,     # largest loop only (proxy)
    num_all_cycle_atoms,   # inclusive main cycle atom count
    num_cycle_c,
    num_cycle_o,
    num_cycle_n,
    num_cycle_s,
    num_o_neigh,
    num_n_neigh,
    num_s_neigh,
) = get_cycle_props(mol, real_name)

num_electroneg = int(num_o_neigh + num_n_neigh + num_s_neigh)

# Aromatic-like counts
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

# Optional sanity check (helps catch future regressions)
# If there are non C/N/O/S atoms in the main cycle (e.g., halogens), this will fail.
# If you *expect* only C/N/O/S in cycle, keep it on; otherwise comment out.
# assert int(num_all_cycle_atoms) == int(num_cycle_c + num_cycle_n + num_cycle_o + num_cycle_s), \
#     "Cycle element counts do not sum to total cycle atoms!"

host_props = pd.DataFrame({
    'real_name': real_name,
    'paper_name': paper_name,
    'EstimatedLonePairs': lone_pairs_total,
    'num_electroneg_1_3_neighbors': num_electroneg,
    'mean_ring_planarity': round(planarity_score, 4),
    'ElectronegativitySum': electronegativity_total,
    'Net partial charge': round(net_partial_charge, 4),
    'NCO_count': int(nco_count),
    'num_nitrogens': num_nitrogens,
    'num_oxygens': num_oxygens,
    'num_sulfurs': num_sulfurs,
    'num_hbd': num_hbd,
    'num_hba': num_hba,
    'num_carbon_aromatic_rings': int(carbon_aromatic),
    'num_nitrogen_aromatic_rings': int(nitrogen_aromatic),

    # main cycle counts (inclusive definition)
    'num_all_cycle_atoms': int(num_all_cycle_atoms),
    'num_cycle_nitrogens': int(num_cycle_n),
    'num_cycle_oxygens': int(num_cycle_o),
    'num_cycle_sulfurs': int(num_cycle_s),
    'num_cycle_carbons': int(num_cycle_c),

    # if you want to store the pocket proxy too, uncomment:
    # 'num_pocket_cycle_atoms': int(pocket_cycle_size),
}, index=[0])

prop_folder = 'all-host-props'
os.makedirs(prop_folder, exist_ok=True)
host_props.to_csv(f'{prop_folder}/host_{real_name}_props.csv', index=False)
print(f"[OK] Wrote {prop_folder}/host_{real_name}_props.csv")

