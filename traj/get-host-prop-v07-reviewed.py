"""
Extract molecular features/properties of host molecules.
Modifications from v06:
    - 2 main cycle definitions are used. One that only looks at the largest loop 
        (excludes 1 side of each benzene ring), and one that looks at any connected
        rings as part of the main cycle (i.e. both sides of an aromatic ring and all
        atoms within a rigid aromatic structure that the largest loop passes through
        are considered part of the main cycle). 
    - The 1st main cycle definition is used for indicating the size of the pocket
    - The 2nd main cycle definition is used for finding possible binding sites in 
        the main cycle (specifically, N, O, and S atoms that are 1-2 or 1-3 
        neighbors to a main cycle atom)
"""

import sys , copy
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import os
from numba import jit, njit
from joblib import Parallel, delayed
from itertools import chain
from rdkit.Chem import Draw, rdDepictor # for debugging

# ================== Host name mapping (paper_name <-> real_name) ==================
# paper_name is H1..H36 used in the paper
# real_name is the comment in the LaTeX table WITHOUT the leading "H" (e.g., 107, 133, 83charge, roelen1water, ...)
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

# ================== Numba-accelerated functions ==================
@jit(nopython=True)
def _count_atoms_fast(atomic_nums, target_num):
    count = 0
    for num in atomic_nums:
        if num == target_num:
            count += 1
    return count

@jit(nopython=True)
def _compute_centroid(coords):
    n_atoms, n_dims = coords.shape
    centroid = np.zeros(n_dims)
    for i in range(n_atoms):
        for j in range(n_dims):
            centroid[j] += coords[i, j]
    for j in range(n_dims):
        centroid[j] /= n_atoms
    return centroid    

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
        rmsd = np.sqrt(np.mean(np.sum((coords - flattened)**2, axis=1)))
        planarity_scores.append(rmsd)
    return np.mean(planarity_scores) if planarity_scores else 0.0

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
    lone_pair_count = 0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        ve = valence_electrons.get(z, 0)
        bonds = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])
        formal_charge = atom.GetFormalCharge()
        lp = (ve - bonds - formal_charge) / 2.0
        if lp > 0:
            lone_pair_count += lp
    return round(lone_pair_count, 2)

# New function to count N-C=O groups
def count_nco_groups(mol):
    pattern = Chem.MolFromSmarts('[NX3][CX3]=[OX1]')
    return len(mol.GetSubstructMatches(pattern))

def get_tail_ends(mol, n3_idx, n2_idx):
    '''find remaining atoms in an identified tail by looking for 
    '''
    tail_ends = []
    visited = {n2_idx, n3_idx}  # Track visited atoms to avoid cycles
    
    # get neighbors of atom 3, excluding atom 2
    atom_3_neighbors = mol.GetAtomWithIdx(n3_idx).GetNeighbors()
    atom_3_neighbors = [atom.GetIdx() for atom in atom_3_neighbors if atom.GetIdx() != n2_idx]
    
    # add to list of neighbors
    tail_ends.extend(atom_3_neighbors)
    visited.update(atom_3_neighbors)
    
    # get remainder of tail recursively
    current_neighs = list(atom_3_neighbors)
    while len(current_neighs) > 0:
        next_neighs = []
        for atom_idx in current_neighs:
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            for neighbor in neighbors:
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    next_neighs.append(neighbor_idx)
                    tail_ends.append(neighbor_idx)
        current_neighs = next_neighs
    
    return tail_ends

def get_tail_atoms(mol, ref_atom, large_cycle_atoms_idx, beyond_13=False):
    '''find 1-2 and 1-3 neighbors of large cycle atoms and find 
        1-3 neighbors to them that are not already part of the ring
        (i.e. possible tails)
    Args:
        mol (rdkit.Chem.rdchem.Mol): molecule in question
        ref_atom (rdkit.Chem.rdchem.Atom): reference atom from large cycle
        large_cycle_atoms_idx (list, int): atoms considered part of the main cycle
        beyond_13 (bool): whether or not to count atoms beyond 1-3 neighbors of
            the reference atom
    '''
    # init list of tail atoms    
    tail_atoms = []
    # get 1-2 neighbors of reference atom
    neigh_12 = ref_atom.GetNeighbors()
    neigh_12 = [atom.GetIdx() for atom in neigh_12]
    # remove 1-2 neighbors that are part of the largest cycle/s
    neigh_12 = list(set(neigh_12) - set(large_cycle_atoms_idx))
    if len(neigh_12) == 0:
        return []
    # get 1-3 neighbors
    for n2_idx in neigh_12:
        tail_atoms.append(n2_idx)
        neigh_12_atom = mol.GetAtomWithIdx(n2_idx)
        # get 1-3 neighbors
        neigh_13 = neigh_12_atom.GetNeighbors()
        neigh_13 = [atom.GetIdx() for atom in neigh_13]
        # remove any atoms that are part of the largest cycle/s
        neigh_13 = list(set(neigh_13) - set(large_cycle_atoms_idx))
        # loop over 1-3 neighbors
        for n3_idx in neigh_13:
            tail_atoms.append(n3_idx)
            # add remainder of tail connected to atom 3 (if any, and/or desired)
            if beyond_13: tail_atoms.extend(get_tail_ends(mol, n3_idx, n2_idx))
    
    return set(tail_atoms)  

def draw_mol_with_highlights(mol, highlights=None, outpath='0test.png', size=(800, 800), highlight_color=(1.0, 0.2, 0.2)):
    """
    Draw a molecule with atom indices as map numbers and optional highlighted atoms. Helpful for debugging.
    """
    mol = Chem.Mol(mol)  # copy so you don't modify the original
    
    # Force generation of a flat 2D layout
    rdDepictor.Compute2DCoords(mol, clearConfs=True)

    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", str(atom.GetIdx()))

    dopts = Draw.MolDrawOptions()
    dopts.baseFontSize = 0.6

    if highlights is None:
        highlight_atom_colors = None
    else:
        # normalize indices to ints and filter invalid ones
        highlights = [int(i) for i in highlights if isinstance(i, (int, str)) or hasattr(i, '__int__')]
        highlights = [i for i in highlights if 0 <= i < mol.GetNumAtoms()]

        # build RDKit highlight color dict: atom_idx -> (r,g,b) with 0..1 floats
        highlight_atom_colors = {idx: highlight_color for idx in highlights}

    img = Draw.MolToImage(
        mol,
        size=size,
        options=dopts,
        highlightAtoms=highlights,
        highlightAtomColors=highlight_atom_colors,
    )

    img.save(outpath)
    return img
    
def find_main_ring_inclusive(mol, host_name):
    # find all rings in molecule
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if not atom_rings:
        return ...

    # find largest rings and combine them
    largest_size = max(len(r) for r in atom_rings)
    main_cycles = [r for r in atom_rings if len(r) == largest_size]
    main_cycle_atoms = set().union(*main_cycles)
    non_inclusive_main_cycle = copy.copy(main_cycle_atoms)
    img = draw_mol_with_highlights(mol, highlights = main_cycle_atoms, outpath=f'host-{host_name}/main_cycle_atoms.png')

    # find atoms that are part of rings where some atoms of the rings
    # are counted but some are not (e.g. the benzene ring clusters in host_41)
    for _ in range(2): 
        # loop is performed multiple times to account for aromatic clusters.
        # In case a ring 'A' touches a ring 'B' that is part of the cluster 
        # before all ring 'B' atoms are added to the main_cycle_atoms
        for ring in atom_rings:
            if set(ring) in main_cycles: continue
            for atom_idx in ring:
                if atom_idx in main_cycle_atoms: continue
                atom = mol.GetAtomWithIdx(atom_idx)
                atom_neighbors = [neigh_atom.GetIdx() for neigh_atom in atom.GetNeighbors()] # list of indices
                neighbors_in_main_cycle = set(atom_neighbors).intersection(main_cycle_atoms)
                if len(neighbors_in_main_cycle) > 0: main_cycle_atoms.add(atom_idx)
    
    # draw molecule with main cycle highlighted
    img = draw_mol_with_highlights(mol, highlights = main_cycle_atoms, outpath=f'host-{host_name}/main_cycle_atoms_inclusive.png')

    return main_cycle_atoms, non_inclusive_main_cycle

def get_cycle_props(mol, host_name):
    # define electronegative atoms to find
    target_symbols = ['O', 'N', 'S']

    # get "main cycle" atoms by the inclusive definition
    main_cycle_atoms, non_inclusive_main_cycle = find_main_ring_inclusive(mol, host_name)

    # initialize set to store all ring 1-3 neighbors
    cycle_neighbors_13 = set()

    # get max size of set
    n_atoms_total = len(mol.GetAtoms())

    # loop over main_cycle atoms and collect 1-3 neighbors
    for ref_atom in main_cycle_atoms:
        # get 1-3 neighbors of ref_atom that are not in the main cycle
        ref_neighbors = get_tail_atoms(mol, mol.GetAtomWithIdx(ref_atom), main_cycle_atoms, beyond_13=False)
        # add to set
        cycle_neighbors_13 = cycle_neighbors_13.union(ref_neighbors)
        # check if all atoms in the molecule have been accounted for
        if len(cycle_neighbors_13.union(main_cycle_atoms)) == n_atoms_total: break

    img = draw_mol_with_highlights(mol, highlights = list(cycle_neighbors_13), outpath=f'host-{host_name}/main_cycle_neighbors.png')

    # count desired atom types in list of neighbors
    cycle_neighbors_symbols = [mol.GetAtomWithIdx(neigh_idx).GetSymbol() for neigh_idx in cycle_neighbors_13]
    target_counts = {}
    for target_symbol in target_symbols:
        target_counts[target_symbol] = len(np.where(np.array(cycle_neighbors_symbols) == target_symbol)[0])

    electroneg_nieghs = np.array([neigh_idx if cycle_neighbors_symbols[n] in target_symbols else None for n, neigh_idx in enumerate(cycle_neighbors_13)])
    electroneg_nieghs = electroneg_nieghs[electroneg_nieghs!=None]
    img = draw_mol_with_highlights(mol, highlights=electroneg_nieghs, outpath=f'host-{host_name}/main_cycle_neighbors_electroneg.png')

    # count electornegative ring neighbors
    O_neighbors = len(np.where(np.array(cycle_neighbors_symbols) == 'O')[0])
    N_neighbors = len(np.where(np.array(cycle_neighbors_symbols) == 'N')[0])
    S_neighbors = len(np.where(np.array(cycle_neighbors_symbols) == 'S')[0])

    # Count atom types in ring
    main_cycle_symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in main_cycle_atoms]
    C_main_cycle = len(np.where(np.array(main_cycle_symbols) == 'C')[0])
    O_main_cycle = len(np.where(np.array(main_cycle_symbols) == 'O')[0])
    N_main_cycle = len(np.where(np.array(main_cycle_symbols) == 'N')[0])
    S_main_cycle = len(np.where(np.array(main_cycle_symbols) == 'S')[0])

    return len(non_inclusive_main_cycle), C_main_cycle, O_main_cycle, N_main_cycle, S_main_cycle, O_neighbors, N_neighbors, S_neighbors

# ================== Original functions (optimized) ==================

def count_aromatic_like_rings(mol):
    carbon_aromatic_like = 0
    nitrogen_hetero_aromatic_like = 0
    for ring in mol.GetRingInfo().AtomRings(): # no builtin function/attribute?
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
host_path = sys.argv[1]
# host_path = "all-host-mol2s/fixed_host_41.mol2"
real_name = os.path.splitext(os.path.basename(host_path))[0].split('_')[-1].strip()

# real_name comes from filename tag, normalized so "H107" -> "107"
real_name = _canon_for_mapping(real_name)

# paper_name is derived from real_name if present in the mapping; otherwise fallback
paper_name = REAL_TO_PAPER.get(real_name, real_name)

mol = Chem.MolFromMol2File(host_path, removeHs=False)
Chem.SanitizeMol(mol)

# Count N-C=O groups
nco_count = count_nco_groups(mol)

# Property calculations with Numba where possible
atomic_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
num_atoms = len(atomic_nums)
num_carbons = _count_atoms_fast(atomic_nums, 6)
num_nitrogens = _count_atoms_fast(atomic_nums, 7)
num_oxygens = _count_atoms_fast(atomic_nums, 8)
num_sulfurs = _count_atoms_fast(atomic_nums, 16)

num_hbd = Descriptors.NumHDonors(mol)
num_hba = Descriptors.NumHAcceptors(mol)

# detect_large_cycles() returns:
#       return len(non_inclusive_main_cycle), C_main_cycle, O_main_cycle, N_main_cycle, S_main_cycle, O_neighbors, N_neighbors, S_neighbors
num_cycle_atoms, num_cycle_c, num_cycle_o, num_cycle_n, num_cycle_s, num_o_neigh, num_n_neigh, num_s_neigh  = get_cycle_props(mol, real_name)
num_electroneg = num_o_neigh + num_n_neigh + num_s_neigh

carbon_aromatic, nitrogen_aromatic = count_aromatic_like_rings(mol)
carbon_aromatic, nitrogen_aromatic = count_aromatic_like_within_main_cycles(mol)
planarity_score = compute_ring_planarity(mol)

# New descriptors
electronegativity_total = electronegativity_sum(mol)
lone_pairs_total = estimate_lone_pairs(mol)

# Partial charge calculation
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

# DataFrame creation (all features included)
host_props = pd.DataFrame({
    'real_name': real_name,
    'paper_name': paper_name,
    'EstimatedLonePairs': lone_pairs_total,
    'num_electroneg_1_3_neighbors': num_electroneg,
    'mean_ring_planarity': round(planarity_score, 4),
    'ElectronegativitySum': electronegativity_total,
    'Net partial charge': round(net_partial_charge, 4),
    'NCO_count': nco_count,  # Added N-C=O count
    'num_nitrogens': num_nitrogens,
    'num_oxygens': num_oxygens,
    'num_sulfurs': num_sulfurs,
    'num_hbd': num_hbd,
    'num_hba': num_hba,
    'num_carbon_aromatic_rings': carbon_aromatic,
    'num_nitrogen_aromatic_rings': nitrogen_aromatic,
    'num_all_cycle_atoms': num_cycle_atoms,
    'num_cycle_nitrogens': num_cycle_n,
    'num_cycle_oxygens': num_cycle_o,
    'num_cycle_sulfurs': num_cycle_s,
    'num_cycle_carbons': num_cycle_c,
}, index=[0])

# Save results
prop_folder = 'all-host-props'
os.makedirs(prop_folder, exist_ok=True)
host_props.to_csv(f'{prop_folder}/host_{real_name}_props.csv', index=False)

