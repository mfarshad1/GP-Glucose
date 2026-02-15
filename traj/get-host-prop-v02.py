import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdmolops

def detect_pr_rings(mol):
    """Count pyridine-like rings (6-membered with at least 1 nitrogen)"""
    return sum(
        1 for ring in mol.GetRingInfo().AtomRings() 
        if len(ring) == 6 and any(mol.GetAtomWithIdx(a).GetAtomicNum() == 7 for a in ring)
    )

def analyze_nitrogens_oxygens(mol):
    """Classify N/O atoms by cycle proximity and hydrogen attachment"""
    rings = mol.GetRingInfo().AtomRings()
    cycle_atoms = {a for ring in rings for a in ring}
    
    results = {
        # Inside cycles
        'N_in_cycles_with_H': 0, 'N_in_cycles_no_H': 0,
        'O_in_cycles_with_H': 0, 'O_in_cycles_no_H': 0,
        # Near cycles (1-3 bonds)
        'N_near_cycles_with_H': 0, 'N_near_cycles_no_H': 0,
        'O_near_cycles_with_H': 0, 'O_near_cycles_no_H': 0,
        # Far from cycles (>3 bonds)
        'N_far_away_with_H': 0, 'N_far_away_no_H': 0,
        'O_far_away_with_H': 0, 'O_far_away_no_H': 0,
        # Large cycle composition
        'num_all_cycle_atoms': 0, 'num_cycle_nitrogens': 0,
        'num_cycle_oxygens': 0, 'num_cycle_sulfurs': 0
    }
    
    # Large cycle analysis (original functionality)
    large_cycles = [ring for ring in rings if len(ring) > 6]
    if large_cycles:
        large_cycle_atoms = {a for ring in large_cycles for a in ring}
        results.update({
            'num_all_cycle_atoms': len(large_cycle_atoms),
            'num_cycle_nitrogens': sum(mol.GetAtomWithIdx(a).GetAtomicNum() == 7 for a in large_cycle_atoms),
            'num_cycle_oxygens': sum(mol.GetAtomWithIdx(a).GetAtomicNum() == 8 for a in large_cycle_atoms),
            'num_cycle_sulfurs': sum(mol.GetAtomWithIdx(a).GetAtomicNum() == 16 for a in large_cycle_atoms)
        })
    
    # N/O classification
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in {7, 8}:
            continue
            
        idx = atom.GetIdx()
        has_h = atom.GetTotalNumHs() > 0
        element = 'N' if atomic_num == 7 else 'O'
        
        if idx in cycle_atoms:
            category = 'in_cycles'
        else:
            min_dist = min(
                len(rdmolops.GetShortestPath(mol, idx, ca)) - 1 
                for ca in cycle_atoms
            ) if cycle_atoms else float('inf')
            category = 'near_cycles' if min_dist <= 3 else 'far_away'
        
        results[f'{element}_{category}_{"with_H" if has_h else "no_H"}'] += 1
    
    return results

def main(mol2_path):
    mol = Chem.MolFromMol2File(mol2_path, removeHs=False)
    host_name = mol2_path.split('/')[-1].split('.')[0].split('_')[-1]
    
    # Basic properties
    props = {
        'host_name': host_name,
        'num_atoms': mol.GetNumAtoms(),
        'num_bz_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'num_pr_aromatic_rings': detect_pr_rings(mol),
        'num_hbd': Descriptors.NumHDonors(mol),
        'num_hba': Descriptors.NumHAcceptors(mol)
    }
    
    # N/O analysis
    props.update(analyze_nitrogens_oxygens(mol))
    
    # Save results
    pd.DataFrame([props]).to_csv(f'all-host-props/host_{host_name}_props.csv', index=False)

if __name__ == '__main__':
    main(sys.argv[1])
