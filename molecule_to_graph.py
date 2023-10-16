from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

def one_hot_encoding(value, permitted_values):
    if value not in permitted_values:
        value = permitted_values[-1]
    
    encoding = [int(is_value_equal) for is_value_equal in list(map(lambda v: value == v,permitted_values))]
    return encoding

def get_atom_in_molecule_count(smiles_list):
    atom_count = {}
    for smiles in smiles_list:
        print(smiles)
        mol = Chem.MolFromSmiles(smiles)
        atom_in_mol = {}
        for atom in mol.GetAtoms():
            s = atom.GetSymbol()
            if s not in atom_in_mol:
                atom_in_mol[s] = True
        for atom_symbol in atom_in_mol:
            if atom_symbol in atom_count:
                atom_count[atom_symbol] += 1
            else:
                atom_count[atom_symbol] = 1
    return atom_count

def get_atomic_features(atom):
    permitted_atoms = ['Cl', 'C', 'O', 'Br', 'N', 'F', 'S', 'I', 'H', 'Si', 'Li',"Unknown"]

    atom_type_encoding = one_hot_encoding(atom.GetSymbol(),permitted_atoms)
    n_heavy_neighbors_encoding = one_hot_encoding(atom.GetDegree(),[0,1,2,3,4,"More than 4"])
    formal_charge_encoding = one_hot_encoding(atom.GetFormalCharge(),[-3,-2,-1,0,1,2,3,"Other"])
    hybridisation_type_encoding = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_encoding = [int(atom.IsInRing())]
    
    is_aromatic_encoding = [int(atom.GetIsAromatic())]
    
    atomic_mass = [float((atom.GetMass()))]
    
    vdw_radius = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())))]
    
    covalent_radius = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())))]
    atom_feature_vector = atom_type_encoding + n_heavy_neighbors_encoding + formal_charge_encoding \
                        + hybridisation_type_encoding + is_in_a_ring_encoding + is_aromatic_encoding \
                        + atomic_mass + vdw_radius + covalent_radius

    return np.array(atom_feature_vector)

def get_bond_features(bond):
    permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    return np.array(bond_feature_vector)

def create_graph_molecule(smiles, target):
    # convert SMILES to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()

    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atomic_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
    
    # construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atomic_features(atom)
        
    X = torch.tensor(X, dtype = torch.float)
    
    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim = 0)
    
    # construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
    
    for (k, (i,j)) in enumerate(zip(rows, cols)):
        
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
    
    EF = torch.tensor(EF, dtype = torch.float)
    
    # construct label tensor
    y_tensor = torch.tensor(np.array([target]), dtype = torch.float)
    
    return Data(X, edge_index=E, edge_attr=EF,y=y_tensor)

def create_graph_list(smiles_list,target_list):
    data = []
    for smiles,target in zip(smiles_list,target_list):
        data.append(create_graph_molecule(smiles,target))
    return data