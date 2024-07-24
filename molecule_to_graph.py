from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from auglichem.molecule import Compose, RandomAtomMask, RandomBondDelete, MotifRemoval

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
    permitted_atoms = ['Cl', 'C', 'O', 'Br', 'N', 'F', 'S', 'I', 'Si','P',"Unknown"]

    atom_type_encoding = one_hot_encoding(atom.GetSymbol(),permitted_atoms)
    n_heavy_neighbors_encoding = one_hot_encoding(atom.GetDegree(),[0,1,2,3,4,"More than 4"])
    formal_charge_encoding = one_hot_encoding(atom.GetFormalCharge(),[-3,-2,-1,0,1,2,3,"Other"])
    hybridisation_type_encoding = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_encoding = [int(atom.IsInRing())]
    
    is_aromatic_encoding = [int(atom.GetIsAromatic())]
    
    atomic_mass = [float((atom.GetMass()))]
    
    vdw_radius = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())))]

    permitted_valence = [1,2,3,4,"More than 4"]
    valence = one_hot_encoding(atom.GetTotalValence(),permitted_valence)
    
    covalent_radius = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())))]

    permitted_Hs = [0,1,2,3,4,"More than 4"]
    number_Hs = one_hot_encoding(atom.GetTotalNumHs(includeNeighbors=True),permitted_Hs)


    atom_feature_vector = atom_type_encoding + n_heavy_neighbors_encoding + formal_charge_encoding \
                        + hybridisation_type_encoding + is_in_a_ring_encoding + is_aromatic_encoding \
                        + atomic_mass + vdw_radius + covalent_radius + valence + number_Hs

    return np.array(atom_feature_vector)

def get_bond_features(bond):
    permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    return np.array(bond_feature_vector)

def create_graph_molecule(smiles, target=None):
    # SMILES a mol
    canon_smiles = Chem.CanonSmiles(smiles)
    mol = Chem.MolFromSmiles(canon_smiles)

    graph = mol_to_graph(mol, target)
    graph.smiles = smiles

    return graph

def mol_to_graph(mol, target=None):
    # Dimensiones
    n_nodes = mol.GetNumAtoms()
    n_edges = 2*mol.GetNumBonds()

    #Features para cada atomo
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atomic_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
    
    # Matriz de features: (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))
    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atomic_features(atom)
        
    X = torch.tensor(X, dtype = torch.float)
    
    # Indices para ejes del grafo (2, n_edges)
    rows = []
    cols = []
    for bond in mol.GetBonds():
        inicio, final = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [inicio, final]
        cols += [final, inicio]

    torch_rows = torch.tensor(rows,dtype=torch.long)
    torch_cols = torch.tensor(cols,dtype=torch.long)

    E = torch.stack([torch_rows, torch_cols], dim = 0)
    
    # Features de los ejes (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))
    
    for (k, (i,j)) in enumerate(zip(rows, cols)):
        
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
    
    EF = torch.tensor(EF, dtype = torch.float)
    
    
    graph = Data(X, edge_index=E, edge_attr=EF) 

    if target != None: 
        y_tensor = torch.tensor(np.array([target]), dtype = torch.float)
        graph.y = y_tensor
        
    return graph



def create_graph_list(smiles_list,target_list):
    data = []
    for smiles,target in zip(smiles_list,target_list):
        data.append(create_graph_molecule(smiles,target))
    return data

def augment_data(graph_molecule):
    atom_masker = RandomAtomMask(0.5)
    augmented = atom_masker(graph_molecule)
    print(augmented)



if __name__ == "__main__":
    smiles = "Brc1ccccc1Br"
    graph = create_graph_molecule(smiles)
    for i in range(10):
        augment_data(graph)

