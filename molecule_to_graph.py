from rdkit import Chem
import numpy as np
import torch
from torch_geometric import edge_index
from torch_geometric.data import Data
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from auglichem.molecule import Compose, RandomAtomMask, RandomBondDelete, MotifRemoval
import random

device = "cuda"

class MolToGraph:
    
    def __init__(self):
        unrelated_smiles = "C-C"
        self.unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        self.permitted_atoms = ['Cl', 'C', 'O', 'Br', 'N', 'F', 'S', 'I', 'Si','P',"Unknown"]
        pass

    def create_graph_list(self, smiles_list,target_list, rep=1):
        data = []
        for smiles,target in zip(smiles_list,target_list):
            for _ in range(rep):
                data.append(self.smiles_to_graph(smiles,target))
        return data

    def smiles_to_graph(self, smiles, target=None):
        canon_smiles = Chem.CanonSmiles(smiles)
        mol = Chem.MolFromSmiles(canon_smiles)
        graph = self.mol_to_graph(mol, target)
        graph.smiles = smiles
        return graph.to(device)

    def get_node_features(self,mol):
        n_nodes = mol.GetNumAtoms()
        n_node_features = len(self.get_atomic_features(self.unrelated_mol.GetAtomWithIdx(0)))
        node_features = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            node_features[atom.GetIdx(), :] = self.get_atomic_features(atom)
            
        node_features = torch.tensor(node_features, dtype = torch.float)

        return node_features

    def get_edge_features(self,mol):
        n_edges = 2*mol.GetNumBonds()
        n_edge_features = len(MolToGraph.get_bond_features(self.unrelated_mol.GetBondBetweenAtoms(0,1)))
        rows = []
        cols = []
        for bond in mol.GetBonds():
            inicio, final = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            rows += [inicio, final]
            cols += [final, inicio]

        torch_rows = torch.tensor(rows,dtype=torch.long)
        torch_cols = torch.tensor(cols,dtype=torch.long)

        edge_index = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # Features de los ejes (n_edges, n_edge_features)
        edge_features = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            edge_features[k] = MolToGraph.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        edge_features = torch.tensor(edge_features, dtype = torch.float)

        return edge_index,edge_features

    def mol_to_graph(self, mol, target=None):
        node_features = self.get_node_features(mol)
        edge_index, edge_features = self.get_edge_features(mol)
        
        graph = Data(node_features, edge_index=edge_index, edge_attr=edge_features) 

        if target != None: 
            y_tensor = torch.tensor(np.array([target]), dtype = torch.float)
            graph.y = y_tensor
            
        return graph

    def get_atomic_features(self,atom):

        atom_type_encoding = MolToGraph.one_hot_encoding(atom.GetSymbol(),self.permitted_atoms)
        n_heavy_neighbors_encoding = MolToGraph.one_hot_encoding(atom.GetDegree(),[0,1,2,3,4,"More than 4"])
        formal_charge_encoding = MolToGraph.one_hot_encoding(atom.GetFormalCharge(),[-3,-2,-1,0,1,2,3,"Other"])
        hybridisation_type_encoding = MolToGraph.one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        
        is_in_a_ring_encoding = [int(atom.IsInRing())]
        
        is_aromatic_encoding = [int(atom.GetIsAromatic())]
        
        atomic_mass = [float((atom.GetMass()))]
        
        vdw_radius = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum())))]

        permitted_valence = [1,2,3,4,"More than 4"]
        valence = MolToGraph.one_hot_encoding(atom.GetTotalValence(),permitted_valence)
        
        covalent_radius = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum())))]

        permitted_Hs = [0,1,2,3,4,"More than 4"]
        number_Hs = MolToGraph.one_hot_encoding(atom.GetTotalNumHs(includeNeighbors=True),permitted_Hs)


        atom_feature_vector = atom_type_encoding + n_heavy_neighbors_encoding + formal_charge_encoding \
                            + hybridisation_type_encoding + is_in_a_ring_encoding + is_aromatic_encoding \
                            + atomic_mass + vdw_radius + covalent_radius + valence + number_Hs

        return np.array(atom_feature_vector)

    def set_permitted_atoms(self, permitted_atoms):
        self.permitted_atoms = permitted_atoms

    @staticmethod
    def one_hot_encoding(value, permitted_values):
        if value not in permitted_values:
            value = permitted_values[-1]
        
        encoding = [int(is_value_equal) for is_value_equal in list(map(lambda v: value == v,permitted_values))]
        return encoding

    @staticmethod
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


    @staticmethod
    def get_bond_features(bond):
        permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        
        bond_type_enc = MolToGraph.one_hot_encoding(bond.GetBondType(), permitted_bond_types)
        
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        
        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
        
        return np.array(bond_feature_vector)

class AtomMaskingMolToGraph(MolToGraph):

    def __init__(self, prob, seed=None):
        super().__init__()
        if seed is not None:
            random.seed(seed)
        self.prob = prob

    def get_node_features(self,mol):
        n_nodes = mol.GetNumAtoms()
        n_node_features = len(self.get_atomic_features(self.unrelated_mol.GetAtomWithIdx(0)))
        node_features = np.zeros((n_nodes, n_node_features))
        mask = np.zeros(n_node_features)
        number_atoms_mask = int(min(max(1,self.prob*n_nodes),n_nodes))
        masked_atoms = random.sample(range(n_nodes),number_atoms_mask)
        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            if atom_id in masked_atoms:
                node_features[atom_id, :] = np.copy(mask)
            else:
                node_features[atom_id, :] = self.get_atomic_features(atom)
            
        node_features = torch.tensor(node_features, dtype = torch.float)

        return node_features

class EdgeRemovalMolToGraph(MolToGraph):

    def __init__(self, prob, seed=None):
        super().__init__()
        if seed is not None:
            random.seed(seed)
        self.prob = prob

    def get_edge_features(self,mol):
        n_edge_features = len(MolToGraph.get_bond_features(self.unrelated_mol.GetBondBetweenAtoms(0,1)))
        n_edge_remove = int(min(max(1,self.prob*mol.GetNumBonds()),mol.GetNumBonds()))
        edges_to_remove = random.sample(range(mol.GetNumBonds()),n_edge_remove)
        n_edges = 2*(mol.GetNumBonds() - n_edge_remove)
        rows = []
        cols = []
        for bond_id,bond in enumerate(mol.GetBonds()):
            if bond_id not in edges_to_remove:
                inicio, final = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [inicio, final]
                cols += [final, inicio]

        torch_rows = torch.tensor(rows,dtype=torch.long)
        torch_cols = torch.tensor(cols,dtype=torch.long)

        edge_index = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # Features de los ejes (n_edges, n_edge_features)
        edge_features = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            edge_features[k] = MolToGraph.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        edge_features = torch.tensor(edge_features, dtype = torch.float)

        return edge_index,edge_features

class SubgraphRemovalMolToGraph(MolToGraph):

    def __init__(self, prob, seed=None):
        super().__init__()
        if seed is not None:
            random.seed(seed)
        self.prob = prob
        self.masked_nodes_ids = []

    def get_node_features(self,mol):
        n_nodes = mol.GetNumAtoms()
        n_node_features = len(self.get_atomic_features(self.unrelated_mol.GetAtomWithIdx(0)))
        number_nodes_to_mask = max(1,self.prob*n_nodes)
        node_features = np.zeros((n_nodes, n_node_features))
        mask = np.zeros(n_node_features)
        initial_node = random.randint(0,n_nodes-1)
        masked_atoms = []
        atoms_to_mask = [initial_node]
        while len(masked_atoms) < number_nodes_to_mask and len(atoms_to_mask) > 0:
            next_to_mask = atoms_to_mask.pop(0)
            masked_atoms.append(next_to_mask)
            atom_to_mask = mol.GetAtomWithIdx(next_to_mask)
            neighbors = atom_to_mask.GetNeighbors()
            for neighbor in neighbors:
                neighbor_id = neighbor.GetIdx()
                if neighbor_id not in masked_atoms and neighbor_id not in atoms_to_mask:
                    atoms_to_mask.append(neighbor_id)

        self.masked_nodes_ids = masked_atoms

        for atom in mol.GetAtoms():
            atom_id = atom.GetIdx()
            if atom_id in masked_atoms:
                node_features[atom_id, :] = np.copy(mask)
            else:
                node_features[atom_id, :] = self.get_atomic_features(atom)
        node_features = torch.tensor(node_features, dtype = torch.float)

        return node_features

    def get_edge_features(self,mol):
        n_edge_features = len(MolToGraph.get_bond_features(self.unrelated_mol.GetBondBetweenAtoms(0,1)))
        rows = []
        cols = []
        for bond_id,bond in enumerate(mol.GetBonds()):
            inicio, final = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if inicio not in self.masked_nodes_ids or final not in self.masked_nodes_ids:
                rows += [inicio, final]
                cols += [final, inicio]

        n_edges = len(rows)
        torch_rows = torch.tensor(rows,dtype=torch.long)
        torch_cols = torch.tensor(cols,dtype=torch.long)

        edge_index = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # Features de los ejes (n_edges, n_edge_features)
        edge_features = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            edge_features[k] = MolToGraph.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        edge_features = torch.tensor(edge_features, dtype = torch.float)

        return edge_index,edge_features

def create_graphs(smiles,target,create_graphs_params):
    normal_rep = 1
    atom_masking_rep = create_graphs_params["atom_masking_rep"]
    atom_masking_prob = create_graphs_params["atom_masking_prob"]
    edge_removal_rep = create_graphs_params["edge_removal_rep"]
    edge_removal_prob = create_graphs_params["edge_removal_prob"]
    subgraph_removal_rep = create_graphs_params["subgraph_removal_rep"]
    subgraph_removal_prob = create_graphs_params["subgraph_removal_prob"]

    graphs = []

    graphs.extend(create_graphs_normal(smiles,target,normal_rep))
    graphs.extend(create_graphs_atom_masking(smiles,target,atom_masking_prob, atom_masking_rep))
    graphs.extend(create_graphs_edge_removal(smiles,target,edge_removal_prob, edge_removal_rep))
    graphs.extend(create_graphs_subgraph_removal(smiles,target,subgraph_removal_prob, subgraph_removal_rep))

    return graphs

def create_graphs_normal(smiles, target, rep):
    if rep > 0:
        graph_generator = MolToGraph()
        graphs = graph_generator.create_graph_list(smiles,target)
        return graphs
    else:
        return []


def create_graphs_atom_masking(smiles,target,prob,rep):
    if rep > 0:
        atom_masking_generator = AtomMaskingMolToGraph(prob)
        graphs = atom_masking_generator.create_graph_list(smiles,target,rep)
        return graphs
    else:
        return []

def create_graphs_edge_removal(smiles,target,prob,rep):
    if rep > 0:
        edge_removal_generator = EdgeRemovalMolToGraph(prob)
        graphs = edge_removal_generator.create_graph_list(smiles,target,rep)
        return graphs
    else:
        return []

def create_graphs_subgraph_removal(smiles,target,prob,rep):
    if rep > 0:
        subgraph_removal_generator = SubgraphRemovalMolToGraph(prob)
        graphs = subgraph_removal_generator.create_graph_list(smiles,target,rep)
        return graphs
    else:
        return []

if __name__ == "__main__":
    smiles = "Brc1ccccc1Br"
    mol_to_graph = MolToGraph()
    atom_masking = AtomMaskingMolToGraph(0.5,0)
    edge_removal = EdgeRemovalMolToGraph(0.5,0)
    subgraph_removal = SubgraphRemovalMolToGraph(0.5,0)
    normal_graphs = mol_to_graph.create_graph_list([smiles],[1],rep=1)
    atom_masking_graphs = atom_masking.create_graph_list([smiles],[1],rep=4)
    edge_removal_graphs = edge_removal.create_graph_list([smiles],[1],rep=4)
    subgraph_removal_graphs = subgraph_removal.create_graph_list([smiles],[1],rep=4)
    print(normal_graphs)
    print(atom_masking_graphs)
    print(edge_removal_graphs)
    print(subgraph_removal_graphs)

