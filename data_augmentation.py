import copy
import random
import torch
import math

from auglichem.molecule import MotifRemoval

class RandomAtomMask:

    def __init__(self, prob):
        self.prob = prob

    def apply_augmentation(self, molecule_graph, seed=None):
        if seed != None:
            random.seed(seed)

        number_atoms = molecule_graph.x.size(0)
        number_masked_atoms = max(1, math.floor(self.prob*number_atoms))
        masked_atoms = random.sample(range(number_atoms), number_masked_atoms)
        node_size = molecule_graph.x.size(1)
    
        augmented_graph = copy.deepcopy(molecule_graph)
        for atom_idx in masked_atoms:
            augmented_graph.x[atom_idx, :] = torch.tensor([0 for _ in range(node_size)])
        return augmented_graph

class RandomBondDeletion:

    def __init__(self,prob):
        self.prob = prob
        
    def apply_augmentation(self, molecule_graph, seed=None):
        if seed != None:
            random.seed(seed)

        number_bonds = molecule_graph.edge_index.size(1) // 2
        number_edge_features = molecule_graph.edge_attr.size(1)
        number_deleted_bonds = max(1, math.floor(self.prob*number_bonds))
        deleted_bonds_single = random.sample(range(number_bonds), number_deleted_bonds)
        deleted_bonds = [2*idx for idx in deleted_bonds_single] + [2*idx + 1 for idx in deleted_bonds_single]
    
        augmented_graph = copy.deepcopy(molecule_graph)
        augmented_graph.edge_index = torch.zeros((2,2*(number_bonds-number_deleted_bonds)),dtype=torch.long)
        augmented_graph.edge_attr = torch.zeros((2*(number_bonds-number_deleted_bonds),number_edge_features),dtype=torch.float)
        count = 0
        for bond_idx in range(2*number_bonds):
            if bond_idx not in deleted_bonds:
                augmented_graph.edge_index[:,count] = molecule_graph.edge_index[:,bond_idx]
                augmented_graph.edge_attr[count,:] = molecule_graph.edge_attr[bond_idx,:]
                count += 1
        return augmented_graph

