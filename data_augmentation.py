import copy
import random
import torch
import math
from molecule_to_graph import create_graph_molecule
import torch

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
        mask = torch.tensor([0 for _ in range(node_size)])
        for atom_idx in masked_atoms:
            augmented_graph.x[atom_idx, :] = mask
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
        # Delete in both directions
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

class RandomSubgraphDeletion:

    def __init__(self, prob):
        self.prob = prob

    def apply_augmentation(self, molecule_graph, seed=None):
        if seed != None:
            random.seed(seed)


        number_atoms = molecule_graph.x.size(0)
        number_masked_atoms = max(1, math.floor(self.prob*number_atoms))
        node_size = molecule_graph.x.size(1)
        augmented_graph = copy.deepcopy(molecule_graph)
        mask = torch.tensor([0 for _ in range(node_size)])
        
        initial_atom_id = torch.tensor(random.randint(0,number_atoms))
        current_masked_count = 0
        current_masked_atoms_ids = []
        atoms_to_mask_ids = [initial_atom_id]
        while current_masked_count < number_masked_atoms and len(atoms_to_mask_ids) > 0:
            next_atom_id = atoms_to_mask_ids.pop(0)
            current_masked_count += 1
            current_masked_atoms_ids.append(next_atom_id)
            augmented_graph.x[next_atom_id, :] = mask
            neighbors = augmented_graph.edge_index[1, augmented_graph.edge_index[0] == next_atom_id]
            for neighbor_id in neighbors:
                if neighbor_id not in current_masked_atoms_ids and neighbor_id not in atoms_to_mask_ids:
                    atoms_to_mask_ids.append(neighbor_id)
            print(next_atom_id,neighbors)
        current_masked_atoms_ids = torch.tensor(current_masked_atoms_ids)
        x = augmented_graph.subgraph(current_masked_atoms_ids)
        
    

if __name__ == "__main__":
    graph = create_graph_molecule(smiles="Brc1ccccc1Br",target=1)
    print(graph)
    data_aug =RandomSubgraphDeletion(0.5)
    data_aug.apply_augmentation(graph)


