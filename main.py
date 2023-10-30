from data_reader import read_smiles_target
from molecule_to_graph import create_graph_list, get_atom_in_molecule_count, create_graph_molecule
from torch_geometric.loader import DataLoader
from model import GNN
import torch.optim
from math import sqrt
from sklearn.model_selection import train_test_split
from rdkit import Chem
from sklearn.model_selection import KFold
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import Draw


smiles, target = read_smiles_target("log_koa_smiles_2.csv")

train_smiles,test_smiles,train_target,test_target = train_test_split(smiles,target,test_size=0.15,random_state=3006)

train_molecules_graphs = create_graph_list(train_smiles,train_target)
test_molecules_graphs = create_graph_list(test_smiles,test_target)


def cross_validation(fold_count,train_data,max_epoch,train_batch_size=64,validation_batch_size=1,seed=3006):
    kf = KFold(fold_count,shuffle=True,random_state=3006)
    folds = kf.split(train_data)
    for fold, (train_index,validation_index) in enumerate(folds):
        torch.manual_seed(seed)
        gnn = GNN(train_molecules_graphs[0].num_features,6,64,4)
        gnn = gnn.to('cuda')
        optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        train_fold = [train_data[i] for i in train_index]
        validation_fold = [train_data[i] for i in validation_index]
        train_loader_fold = DataLoader(dataset = train_fold, batch_size = train_batch_size)
        validation_loader_fold = DataLoader(dataset = validation_fold, batch_size = validation_batch_size)
        for epoch in range(max_epoch):
            train(gnn,optimizer,criterion,train_loader_fold)
            train_acc = test(gnn,train_loader_fold)
            validation_acc = test(gnn,validation_loader_fold)
            print(f'Fold {fold:02d} Epoch: {epoch:04d}, Train Acc: {train_acc}, Validation Acc: {validation_acc}')


def train(gnn,optimizer,criterion,loader):
    gnn.train()
    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to('cuda')
        out = gnn(data.x, data.edge_index, data.edge_attr,data.batch)  # Perform a single forward pass.
        loss = criterion(out, torch.unsqueeze(data.y,1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(gnn,loader):
    gnn.eval()
    error = 0
    for data in loader: 
        data = data.to('cuda')
        out = gnn(data.x, data.edge_index, data.edge_attr,data.batch)  
        error += torch.nn.MSELoss(reduction="sum")(out, torch.unsqueeze(data.y,1)).item()
    return sqrt(error/len(loader.dataset))

def contributions(gnn,smiles,filename="mol.png"):
    gnn.eval()
    graph = create_graph_molecule(smiles).to("cuda")
    canon_smiles = Chem.CanonSmiles(smiles)
    mol = Chem.MolFromSmiles(canon_smiles)
    batch = torch.tensor([0 for i in range(len(graph.x))]).to("cuda")
    out = gnn(graph.x,graph.edge_index,graph.edge_attr,batch)
    if len(mol.GetAtoms()) > 1:
        for atom in mol.GetAtoms():
            masked_atom = atom.GetIdx()
            masked_out = gnn(graph.x,graph.edge_index,graph.edge_attr,batch,True,masked_atom)
            contribution = (out-masked_out).item()
            atom.SetProp('atomNote',f"{contribution:.3f}")
    img = Draw.MolToImage(mol,filename=filename,size=(600,600))
    img.text((50,50),f"Prediction = {out.item():.3f}",fill=(0,0,0))

cross_validation(10,train_molecules_graphs,1000)