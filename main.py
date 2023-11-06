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
from PIL import ImageDraw
from PIL import ImageFont
import pandas as pd
import matplotlib.pyplot as plt


smiles, target = read_smiles_target("log_koa_smiles_2.csv")

train_smiles,test_smiles,train_target,test_target = train_test_split(smiles,target,test_size=0.15,random_state=3006)

train_molecules_graphs = create_graph_list(train_smiles,train_target)
test_molecules_graphs = create_graph_list(test_smiles,test_target)

device = 'cuda'

def create_gnn(gnn_args):
    gnn = GNN(  gnn_args["conv_layer_count"],gnn_args["conv_layer_dropout"], \
                gnn_args["fc_layer_count"],gnn_args["fc_layer_dropout"] , \
                gnn_args["node_feature_count"],gnn_args["edge_feature_count"], \
                gnn_args["hidden_layer_size"], gnn_args["attention_heads"], \
                skip_connection=gnn_args["skip_connection"])
    gnn = gnn.to(device)
    return gnn


def cross_validation(gnn_args,fold_count,train_data,max_epoch,train_batch_size=64,validation_batch_size=1,seed=3006):
    kf = KFold(fold_count,shuffle=True,random_state=3006)
    folds = kf.split(train_data)
    results = {}
    for fold, (train_index,validation_index) in enumerate(folds):
        torch.manual_seed(seed)
        gnn = create_gnn(gnn_args)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        train_fold = [train_data[i] for i in train_index]
        validation_fold = [train_data[i] for i in validation_index]
        train_loader_fold = DataLoader(dataset = train_fold, batch_size = train_batch_size)
        validation_loader_fold = DataLoader(dataset = validation_fold, batch_size = validation_batch_size)
        results[fold] = {"training":{},"validation":{}}
        train_accs = []
        validation_accs = []
        for epoch in range(max_epoch):
            train(gnn,optimizer,criterion,train_loader_fold)
            train_acc = test(gnn,train_loader_fold)
            validation_acc = test(gnn,validation_loader_fold)
            results[fold]["training"][epoch] = train_acc
            train_accs.append(train_acc)
            results[fold]["validation"][epoch] = validation_acc
            validation_accs.append(validation_accs)
            print(f'Fold {fold:02d} Epoch: {epoch:04d}, Train Acc: {train_acc}, Validation Acc: {validation_acc}')
        x = [epoch for epoch in range(1000)]
        plt.plot(x,train_accs)
        plt.plot(x,validation_accs)
        plt.show()
    pd.DataFrame(results).to_csv("cv.csv")

def get_trained_model(gnn_args,train_data,max_epoch,train_batch_size=64,seed=3006):
    torch.manual_seed(seed)
    gnn = create_gnn(gnn_args)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    train_loader = DataLoader(dataset = train_data, batch_size = train_batch_size)
    for epoch in range(max_epoch):
        train(gnn,optimizer,criterion,train_loader)
        train_acc = test(gnn,train_loader)
        print(f'Epoch: {epoch:04d}, Train Acc: {train_acc}')
    return gnn


def train(gnn,optimizer,criterion,loader):
    gnn.train()
    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = gnn(data.x, data.edge_index, data.edge_attr,data.batch)  # Perform a single forward pass.
        loss = criterion(out, torch.unsqueeze(data.y,1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(gnn,loader):
    gnn.eval()
    error = 0
    for data in loader: 
        data = data.to(device)
        out = gnn(data.x, data.edge_index, data.edge_attr,data.batch)  
        error += torch.nn.MSELoss(reduction="sum")(out, torch.unsqueeze(data.y,1)).item()
    return sqrt(error/len(loader.dataset))

def contributions(gnn,smiles,filename="mol.png"):
    gnn.eval()
    graph = create_graph_molecule(smiles).to(device)
    canon_smiles = Chem.CanonSmiles(smiles)
    mol = Chem.MolFromSmiles(canon_smiles)
    batch = torch.tensor([0 for i in range(len(graph.x))]).to(device)
    out = gnn(graph.x,graph.edge_index,graph.edge_attr,batch)
    if len(mol.GetAtoms()) > 1:
        for atom in mol.GetAtoms():
            masked_atom = atom.GetIdx()
            masked_out = gnn(graph.x,graph.edge_index,graph.edge_attr,batch,True,masked_atom)
            contribution = (out-masked_out).item()
            atom.SetProp('atomNote',f"{contribution:.3f}")
    img = Draw.MolToImage(mol,size=(600,600))
    prediction = out.item()
    font = ImageFont.truetype("arial.ttf",40)
    ImageDraw.Draw(img).text((50,50),f"Prediction : {prediction:.3f}",fill=(0,0,0),font=font)
    img.save(filename)

gnn_args = {
    "node_feature_count": train_molecules_graphs[0].num_features,
    "edge_feature_count": 6,
    "hidden_layer_size": 64,
    "attention_heads": 4,
    "conv_layer_count" : 3,
    "conv_layer_dropout" : [0,0,0],
    "fc_layer_count" : 3,
    "fc_layer_dropout" : [0,0,0],
    "skip_connection": True
}

cross_validation(gnn_args,10,train_molecules_graphs,1000)
#gnn = get_trained_model(gnn_args,train_molecules_graphs,100)
#for i,smiles in enumerate(train_smiles):
    #contributions(gnn,smiles,str(i)+".png")