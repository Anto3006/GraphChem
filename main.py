from data_reader import read_smiles_target
from molecule_to_graph import create_graph_list, get_atom_in_molecule_count
from torch_geometric.loader import DataLoader
from model import GNN
import torch.optim
from math import sqrt
from sklearn.model_selection import train_test_split

smiles, target = read_smiles_target("log_Koa_smiles.csv")

train_smiles,test_smiles,train_target,test_target = train_test_split(smiles,target,test_size=0.15,random_state=3006)

train_molecules_graphs = create_graph_list(train_smiles,train_target)
test_molecules_graphs = create_graph_list(test_smiles,test_target)

train_loader = DataLoader(dataset = train_molecules_graphs, batch_size = 64)
test_loader = DataLoader(dataset = test_molecules_graphs, batch_size = 1)

gnn = GNN(train_molecules_graphs[0].num_features,6,64,8)
gnn = gnn.to('cuda')
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

def train(loader):
    gnn.train()

    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to('cuda')
        out = gnn(data.x, data.edge_index, data.edge_attr,data.batch)  # Perform a single forward pass.
        loss = criterion(out, torch.unsqueeze(data.y,1))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    gnn.eval()
    error = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to('cuda')
        out = gnn(data.x, data.edge_index, data.edge_attr,data.batch)  
        error += torch.nn.MSELoss(reduction="sum")(out, torch.unsqueeze(data.y,1)).item()
    return sqrt(error/len(loader.dataset))

for epoch in range(1500):
    train(train_loader)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')