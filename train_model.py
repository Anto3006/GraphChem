from model import GNN
import torch
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from math import sqrt

device = 'cuda'

def create_gnn(gnn_args,optimizer_args,node_feature_count,edge_feature_count):
    gnn = GNN(  gnn_args["conv_layer_count"],gnn_args["conv_layer_dropout"], \
                gnn_args["fc_layer_count"],gnn_args["fc_layer_dropout"] , \
                node_feature_count,edge_feature_count, \
                gnn_args["hidden_layer_size"], gnn_args["attention_heads"], \
                skip_connection=gnn_args["skip_connection"])
    gnn = gnn.to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr = optimizer_args["lr"], weight_decay=optimizer_args["weight_decay"])
    return gnn,optimizer

def cross_validation(gnn_args,optimizer_args,fold_count,train_data,max_epoch,train_batch_size=64,validation_batch_size=1,seed=3006):
    kf = KFold(fold_count,shuffle=True,random_state=3006)
    folds = kf.split(train_data)
    results = {}
    node_feature_count = train_data[0].num_features
    edge_feature_count = train_data[0].num_edge_features
    for fold, (train_index,validation_index) in enumerate(folds):
        torch.manual_seed(seed)
        gnn,optimizer = create_gnn(gnn_args,optimizer_args,node_feature_count,edge_feature_count)
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
            validation_accs.append(validation_acc)
            print(f'Fold {fold:02d} Epoch: {epoch:04d}, Train Acc: {train_acc}, Validation Acc: {validation_acc}')
        x = [epoch for epoch in range(max_epoch)]
        print("Graficando")
        plt.plot(x,train_accs)
        print("Train graficado")
        plt.plot(x,validation_accs)
        print("Validation graficado")
        plt.savefig("fold_cv_" + str(fold)+".png")
        print("Figura guardada")
        plt.cla()
    pd.DataFrame(results).to_csv("cv.csv")

def get_trained_model(gnn_args,optimizer_args,train_data,max_epoch,train_batch_size=64,seed=3006):
    torch.manual_seed(seed)
    node_feature_count = train_data[0].num_features
    edge_feature_count = train_data[0].num_edge_features
    gnn,optimizer = create_gnn(gnn_args,optimizer_args,node_feature_count,edge_feature_count)
    criterion = torch.nn.MSELoss()
    train_loader = DataLoader(dataset = train_data, batch_size = train_batch_size)
    for epoch in range(max_epoch):
        train(gnn,optimizer,criterion,train_loader)
        train_acc = test(gnn,train_loader)
        print(f'Epoch: {epoch:04d}, Train Acc: {train_acc}')
    return gnn

def save_trained_model(gnn,model_filename="model.pickle"):
    model_file = open(model_filename,"wb")
    pickle.dump(gnn,model_file)
    model_file.close()


def train(gnn,optimizer,criterion,loader):
    gnn.train()
    for data in loader:  # Iterate in batches over the training dataset.
        #data = data.to(device)
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

def evaluate_model(model_filename,test_data):
    model = None
    try:
        model_file = open(model_filename,"rb")
        model = pickle.load(model_file)
        model_file.close()
        test_loader = DataLoader(dataset = test_data, batch_size = 1)
        result = test(model,test_loader)
        return result
    except:
        print("Error: model not found")
        return None

def load_model(model_filename):
    model = None
    try:
        model_file = open(model_filename,"rb")
        model = pickle.load(model_file)
        model_file.close()
    except:
        print("Error: model not found")
    return model
