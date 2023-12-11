from data_reader import read_smiles_target
from molecule_to_graph import create_graph_list
from sklearn.model_selection import train_test_split


smiles, target = read_smiles_target("log_koa_smiles_2.csv")

train_smiles,test_smiles,train_target,test_target = train_test_split(smiles,target,test_size=0.15,random_state=3006)

train_molecules_graphs = create_graph_list(train_smiles,train_target)
test_molecules_graphs = create_graph_list(test_smiles,test_target)

gnn_args = {
    "node_feature_count": train_molecules_graphs[0].num_features,
    "edge_feature_count": train_molecules_graphs[0].num_edge_features,
    "hidden_layer_size": 256,
    "attention_heads": 4,
    "conv_layer_count" : 3,
    "conv_layer_dropout" : [0.2,0.2,0.2],
    "fc_layer_count" : 5,
    "fc_layer_dropout" : [0,0,0,0,0],
    "skip_connection": True
}

optimizer_args = {
    "lr": 0.00005,
    "weight_decay": 0.0005
}

"""
#cross_validation(gnn_args,optimizer_args,10,train_molecules_graphs,300,train_batch_size=128)
gnn = get_trained_model(gnn_args,optimizer_args,train_molecules_graphs,50,train_batch_size=128,seed=3006)
contr_fgs = {}
a = {}
for i,smiles in enumerate(train_smiles[:200]):
    contr_fgs_mol,x = contributions_fgs(gnn,smiles,str(i)+".png")
    for fg_type in contr_fgs_mol:
        if fg_type in contr_fgs:
            contr_fgs[fg_type].extend(contr_fgs_mol[fg_type])
            a[fg_type].append(x[fg_type])
        else:
            contr_fgs[fg_type] = contr_fgs_mol[fg_type]
            a[fg_type] = [x[fg_type]]

fg_file = open("fgs.csv","w")
fg_file.write("FG,AMOUNT,AVG,STD\n")
for fg_type in contr_fgs:
    print(fg_type,a[fg_type])
    avg = sum(contr_fgs[fg_type])/len(contr_fgs[fg_type])
    sd = np.std(contr_fgs[fg_type])
    fg_file.write(f"{fg_type},{len(contr_fgs[fg_type])},{avg},{sd}\n")
fg_file.close()

"""