from data_reader import read_smiles_target
from molecule_to_graph import create_graphs
from sklearn.model_selection import train_test_split
from parameterLoader import read_gnn_parameter_file, read_command_line_args, read_graph_creation_file
from train_model import cross_validation, get_trained_model, save_trained_model, evaluate_model, load_model
from contributions import contributions_atoms,contributions_fgs
import numpy as np

GNN_PARAMETER_FILE = "parametersGNN.json"
GRAPH_CREATION_PARAMETER_FILE ="parametersGraphCreation.json"

def main():
    args = read_command_line_args()
    gnn_args,optimizer_args = read_gnn_parameter_file(GNN_PARAMETER_FILE)
    graph_creation_params = read_graph_creation_file(GRAPH_CREATION_PARAMETER_FILE)
    if args.task == "cv":
        folds = args.folds
        split = args.split
        train_file = args.file
        epoch = args.epoch
        batch = args.batch
        smiles, target = read_smiles_target(train_file)
        if split > 0:
            smiles,_,target,_ = train_test_split(smiles,target,test_size=split*0.01,random_state=3006)
        molecules_graphs = create_graphs(smiles,target,graph_creation_params)
        cross_validation(gnn_args,optimizer_args,folds,molecules_graphs,epoch,batch)
    elif args.task == "save":
        train_file = args.file
        split = args.split
        epoch = args.epoch
        batch = args.batch
        model_name = args.model
        smiles, target = read_smiles_target(train_file)
        if split > 0:
            smiles,_,target,_ = train_test_split(smiles,target,test_size=split*0.01,random_state=3006)
        molecules_graphs = create_graphs(smiles,target,graph_creation_params)
        model = get_trained_model(gnn_args,optimizer_args,molecules_graphs,epoch,batch)
        save_trained_model(model,model_name)
    elif args.task == "eval":
        test_file = args.file
        model_name = args.model
        smiles, target = read_smiles_target(test_file)
        molecules_graphs = create_graphs(smiles,target,graph_creation_params)
        result = evaluate_model(model_name,molecules_graphs)
        if result is not None:
            print("Evaluation: ",result)
    elif args.task == "contr_atoms":
        contr_file = args.file
        model_name = args.model
        smiles, _ = read_smiles_target(contr_file)
        model = load_model(model_name)
        if model is not None:
            for i,s in enumerate(smiles):
                contributions_atoms(model,s,f"mol_{i}.png")
    elif args.task == "contr_fgs":
        contr_file = args.file
        model_name = args.model
        smiles, _ = read_smiles_target(contr_file)
        model = load_model(model_name)
        contr_fgs = {}
        if model is not None:
            for i,s in enumerate(smiles):
                print(i)
                contr_fgs_mol,x = contributions_fgs(model,s,f"mol_{i}.png")
                for fg_type in contr_fgs_mol:
                    if fg_type in contr_fgs:
                        contr_fgs[fg_type].extend(contr_fgs_mol[fg_type])
                    else:
                        contr_fgs[fg_type] = contr_fgs_mol[fg_type]
            fg_file = open("fgs.csv","w")
            fg_file.write("FG,AMOUNT,AVG,STD\n")
            for fg_type in contr_fgs:
                print(fg_type,a[fg_type])
                avg = sum(contr_fgs[fg_type])/len(contr_fgs[fg_type])
                sd = np.std(contr_fgs[fg_type])
                fg_file.write(f"{fg_type},{len(contr_fgs[fg_type])},{avg},{sd}\n")
            fg_file.close()
    else:
        print("Error: option not valid")


main()

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
