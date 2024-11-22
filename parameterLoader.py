import json
import argparse

def read_gnn_parameter_file(filename):
    parameter_file = open(filename,"r")
    parameter_data = json.load(parameter_file)
    parameter_file.close()
    return parameter_data["gnn_args"],parameter_data["optimizer_args"]

def read_graph_creation_file(filename):
    parameter_file = open(filename,"r")
    parameter_data = json.load(parameter_file)
    parameter_file.close()
    return parameter_data

def read_command_line_args():
    parser = argparse.ArgumentParser(prog="GNN",
                                     description="Graph Neural Network for chemical properties prediction")
    parser.add_argument("task",choices=["cv","save","eval","contr_atoms","contr_fgs"])
    parser.add_argument("-v","--folds",default=5,required=False,type=int)
    parser.add_argument("-f","--file",required=True,type=str)
    parser.add_argument("-s","--split",default=0,required=False,type=int)
    parser.add_argument("-m","--model",required=False,type=str,default="model.pickle")
    parser.add_argument("-e","--epoch",required=False,type=int,default=100)
    parser.add_argument("-t","--train",required=False)
    parser.add_argument("-b","--batch",required=False,type=int,default=512)

    args = parser.parse_args()

    return args

