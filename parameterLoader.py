import json
import argparse

def read_gnn_parameter_file(filename):
    parameter_file = open(filename,"r")
    parameter_data = json.load(parameter_file)
    parameter_file.close()
    return parameter_data["gnn_args"],parameter_data["optimizer_args"]

def read_command_line_args():
    parser = argparse.ArgumentParser(prog="GNN",
                                     description="Graph Neural Network for chemical properties prediction")
    parser.add_argument("task",choices=["cv","save","eval","contr_atoms","contr_fgs"],required=True)
    parser.add_argument("-f","--folds",default=5,required=False,type=int)
    parser.add_argument("")