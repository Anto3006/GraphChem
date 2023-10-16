import pandas as pd
from rdkit.Chem import CanonSmiles

def read_smiles_target(filename):
    data = pd.read_csv(filename)
    canon_smiles = []
    for smiles in data["smiles"]:
        canon_smiles.append(CanonSmiles(smiles))
    return canon_smiles,data[data.columns[1]]
