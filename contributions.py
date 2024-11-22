from rdkit.Chem import Draw
from PIL import ImageDraw
from PIL import ImageFont
from molecule_to_graph import MolToGraph
from rdkit.Chem import CanonSmiles, MolFromSmiles
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir,"IFG"))
import ifg as IFG
from torch import tensor
from rdkit.Chem.Draw import rdMolDraw2D
import random


device = "cuda"
fg_colors = {}

def contributions_atoms(gnn,smiles,filename="mol.png"):
    gnn.eval()
    graph_generator = MolToGraph()
    graph = graph_generator.smiles_to_graph(smiles).to(device)
    canon_smiles = CanonSmiles(smiles)
    mol = MolFromSmiles(canon_smiles)
    batch = tensor([0 for i in range(len(graph.x))]).to(device)
    out = gnn(graph.x,graph.edge_index,graph.edge_attr,batch)
    if len(mol.GetAtoms()) > 1:
        for atom in mol.GetAtoms():
            masked_atom = atom.GetIdx()
            masked_out = gnn(graph.x,graph.edge_index,graph.edge_attr,batch,True,[masked_atom])
            contribution = (out-masked_out).item()
            atom.SetProp('atomNote',f"{contribution:.3f}")
    img = Draw.MolToImage(mol,size=(600,600))
    prediction = out.item()
    font = ImageFont.truetype("arial.ttf",40)
    ImageDraw.Draw(img).text((50,50),f"Prediction : {prediction:.3f}",fill=(0,0,0),font=font)
    img.save(filename)

def contributions_fgs(gnn,smiles,filename="mol.png"):
    gnn.eval()
    graph_generator = MolToGraph()
    graph = graph_generator.smiles_to_graph(smiles).to(device)
    canon_smiles = CanonSmiles(smiles)
    mol = MolFromSmiles(canon_smiles)
    fgs = IFG.identify_functional_groups(mol)
    batch = tensor([0 for i in range(len(graph.x))]).to(device)
    out = gnn(graph.x,graph.edge_index,graph.edge_attr,batch)
    fgs_atoms = []
    contr = []
    contr_fgs = {}
    x = {}
    if len(mol.GetAtoms()) > 1:
        for fg in fgs:
            masked_atoms = list(fg.atomIds)
            if len(masked_atoms) < len(mol.GetAtoms()):
                masked_out = gnn(graph.x,graph.edge_index,graph.edge_attr,batch,True,masked_atoms)
                contribution = (out-masked_out).item()
                contr.append(contribution)
                fg_type = fg.type
                x[fg_type] = len(masked_atoms)
                if fg_type in contr_fgs:
                    contr_fgs[fg_type].append(contribution)
                else:
                    contr_fgs[fg_type] = [contribution]
                fgs_atoms.append(list(masked_atoms))
        if len(contr) >= 1:
            print("Drawing")
            draw_molecule_highlight_fg(mol,fgs,contr,filename)
    return contr_fgs,x

def draw_molecule_highlight_fg(mol,fgs,contributions,name):
    drawing = rdMolDraw2D.MolDraw2DCairo(500,500)
    atoms = []
    bonds = []
    atom_colors = {}
    bond_colors = {}
    i = 0
    for fg,contribution in zip(fgs,contributions):
        if fg.type not in fg_colors:
            fg_colors[fg.type] = (random.random(),random.random(),random.random())
        fg_color = fg_colors[fg.type]
        atoms_fg = fg.atomIds
        mol.GetAtomWithIdx(atoms_fg[0]).SetProp('atomNote', f"{contribution:.2f}")
        for atom in atoms_fg:
            atoms.append(atom)
            atom_colors[atom] = fg_color
        for atom1 in atoms_fg:
            for atom2 in atoms_fg:
                if atom1 != atom2:
                    bond = mol.GetBondBetweenAtoms(atom1,atom2)
                    if bond:
                        bonds.append(bond.GetIdx())
                        bond_colors[bond.GetIdx()] = fg_color
        i += 1
    rdMolDraw2D.PrepareAndDrawMolecule(drawing, mol, highlightAtoms=atoms, \
                                        highlightBonds=bonds, \
                                        highlightBondColors=bond_colors, \
                                        highlightAtomColors=atom_colors)
    drawing.FinishDrawing()
    drawing.WriteDrawingText(name)
