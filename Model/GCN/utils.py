from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
from Model.GCN import mol2graph

def get_data(data_path):
    mols, labels = [], []
    with open(data_path, 'r') as f:
        smis = f.read().splitlines()
    for smi in smis:
        smi = smi.split(' ')
        labels.append(int(smi[0].strip('[]')))
        smi = smi[1:]
        smi = ''.join(smi)
        mols.append(Chem.MolFromSmiles(smi)) 
    return np.array(mols), np.array(labels)

"""
get_neg_sample: select negative sample according to the frequent distribution of library.
Correct fragments(y) and fragments couldn't be connected to target(y_mask) are masked. """
@torch.no_grad()
def get_neg_sample(freq, y):
    # y: (batch_size, )
    # freq: (1, ), frequency of templates
    batch_size = y.size(0)
    freq = freq.repeat(batch_size, 1)
    freq.scatter_(1, y.unsqueeze(1), 0)
    neg_idxs = torch.multinomial(freq, 1, True).view(-1)
    return neg_idxs

def template_prediction(GCN_model, input_smi, num_sampling, GCN_device=None):
    mol = Chem.MolFromSmiles(input_smi)
    data = mol2graph.mol2vec(mol).to(GCN_device)
    with torch.no_grad():
        output = GCN_model.forward(data.x, data.edge_index, data.batch).squeeze() # shape(1, 1000) -> (1000,)
        try:
            _, indices = torch.topk(output, num_sampling)
        except:
            indices = None
    return indices

def batch_template_prediction(GCN_model, input_smi, num_sampling=5, GCN_device=None):
    mol = Chem.MolFromSmiles(input_smi)
    data = mol2graph.mol2vec(mol).to(GCN_device)
    output = GCN_model.forward(data.x, data.edge_index, data.batch).squeeze() # shape(1, 1000) -> (1000,)
    _, indices = torch.topk(output, num_sampling)
    return indices
    
def check_templates(indices, input_smi, r_dict):
    matched_indices = []
    molecule = Chem.MolFromSmiles(input_smi)
    for i in indices:
        idx = str(i.item())
        rsmi = r_dict[idx]
        rxn = AllChem.ReactionFromSmarts(rsmi)
        reactants = rxn.GetReactants()
        flag = False
        for reactant in reactants:
            if molecule.HasSubstructMatch(reactant):
                flag = True
        if flag == True:
            matched_indices.append(f'[{i.item()}]')
    return matched_indices # list of string, ex) ['[0]', '[123]', ... '[742]']

