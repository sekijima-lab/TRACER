import numpy as np
import pandas as pd
import pickle
import hydra

import warnings
warnings.filterwarnings('ignore')

import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem, QED

def getReward(name):
    if name == "QED":
        return QEDReward()
    else:
        with open(hydra.utils.get_original_cwd() + f'/Model/QSAR/qsar_{name}_optimized.pkl', mode='rb') as f:
            qwar_model = pickle.load(f)
        return QSAR_Reward(qwar_model)

class Reward:
    def __init__(self):
        self.vmin = -100
        self.max_r = -10000
        return

    def reward(self):
        raise NotImplementedError()

class QSAR_Reward(Reward):
    def __init__(self, qsar_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qsar_model = qsar_model

    def reward(self, score_que:list = None):
        max_smi = None
        scores = []
        mols = [Chem.MolFromSmiles(smi) for smi in score_que]
        ecfps = []
        None_indices = []
        for i, mol in enumerate(mols):
            if mol is not None:
                ecfps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
            else:
                None_indices.append(i)
                ecfps.append([0]*2048)
        if len(ecfps) == 0:
            return [], None, None
        ecfp6_array = np.array(ecfps)
        X = pd.DataFrame(ecfp6_array, columns=[f'bit_{i}' for i in range(2048)])
        y_pred = self.qsar_model.predict_proba(X)[:, 1]
        for None_idx in None_indices:
            y_pred[None_idx] = np.nan
        max_score = np.nanmax(y_pred)
        for smi, score in zip(score_que, y_pred):
            if score == np.nan:
                pass
            elif score == max_score:
                max_smi = smi
            scores.append((smi, score))
        return scores, max_smi, max_score
    
    def reward_remove_nan(self, score_que:list = None):
        max_smi = None
        scores = []
        # convert smiles to mol if mol is not none.
        valid_smiles = []
        mols = []
        for smi in score_que:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                mols.append(mol)
        ecfps = []
        for i, mol in enumerate(mols):
            if mol is not None:
                ecfps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048))
        if len(ecfps) == 0:
            return [], None, None
        ecfp6_array = np.array(ecfps)
        X = pd.DataFrame(ecfp6_array, columns=[f'bit_{i}' for i in range(2048)])
        y_pred = self.qsar_model.predict_proba(X)[:, 1]
        max_score = np.nanmax(y_pred)
        for smi, score in zip(valid_smiles, y_pred):
            if score == max_score:
                max_smi = smi
            scores.append((smi, score))
        return scores, max_smi, max_score

class QEDReward(Reward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vmin = 0

    def reward(self, smi):
        mol = Chem.MolFromSmiles(smi)
        try:
            score = QED.qed(mol)
        except:
            score = None

        return score