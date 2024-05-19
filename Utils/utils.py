import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Descriptors

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def smi_tokenizer(smi):
    '''
    Tokenize a SMILES molecule or reaction
    '''
    import re
    pattern =  '(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

class Node:
    def __init__(self):
        self.parent = None
        self.template = None
        self.path = []
        self.depth = -100
        self.visit = 1
        self.children = []
        self.imm_score = 0
        self.cum_score = 0
        self.c = 1
        self.id = -1
        self.rollout_result = ('None', -1000)

    def add_Node(self, c):
        c.parent = self
        c.depth = self.depth + 1
        self.children.append(c)

    def calc_UCB(self):
        if self.visit == 0:
            ucb = 1e+6
        else:
            ucb = self.cum_score/self.visit + self.c*math.sqrt(2*math.log(self.parent.visit)/self.visit)
        return ucb

    def select_children(self):
        children_ucb = []
        for cn in self.children:
            children_ucb.append(cn.calc_UCB())
        max_ind = np.random.choice(np.where(np.array(children_ucb) == max(children_ucb))[0])
        return self.children[max_ind]

    def select_children_rand(self):
        indices = list(range(0, len(self.children)))
        ind = np.random.choice(indices)
        return self.children[ind]


class RootNode(Node):
    def __init__(self, c=1/np.sqrt(2)):
        super().__init__()
        self.smi = '&&'
        self.depth = 0

        self.c = c

class NormalNode(Node):
    def __init__(self, smi, c=1/np.sqrt(2)):
        super().__init__()
        self.smi = smi
        self.c = c
        self.template = None

    def remove_Node(self):
        self.parent.children.remove(self)

def read_smilesset(path):
    smiles_list = []
    with open(path) as f:
        for smiles in f:
            smiles_list.append(smiles.rstrip())

    return smiles_list

# caluculate the number of parameters
def tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


class EarlyStopping:
    def __init__(self, patience=10, ckpt_dir=None):
        '''引数: 最小値の非更新数カウンタ、表示設定、モデル格納path'''

        self.patience = patience    #設定ストップカウンタ
        self.counter = 0            #現在のカウンタ値
        self.best_score = None      #ベストスコア
        self.early_stop = False     #ストップフラグ
        self.val_loss_min = np.Inf   #前回のベストスコア記憶用
        self.path = ckpt_dir         #ベストモデル格納path

    def __call__(self, val_loss, step, optimizer, cur_loss, model):
        '''
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        '''
        score = -val_loss

        if self.best_score is None:  #1Epoch目の処理
            self.best_score = score   #1Epoch目はそのままベストスコアとして記録する
            self.checkpoint(val_loss, step, optimizer, cur_loss, model)  #記録後にモデルを保存してスコア表示する
        elif score < self.best_score:  # ベストスコアを更新できなかった場合
            self.counter += 1   #ストップカウンタを+1
            print(f'Validation loss increased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.checkpoint(val_loss, step, optimizer, cur_loss, model)
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  #現在のカウンタを表示する 
            if self.counter >= self.patience:  #設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  #ベストスコアを更新した場合
            self.best_score = score  #ベストスコアを上書き
            print(f'Validation loss decreased! ({self.val_loss_min:.6f} --> {val_loss:.6f}) Saving model ...')
            self.checkpoint(val_loss, step, optimizer, cur_loss, model)  #モデルを保存してスコア表示
            self.counter = 0  #ストップカウンタリセット

    def checkpoint(self, val_loss, step, optimizer, cur_loss, model):
        torch.save({'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': cur_loss,}, f'{self.path}/ckpt_{step+1}.pth')
        self.val_loss_min = val_loss  #その時のlossを記録する

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0 # latest value
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

# def accuracy(output, target, batch_size, v=None):
#     '''
#     Computes the accuracy of top1 prediction
    
#     output: (seq_length*batch_size, num_tokens)
#     target: (seq_length*batch_size)
#     '''
    
#     pad_mask = (target != v['<pad>']) # padはFalse, それ以外はTrue
#     true_pos = torch.nonzero(pad_mask).squeeze().tolist()
#     out_extracted = output[true_pos]
#     t_extracted = target[true_pos]
#     _, pred = out_extracted.topk(1, 1, True, True) # arg of topk: (k, dim=1, largest=True, sorted=True)
#     pred = pred.t() # (seq*batch, maxk) -> (maxk, seq*batch)
#     correct = pred.eq(t_extracted.reshape(1, -1).expand_as(pred)) # target:(seq*batch, 1) -> (1, seq*batch) -> (maxk, seq*batch)
#     # Tensor.eq: compute element-wise equality, correct: bool matrix
#     correct_rate = (correct[0].float().sum(0, keepdim=True)) / len(t_extracted)
    
#     # compute accuracy per whole molecule
#     target = target.reshape(-1, batch_size)
#     output = output.reshape(-1, batch_size, v.__len__())
#     _, pred = output.topk(10, 2, True, True)
#     top1, top5, top10 = pred[:, :, 0], pred[:, :, 0:4], pred[:, :, 0:9]
#     pred_list = [top1, top5, top10]
#     perfect_acc_list = []
#     EOS_token = v['<eos>']
#     for pred in pred_list:
#         correct_cum = 0
#         for i in range(batch_size):
#             t = target[:, i].tolist()
#             eos_idx = t.index(EOS_token)
#             t = t[0:eos_idx]
#             p = pred[:, i].tolist()
#             p = p[0:len(t)]
#             if t == p:
#                 correct_cum += 1
#         perfect_acc_list.append(correct_cum / batch_size)
#     return correct_rate.item(), perfect_acc_list

def accuracy(output, target, batch_size, v=None):
    '''
    Computes the accuracy of top1 prediction
    
    output: (seq_length*batch_size, num_tokens)
    target: (seq_length*batch_size)
    '''
    
    pad_mask = (target != v['<pad>']) # padはFalse, それ以外はTrue
    true_pos = torch.nonzero(pad_mask).squeeze().tolist()
    out_extracted = output[true_pos]
    t_extracted = target[true_pos]
    _, pred = out_extracted.topk(1, 1, True, True) # arg of topk: (k, dim=1, largest=True, sorted=True)
    pred = pred.t() # (seq*batch, maxk) -> (maxk, seq*batch)
    correct = pred.eq(t_extracted.reshape(1, -1).expand_as(pred)) # target:(seq*batch, 1) -> (1, seq*batch) -> (maxk, seq*batch)
    # Tensor.eq: compute element-wise equality, correct: bool matrix
    correct_rate = (correct[0].float().sum(0, keepdim=True)) / len(t_extracted)
    
    # compute accuracy per whole molecule
    target = target.reshape(-1, batch_size)
    output = output.reshape(-1, batch_size, v.__len__())
    _, pred = output.topk(1, 2, True, True)
    pred = pred.squeeze() # (seq, batch) -> (batch, seq)
    correct_cum = 0
    EOS_token = v['<eos>']
    for i in range(batch_size):
        t = target[:, i].tolist()
        eos_idx = t.index(EOS_token)
        t = t[0:eos_idx]
        p = pred[:, i].tolist()
        p = p[0:len(t)]
        if t == p:
            correct_cum += 1
    perfect_acc = correct_cum / batch_size
    return correct_rate.item(), perfect_acc

def calc_topk_perfect_acc(x, target, batch_size, EOS):
    '''
    x: predicted tensor of shape (seq, batch, k)
    target: (seq, batch)
    '''
    correct_cum = 0
    if x.dim() < 3:
        x = x.unsqueeze(-1)
    for i in range(batch_size):
        t = target[:, i].tolist()
        eos_idx = t.index(EOS)
        t = t[0:eos_idx]
        for j in range(x.size(2)):
            p = x[:, i, j].tolist()
            p = p[0:len(t)]
            if t == p:
                correct_cum += 1
                break
    return correct_cum / batch_size
    

def MW_checker(mol, threshold:int = 500):
    MW = Descriptors.ExactMolWt(mol)
    if MW > threshold:
        return False
    else:
        return True

def is_empty(li):
    return all(not sublist for sublist in li)

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    

# 例えばimport utils とした場合、そのutils.__name__ にはモジュール名（ファイル名）が格納される
# このファイルをimportで呼び出した場合、print(utils.__name__) の出力結果は'utils'
# ただし、importではなくコマンドラインで直接実行された場合は__name__ に __main__ が格納される
# よって、以下はimportされたときには実行されず、コマンドラインで実行されたときにだけ動く
if __name__ == '__main__':
    smiles_list = read_smilesset('Data/input/250k_rndm_zinc_drugs_clean.smi')
    vocab = []
    for smiles in tqdm(smiles_list):
        p = parse_smiles(smiles)
        vocab.extend(p)

    vocab = list(set(vocab))
    vocab.sort()
    print(vocab)
    