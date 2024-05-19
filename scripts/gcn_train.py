import os
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from Model.GCN import mol2graph
from Model.GCN.callbacks import EarlyStopping
from Model.GCN.network import MolecularGCN
from Model.GCN.utils import get_data

import hydra
import datetime
from config.config import cs
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
date = datetime.datetime.now().strftime('%Y%m%d')

def train(model, optimizer, loader):
    model.train()
    loss_all = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        output = model.forward(data.x, data.edge_index, data.batch).squeeze(1)
        loss =  F.cross_entropy(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
        
    return loss_all / len(loader)


def eval(model, loader, ks=None):
    model.eval()
    score_list = []
    with torch.no_grad():
        loss_all = 0
        for data in loader:
            data = data.to(device)
            output = model.forward(data.x, data.edge_index, data.batch) # output.shape = (batch_size, vocab_size)
            loss = F.cross_entropy(output, data.y)
            loss_all += loss.item() * data.num_graphs
            if ks is not None:
                for k in ks:
                    score_list.append(topk_accuracy(data, output, k))
    return  loss_all/len(loader), score_list

def topk_accuracy(data, output, k: int):
    _, pred = output.topk(k, 1, True, True) # (k, dim=1, largest=True, sorted=True)
    pred = pred.t() # (batch, maxk) -> (maxk, batch)
    correct = pred.eq(data.y.unsqueeze(0).expand_as(pred)) # target:(batch,) -> (1, batch) -> (maxk, batch)
    # Tensor.eq: compute element-wise equality, correct: bool matrix
    score = correct.float().sum() / len(data)
    score = score.detach().item()
    return score


@hydra.main(config_path=None, config_name='config', version_base=None)
def main(cfg: DictConfig):
    print('Loading data...')
    train_path = cfg['GCN_train']['train']
    valid_path = cfg['GCN_train']['valid']
    test_path = cfg['GCN_train']['test']
    batch_size = cfg['GCN_train']['batch_size']
    dim = cfg['GCN_train']['dim']
    n_conv_hidden = cfg['GCN_train']['n_conv_hidden']
    n_mlp_hidden = cfg['GCN_train']['n_mlp_hidden']
    dropout = cfg['GCN_train']['dropout']
    lr = cfg['GCN_train']['lr']
    epochs = cfg['GCN_train']['epochs']
    patience = cfg['GCN_train']['patience']
    save_path = cfg['GCN_train']['save_path']
    ks = [1, 3, 5, 10]
    
    mols_train, y_train = get_data(hydra.utils.get_original_cwd() + train_path)
    mols_valid, y_valid = get_data(hydra.utils.get_original_cwd() + valid_path)
    
    print('-'*100)
    print('Training: ', mols_train.shape)
    print('Validation: ', mols_valid.shape)
    print('-'*100)

    labels = y_train.tolist() + y_valid.tolist()
    
    # Mol to Graph
    print('Converting mol to graph...')
    X_train = [mol2graph.mol2vec(m) for m in tqdm(mols_train.tolist())]
    for i, data in enumerate(X_train):
        data.y = torch.LongTensor([y_train[i]]).to(device)
    X_valid = [mol2graph.mol2vec(m) for m in tqdm(mols_valid.tolist())]
    for i, data in enumerate(X_valid):
        data.y = torch.LongTensor([y_valid[i]]).to(device)
    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(X_valid, batch_size=batch_size, shuffle=True, drop_last=True)
    print('completed.')
    print('-'*100)
    
    num = 1
    while True:
        ckpt_dir = hydra.utils.get_original_cwd()+f'{save_path}/checkpoints_{date}_{num}'
        try:
            if any(os.scandir(ckpt_dir)):
                num +=1
                continue
            else:
                break
        except:
            os.makedirs(ckpt_dir, exist_ok=True)
        break
    train_path = cfg['GCN_train']['train']
    valid_path = cfg['GCN_train']['valid']
    test_path = cfg['GCN_train']['test']
    batch_size = cfg['GCN_train']['batch_size']
    dim = cfg['GCN_train']['dim']
    n_conv_hidden = cfg['GCN_train']['n_conv_hidden']
    n_mlp_hidden = cfg['GCN_train']['n_mlp_hidden']
    dropout = cfg['GCN_train']['dropout']
    lr = cfg['GCN_train']['lr']
    epochs = cfg['GCN_train']['epochs']
    patience = cfg['GCN_train']['patience']

    # Model instance construction
    print('Model instance construction')
    model = MolecularGCN(
        dim = dim,
        n_conv_hidden = n_conv_hidden,
        n_mlp_hidden = n_mlp_hidden,
        dropout = dropout
        ).to(device)
    print(model)
    print('-'*100)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    earlystopping = EarlyStopping(patience=patience, path=ckpt_dir + '/ckpt.pth', verbose=True)
    for epoch in range(1, epochs+1):
        # training
        train_loss = train(model, optimizer, train_loader)

        # performance evaluation
        loss_train, _ = eval(model, train_loader)
        loss_valid, score_list = eval(model, valid_loader, ks=ks)
        top1acc = score_list[0]
        top3acc = score_list[1]
        top5acc = score_list[2]
        top10acc = score_list[3]
        
        print(f'Epoch: {epoch}/{epochs}, loss_train: {loss_train:.5}, loss_valid: {loss_valid:.5}')
        print(f'top k accuracy: top1={top1acc:.2}, top3={top3acc:.2}, top5={top5acc:.2}, top10={top10acc:.2}')
        # early stopping detection
        earlystopping(loss_valid, model)
        if earlystopping.early_stop:
            print('Early Stopping!')
            print('-'*100)
            break

if __name__ == '__main__':
    main()