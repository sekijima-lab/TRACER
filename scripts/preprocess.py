import warnings
import hydra
import os
from config.config import cs
from omegaconf import DictConfig
warnings.filterwarnings('ignore')
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
import torchtext.transforms as T

class smi_Dataset(Dataset):
    def __init__(self, src, tgt):
        super().__init__()
        self.src = src
        self.tgt = tgt
        
    def __getitem__(self, i):
        src = self.src[i]
        tgt = self.tgt[i]
        return src, tgt
    
    def __len__(self):
        return len(self.src)

def make_smi_list(path, counter):
    smi_list = []
    max_length = 0
    with open(path,'r') as f:
        for line in f:
            smi_list.append(line.rstrip().split(' '))
        for i in smi_list:
            counter.update(i)
            if len(i) > max_length:
                max_length = len(i)
    return smi_list, max_length
    

def make_counter(src_train_path, tgt_train_path, src_valid_path, tgt_valid_path) -> dict:
    src_counter = Counter()
    tgt_counter = Counter()
    src_train, max_src_train = make_smi_list(src_train_path, src_counter)
    tgt_train, max_tgt_train = make_smi_list(tgt_train_path, tgt_counter)
    src_valid, max_src_valid = make_smi_list(src_valid_path, src_counter)
    tgt_valid, max_tgt_valid = make_smi_list(tgt_valid_path, tgt_counter)
    
    src_max_length = max([max_src_train, max_src_valid])
    tgt_max_length = max([max_tgt_train, max_tgt_valid])
    tgt_max_length = tgt_max_length+2 # bosとeosの分を加算
    
    datasets = []
    datasets.append(src_train)
    datasets.append(tgt_train)
    datasets.append(src_valid)
    datasets.append(tgt_valid)
    
    return {'src_counter': src_counter, 'tgt_counter': tgt_counter,
            'src_max_len': src_max_length, 'tgt_max_len': tgt_max_length, 'datasets': datasets}

def make_transforms(data_dict, make_vocab: bool = False, vocab_load_path=None):
    if make_vocab == False and vocab_load_path is None:
        raise ValueError('The make_transforms function is not being passed the vocab_load_path.')
    if make_vocab:
        counter = data_dict['src_counter'] + data_dict['tgt_counter']
        v = vocab(counter, min_freq=5, specials=(['<unk>', '<pad>', '<bos>', '<eos>']))
        v.set_default_index(v['<unk>'])
    else:
        v = torch.load(vocab_load_path)
    
    src_transforms = T.Sequential(
        T.VocabTransform(v),
        T.ToTensor(padding_value=v['<pad>']),
        T.PadTransform(max_length=data_dict['src_max_len'], pad_value=v['<pad>']) # srcはbosとeosが不要
        )
    
    tgt_transforms = T.Sequential(
        T.VocabTransform(v),
        T.AddToken(token=v['<bos>'], begin=True),
        T.AddToken(token=v['<eos>'], begin=False),
        T.ToTensor(padding_value=v['<pad>']),
        T.PadTransform(max_length=data_dict['tgt_max_len'],pad_value=v['<pad>'])
        )
    
    return src_transforms, tgt_transforms, v
    

def make_dataloader(datasets, src_transforms, tgt_transforms, batch_size):
    '''
    datasets: output of make_counter()
    transforms: output of make_vocab()
    '''
    
    src_train = datasets[0]
    tgt_train = datasets[1]
    src_valid = datasets[2]
    tgt_valid = datasets[3]
    
    src_train, src_valid = src_transforms(src_train), src_transforms(src_valid)
    tgt_train, tgt_valid = tgt_transforms(tgt_train), tgt_transforms(tgt_valid)
    
    train_dataset = smi_Dataset(src=src_train, tgt=tgt_train)
    valid_dataset = smi_Dataset(src=src_valid, tgt=tgt_valid)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True
                                  )
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  drop_last=False,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=True
                                  )

    return train_dataloader, valid_dataloader
    

@hydra.main(config_path=None, config_name='config', version_base=None)
def main(cfg: DictConfig):
    # Loading data
    print('Saving vocabulary...')
    src_train_path = hydra.utils.get_original_cwd()+cfg['prep']['src_train']
    tgt_train_path = hydra.utils.get_original_cwd()+cfg['prep']['tgt_train']
    src_valid_path = hydra.utils.get_original_cwd()+cfg['prep']['src_valid']
    tgt_valid_path = hydra.utils.get_original_cwd()+cfg['prep']['tgt_valid']
    
    data_dict= make_counter(src_train_path=src_valid_path,
                            tgt_train_path=tgt_train_path,
                            src_valid_path=src_train_path,
                            tgt_valid_path=tgt_valid_path)
    
    _, _, v = make_transforms(data_dict=data_dict, make_vocab=True, vocab_load_path=None)
    torch.save(v, hydra.utils.get_original_cwd()+'/vocab.pth')
    print('done.')

if __name__ == '__main__':
    main()




