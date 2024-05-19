import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import time
import math
import hydra
from config.config import cs
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from Model.Transformer.model import TransformerLR, Transformer
from scripts.preprocess import make_counter, make_transforms, make_dataloader
from Utils.utils import tally_parameters, EarlyStopping, AverageMeter, accuracy, torch_fix_seed

import datetime
date = datetime.datetime.now().strftime('%Y%m%d')

torch_fix_seed()

def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num = 1
    while True:
        ckpt_dir = hydra.utils.get_original_cwd()+f'/ckpts/checkpoints_{date}_{num}'
        if os.path.isdir(ckpt_dir):
            num += 1
            continue
        else:
            os.makedirs(ckpt_dir, exist_ok=True)
            break
    print(f'{ckpt_dir} was created.')
    
    data_dict = make_counter(src_train_path=hydra.utils.get_original_cwd()+cfg['train']['src_train'],
                             tgt_train_path=hydra.utils.get_original_cwd()+cfg['train']['tgt_train'],
                             src_valid_path=hydra.utils.get_original_cwd()+cfg['train']['src_valid'],
                             tgt_valid_path=hydra.utils.get_original_cwd()+cfg['train']['tgt_valid']
                             )
    print('making dataloader...')
    src_transforms, tgt_transforms, v = make_transforms(data_dict=data_dict, make_vocab=True, vocab_load_path=None)
    train_dataloader, valid_dataloader = make_dataloader(datasets=data_dict['datasets'], src_transforms=src_transforms,
                                                         tgt_transforms=tgt_transforms,batch_size=cfg['train']['batch_size'])
    print('max length of src sentence:', data_dict['src_max_len'])
    d_model = cfg['model']['dim_model']
    nhead = cfg['model']['nhead']
    dropout = cfg['model']['dropout']
    dim_ff = cfg['model']['dim_ff']
    num_encoder_layers = cfg['model']['num_encoder_layers']
    num_decoder_layers = cfg['model']['num_decoder_layers']
    model = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                        dim_feedforward=dim_ff,vocab=v, dropout=dropout, device=device).to(device)
    cudnn.benchmark = True
    if device == 'cuda':
        model = torch.nn.DataParallel(model) # make parallel
        torch.backends.cudnn.benchmark = True
    
    # count the number of parameters.
    n_params, enc, dec = tally_parameters(model)
    print('encoder: %d' % enc)
    print('decoder: %d' % dec)
    print('* number of parameters: %d' % n_params)
    
    lr = cfg['train']['lr']
    betas = cfg['train']['betas']
    patience = cfg['train']['patience']
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    scheduler = TransformerLR(optimizer, warmup_epochs=8000)
    label_smoothing = cfg['train']['label_smoothing']
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                    reduction='none',
                                    ignore_index=v['<pad>']
                                    )
    earlystopping = EarlyStopping(patience=patience, ckpt_dir=ckpt_dir)
    
    step_num = cfg['train']['step_num']
    log_interval_step = cfg['train']['log_interval']
    valid_interval_steps = cfg['train']['val_interval']
    save_interval_steps = cfg['train']['save_interval']
    accum_count = 1
    
    valid_len = 0
    for _, d in enumerate(valid_dataloader):
        valid_len += len(d[0])
    
    step = 0
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(data_dict['tgt_max_len']-1).to(device)
    scaler = torch.cuda.amp.GradScaler()
    total_loss = 0
    accum_loss = 0
    model.train()
    start_time = time.time()
    print('start training...')
    while step < step_num:
        for i, data in enumerate(train_dataloader):
            src, tgt = data[0].to(device).permute(1, 0), data[1].to(device).permute(1, 0)
            tgt_input = tgt[:-1, :] # (seq, batch)
            tgt_output = tgt[1:, :] # shifted right
            with torch.amp.autocast('cuda'):
                outputs = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask,
                                src_pad_mask=True, tgt_pad_mask=True, memory_pad_mask=True) # out: (seq_length, batch_size, vocab_size)
                loss = (criterion(outputs.reshape(-1, v.__len__()), tgt_output.reshape(-1)).sum() / len(data[0])) / accum_count
            scaler.scale(loss).backward()
            accum_loss += loss.detach().item()
            if ((i + 1) % accum_count == 0) or ((i + 1) == len(train_dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                total_loss += accum_loss
                accum_loss = 0
            
            if (step + 1) % log_interval_step == 0:
                lr = scheduler.get_last_lr()[0]
                cur_loss = total_loss / log_interval_step
                ppl = math.exp(cur_loss)
                end_time = time.time()
                print(f'| step {step+1} | lr {lr:03.5f} | loss {cur_loss:5.5f} | ppl {ppl:8.5f} | time per {log_interval_step} step {end_time - start_time:3.1f}|')
                total_loss = 0
                start_time = time.time()
            
            # validation step
            if (step + 1) % valid_interval_steps == 0:
                model.eval()
                top1 = AverageMeter()
                perfect_acc_top1 = AverageMeter()
                eval_total_loss = 0.
                with torch.no_grad():
                    for val_i, val_data in enumerate(valid_dataloader):
                        src, tgt = val_data[0].to(device).permute(1, 0), val_data[1].to(device).permute(1, 0)
                        tgt_input = tgt[:-1, :]
                        tgt_output = tgt[1:, :]
                        outputs = model(src=src, tgt=tgt_input, tgt_mask=tgt_mask,
                                        src_pad_mask=True, tgt_pad_mask=True, memory_pad_mask=True)
                        tmp_eval_loss = criterion(outputs.reshape(-1, v.__len__()), tgt_output.reshape(-1)).sum() / len(val_data[0])
                        eval_total_loss += tmp_eval_loss.detach().item()
                        partial_top1, perfect_acc = accuracy(outputs.reshape(-1, v.__len__()), tgt_output.reshape(-1), batch_size=tgt_output.size(1), v=v)
                        top1.update(partial_top1, src.size(1))
                        perfect_acc_top1.update(perfect_acc, src.size(1))
                    eval_loss = eval_total_loss / (val_i + 1)
                    print(f'validation step {step+1} | validation loss {eval_loss:5.5f} | partial top1 accuracy {top1.avg:.3f} | perfect top1 accuracy {perfect_acc_top1.avg:.3f}')
                    if (step + 1) % save_interval_steps == 0:
                        earlystopping(val_loss=eval_loss, step=step, optimizer=optimizer, cur_loss=cur_loss, model=model)
                model.train()
                start_time = time.time()
            if earlystopping.early_stop:
                print('Early Stopping!')
                break
        if earlystopping.early_stop:
            break
            
@hydra.main(config_path=None, config_name='config', version_base=None)
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    main()