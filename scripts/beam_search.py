import os
import operator
import itertools
import re
import json
import hydra
from tqdm.auto import tqdm
from config.config import cs
from omegaconf import DictConfig

import rdkit.Chem as Chem
from rdkit.Chem import AllChem

import torch
import torchtext.vocab.vocab as Vocab
import torch.nn.functional as F

from Model.Transformer.model import Transformer
from scripts.preprocess import make_counter ,make_transforms
from Utils.utils import smi_tokenizer
from Model.GCN import network
from Model.GCN.utils import template_prediction, check_templates

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./data/label_template.json') as f:
    r_dict = json.load(f)

class BeamSearchNode(object):
    def __init__(self, previousNode, decoder_input, logProb, length):
        self.prevNode = previousNode
        self.dec_in = decoder_input
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=0.6):
        return self.logp / (((5 + self.leng) / (5 + 1)) ** alpha)
    
def check_templates(indices, input_smi):
    matched_indices = []
    input_smi = input_smi.replace(' ','')
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
    return matched_indices # ['[0]', '[123]', ... '[742]']

def beam_decode(v:Vocab, model=None, input_tokens=None, template_idx=None,
                device=None, inf_max_len=None, beam_width=10, nbest=5, Temp=None,
                beam_templates:list=None):
    
    SOS_token = v['<bos>']
    EOS_token = v['<eos>']
    if template_idx is not None:
        template_idx = re.sub(r'\D', '', template_idx)
        if template_idx not in beam_templates:
            beam_width = 5
            nbest = 1
    
    # A batch of one input for Encoder
    encoder_input = input_tokens

    # Generate encoded features
    with torch.no_grad():
        encoder_input = encoder_input.unsqueeze(-1) # (seq, 1), batch_size=1
        encoder_output, memory_pad_mask = model.encode(encoder_input, src_pad_mask=True) # encoder_output.shape: (seq, 1, d_model)

    # Start with the start of the sentence token
    decoder_input = torch.tensor([[SOS_token]]) # (1,1)

    # Starting node
    counter = itertools.count()
    
    node = BeamSearchNode(previousNode=None,
                          decoder_input=decoder_input,
                          logProb=0, length=0)
    
    with torch.no_grad():
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
        logits = model.decode(memory=encoder_output, tgt=decoder_input.permute(1, 0).to(device), tgt_mask=tgt_mask, memory_pad_mask=memory_pad_mask)
        logits = logits.permute(1, 0, 2) # logits: (seq, 1, vocab) -> (1, seq, vocab), batch=1
        decoder_output = torch.log_softmax(logits[:, -1, :]/Temp, dim=1).to('cpu') # (1, vocab)
    
    tmp_beam_width = min(beam_width, decoder_output.size(1))
    log_prob, indices = torch.topk(decoder_output, tmp_beam_width) # (tmp_beam_with,)
    nextnodes = []
    for new_k in range(tmp_beam_width):
        decoded_t = indices[0][new_k].view(1, -1)
        log_p = log_prob[0][new_k].item()
        next_decoder_input = torch.cat([node.dec_in, decoded_t],dim=1) # dec_in:(1, seq)
        nn = BeamSearchNode(previousNode=node,
                                decoder_input=next_decoder_input,
                                logProb=node.logp + log_p,
                                length=node.leng + 1)
        score = -nn.eval()
        count = next(counter)
        nextnodes.append((score, count, nn))
    
    # start beam search
    for i in range(inf_max_len - 1):
        # fetch the best node
        if i == 0:
            current_nodes = sorted(nextnodes)[:tmp_beam_width]
        else:
            current_nodes = sorted(nextnodes)[:beam_width]
            
        nextnodes=[]
        # current_nodes = [(score, count, node), (score, count, node)...], shape:(beam_width,)
        scores, counts, nodes, decoder_inputs = [], [], [], []
        
        for score, count, node in current_nodes:
            if node.dec_in[0][-1].item() == EOS_token:
                nextnodes.append((score, count, node))
            else:
                scores.append(score)
                counts.append(count)
                nodes.append(node)
                decoder_inputs.append(node.dec_in)
        if not bool(decoder_inputs):
            break
        
        decoder_inputs = torch.vstack(decoder_inputs) # (batch=beam, seq)

        # adjust batch_size
        enc_out = encoder_output.repeat(1, decoder_inputs.size(0), 1)
        mask = memory_pad_mask.repeat(decoder_inputs.size(0), 1)
        
        with torch.no_grad():
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_inputs.size(1)).to(device)
            logits = model.decode(memory=enc_out, tgt=decoder_inputs.permute(1, 0).to(device), tgt_mask=tgt_mask, memory_pad_mask=mask)
            logits = logits.permute(1, 0, 2) # logits: (seq, batch, vocab) -> (batch, seq, vocab)
            decoder_output = torch.log_softmax(logits[:, -1, :]/Temp, dim=1).to('cpu') # extract log_softmax of last token
            # decoder_output.shape = (batch, vocab)
        
        for beam, score in enumerate(scores):
            for token in range(EOS_token, decoder_output.size(-1)): # remove unk, pad, bosは最初から捨てる
                decoded_t = torch.tensor([[token]])
                log_p = decoder_output[beam, token].item()
                next_decoder_input = torch.cat([nodes[beam].dec_in, decoded_t],dim=1)
                node = BeamSearchNode(previousNode=nodes[beam],
                                    decoder_input=next_decoder_input,
                                    logProb=nodes[beam].logp + log_p,
                                    length=nodes[beam].leng + 1)
                score = -node.eval()
                count = next(counter)
                nextnodes.append((score, count, node))
            
    outputs = []
    for score, _, n in sorted(nextnodes, key=operator.itemgetter(0))[:nbest]:
        # endnodes = [(score, node), (score, node)...] 
        output = n.dec_in.squeeze(0).tolist()[1:-1] # remove bos and eos
        output = v.lookup_tokens(output)
        output = ''.join(output)
        outputs.append(output)
    return outputs

def greedy_translate(v:Vocab, model=None, input_tokens=None, device=None, inf_max_len=None):
    '''
    in:
    input_tokens: (seq, batch)
    
    out:
    outputs: list of SMILES(str).
    '''
    
    SOS_token = v['<bos>']
    EOS_token = v['<eos>']

    # A batch of one input for Encoder
    encoder_input = input_tokens.permute(1, 0) # (batch,seq) -> (seq, batch)

    # Generate encoded features
    with torch.no_grad():
        enc_out, memory_pad_mask = model.encode(encoder_input, src_pad_mask=True) # encoder_output.shape: (seq, 1, d_model)

        # Start with the SOS token
        dec_inp = torch.tensor([[SOS_token]]).expand(1, encoder_input.size(1)).to(device) # (1, batch)
        EOS_dic = {i:False for i in range(encoder_input.size(1))}

        for i in range(inf_max_len - 1):
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(dec_inp.size(0)).to(device)
            logits = model.decode(memory=enc_out, tgt=dec_inp, tgt_mask=tgt_mask, memory_pad_mask=memory_pad_mask)
            dec_out = F.softmax(logits[-1, :, :], dim=1) # extract softmax of last token, (batch, vocab)
            next_items = dec_out.topk(1)[1].permute(1, 0) # (seq, batch) -> (batch, seq)
            EOS_indices = (next_items == EOS_token)
            # update EOS_dic
            for j, EOS in enumerate(EOS_indices[0]):
                if EOS:
                    EOS_dic[j] = True
            
            dec_inp = torch.cat([dec_inp, next_items], dim=0)
            if sum(list(EOS_dic.values())) == encoder_input.size(1):
                break
        out = dec_inp.permute(1, 0).to('cpu') # (seq, batch) -> (batch, seq)
        outputs = []
        for i in range(out.size(0)):
            out_tokens = v.lookup_tokens(out[i].tolist())
            try:
                eos_idx = out_tokens.index('<eos>')
                out_tokens = out_tokens[1:eos_idx]
                outputs.append(''.join(out_tokens))
            except ValueError:
                continue
        
    return outputs

def translate(cfg:DictConfig):
    print('Loading...')
    # make transforms and vocabulary
    src_train_path = hydra.utils.get_original_cwd()+cfg['translate']['src_train']
    tgt_train_path = hydra.utils.get_original_cwd()+cfg['translate']['tgt_train']
    src_valid_path = hydra.utils.get_original_cwd()+cfg['translate']['src_valid']
    tgt_valid_path = hydra.utils.get_original_cwd()+cfg['translate']['tgt_valid']
    data_dict = make_counter(src_train_path=src_train_path,
                             tgt_train_path=tgt_train_path,
                             src_valid_path=src_valid_path,
                             tgt_valid_path=tgt_valid_path
                             )
    src_transforms, _, v = make_transforms(data_dict=data_dict, make_vocab=True, vocab_load_path=None)
    
    # load model
    d_model = cfg['model']['dim_model']
    num_encoder_layers = cfg['model']['num_encoder_layers']
    num_decoder_layers = cfg['model']['num_decoder_layers']
    nhead = cfg['model']['nhead']
    dropout = cfg['model']['dropout']
    dim_ff = cfg['model']['dim_ff'] 
    model = Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                        dim_feedforward=dim_ff,vocab=v, dropout=dropout, device=device).to(device)
    ckpt = torch.load(hydra.utils.get_original_cwd() + cfg['model']['ckpt'], map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # make dataset
    src = []
    src_test_path = hydra.utils.get_original_cwd() + cfg['translate']['src_test_path']
    with open(src_test_path,'r') as f:
        for line in f:
            src.append(line.rstrip())
    
    dim_GCN = cfg['GCN_train']['dim']
    n_conv_hidden = cfg['GCN_train']['n_conv_hidden']
    n_mlp_hidden = cfg['GCN_train']['n_mlp_hidden']
    GCN_model = network.MolecularGCN(dim = dim_GCN,
                                 n_conv_hidden = n_conv_hidden,
                                 n_mlp_hidden = n_mlp_hidden,
                                 dropout = dropout).to(device)
    GCN_ckpt = hydra.utils.get_original_cwd() + cfg['translate']['GCN_ckpt']
    GCN_model.load_state_dict(torch.load(GCN_ckpt))
    GCN_model.eval()
    
    out_dir = cfg['translate']['out_dir']
    beam_width = cfg['translate']['beam_size']
    nbest = cfg['translate']['nbest']
    inf_max_len = cfg['translate']['inf_max_len']
    GCN_num_sampling = cfg['translate']['GCN_num_sampling']
    with open(hydra.utils.get_original_cwd() + cfg['translate']['annotated_templates'], 'r') as f:
        beam_templates = f.read().splitlines()
        f.close()
    print(f'The number of sampling for GCN: {GCN_num_sampling}')
    print('Start translation...')
    rsmis =[]
    for input_smi in tqdm(src):
        input_smi = input_smi.replace(' ', '')
        indices = template_prediction(GCN_model=GCN_model, input_smi=input_smi,
                                      num_sampling=GCN_num_sampling, GCN_device=device)
        matched_indices = check_templates(indices, input_smi)
        print(f"{len(matched_indices)} reaction templates are matched for '{input_smi}'.")
        with torch.no_grad():
            for i in matched_indices:
                input_conditional = smi_tokenizer(i + input_smi).split(' ')
                input_tokens = src_transforms(input_conditional).to(device)
                outputs = beam_decode(v=v, model=model, input_tokens=input_tokens, template_idx=i,
                                    device=device, inf_max_len=inf_max_len, beam_width=beam_width, nbest=nbest,
                                    Temp=1, beam_templates=beam_templates)
                for output in outputs:
                    output = smi_tokenizer(output)
                    rsmis.append(i + ' ' + smi_tokenizer(input_smi) + ' >> ' + output)

# set output file name
    os.makedirs(hydra.utils.get_original_cwd() + out_dir, exist_ok=True)
    with open(hydra.utils.get_original_cwd() + f'{out_dir}/out_beam{beam_width}_best{nbest}2.txt','w') as f:
        for rsmi in rsmis:
            f.write(rsmi + '\n')
        f.close()

@hydra.main(config_path=None, config_name='config', version_base=None)
def main(cfg: DictConfig):
    translate(cfg)

if __name__ == '__main__':
    main()