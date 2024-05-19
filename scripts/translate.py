import operator
import torch
import torchtext.vocab.vocab as Vocab
import rdkit.Chem as Chem
import hydra
from config.config import cs
from omegaconf import DictConfig

from Model.Transformer.model import Transformer
from scripts.preprocess import make_counter ,make_transforms

import itertools
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BeamSearchNode(object):
    def __init__(self, previousNode, decoder_input, logProb, length):
        self.prevNode = previousNode
        self.dec_in = decoder_input.to('cpu')
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=0.6):
        return self.logp / (((5 + self.leng) / (5 + 1)) ** alpha)


def beam_decode(cfg: DictConfig, v:Vocab, model=None, input_tokens=None, Temp=1):
    global beam_width, nbest
    SOS_token = v['<bos>']
    EOS_token = v['<eos>']
    beam_width = cfg['translate']['beam_size']
    nbest = cfg['translate']['nbest']
    inf_max_len = cfg['translate']['inf_max_len']
    
    # A batch of one input for Encoder
    encoder_input = input_tokens

    # Generate encoded features
    with torch.no_grad():
        encoder_input = encoder_input.unsqueeze(-1) # (seq, 1), batch_size=1
        encoder_output, memory_pad_mask = model.encode(encoder_input, src_pad_mask=True) # encoder_output.shape: (seq, 1, d_model)

    # Start with the start of the sentence token
    decoder_input = torch.tensor([[SOS_token]]).to(device) # (1,1)

    # Starting node
    counter = itertools.count()
    
    node = BeamSearchNode(previousNode=None,
                          decoder_input=decoder_input,
                          logProb=0, length=0)
    
    with torch.no_grad():
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(decoder_input.size(1)).to(device)
        logits = model.decode(memory=encoder_output, tgt=decoder_input.permute(1, 0), tgt_mask=tgt_mask, memory_pad_mask=memory_pad_mask)
        logits = logits.permute(1, 0, 2) # logits: (seq, 1, vocab) -> (1, seq, vocab), batch=1になってる
        decoder_output = torch.log_softmax(logits[:, -1, :]/Temp, dim=1) # 最後のseqだけ取り出してlog_softmax, (1, vocab)
    
    
    tmp_beam_width = min(beam_width, decoder_output.size(1))
    log_prob, indexes = torch.topk(decoder_output, tmp_beam_width) # (tmp_beam_with,)
    nextnodes = []
    for new_k in range(tmp_beam_width):
        decoded_t = indexes[0][new_k].view(1, -1).to('cpu') # indexを取得, shape: (1,1)
        log_p = log_prob[0][new_k].item() # logpを取得
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
            decoder_output = torch.log_softmax(logits[:, -1, :]/Temp, dim=1) # extract log_softmax of last token
            # decoder_output.shape = (batch, vocab)
            
        for beam, score in enumerate(scores):
            for token in range(EOS_token, decoder_output.size(-1)): # indexを取得, unk, pad, bosは最初から捨てる
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
        # endnodes = [(score, node), (score, node)...] なのでitemgetter(0)でscoreをkeyに指定している
        output = n.dec_in.squeeze(0).tolist()[1:-1] # bosとeos削除
        output = v.lookup_tokens(output)
        output = ' '.join(output)
        outputs.append(output)

    return outputs


def translation(cfg:DictConfig):
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
    
    # make dataset
    src = []
    src_test_path = hydra.utils.get_original_cwd() + cfg['translate']['src_test_path']
    with open(src_test_path,'r') as f:
        for line in f:
            src.append(line.rstrip().split(' '))
    src = src_transforms(src).to(device)

    rsmis =[]
    for i, input_tokens in enumerate(src):
        outputs = beam_decode(cfg=cfg, v=v, model=model, input_tokens=input_tokens)
        input_tokens = input_tokens.tolist()
        input_smi = input_tokens[0:input_tokens.index(v['<pad>'])]
        input_smi = v.lookup_tokens(input_smi)
        input_smi = ' '.join(input_smi)
        for output in outputs:
            rsmis.append(input_smi + ' >> ' + output)
        
    out_dir = cfg['translate']['out_dir']
    filename = cfg['translate']['filename']
    
# set output file name
    os.makedirs(hydra.utils.get_original_cwd() + out_dir, exist_ok=True)
    with open(hydra.utils.get_original_cwd() + f'{out_dir}/out_beam{beam_width}_best{nbest}_file_{filename}.txt','w') as f:
        for rsmi in rsmis:
            f.write(rsmi + '\n')
        f.close()

@hydra.main(config_path=None, config_name='config', version_base=None)
def main(cfg: DictConfig):
    translation(cfg)

if __name__ == '__main__':
    main()