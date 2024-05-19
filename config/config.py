import math

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

@dataclass
class PreProcess:
    augm_size: int = 1
    src_train: str = '/data/USPTO/src_train.txt'
    tgt_train: str = '/data/USPTO/tgt_train.txt'
    src_valid: str = '/data/USPTO/src_valid.txt'
    tgt_valid: str = '/data/USPTO/tgt_valid.txt'
    batch_size: int = 256
    
@dataclass
class ModelConfig:
    dim_model: int = 512
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    nhead: int = 8
    dropout: float = 0.1
    dim_ff: int = 2048
    ckpt:str = '/ckpts/Transformer/ckpt_conditional.pth'

@dataclass
class TrainConfig:
    src_train: str = '/data/USPTO/src_train.txt'
    tgt_train: str = '/data/USPTO/tgt_train.txt'
    src_valid: str = '/data/USPTO/src_valid.txt'
    tgt_valid: str = '/data/USPTO/tgt_valid.txt'
    batch_size: int = 128
    label_smoothing: float = 0.0
    lr: float = 0.001
    betas: tuple = (0.9, 0.998)
    step_num: int = 500000 # set training steps
    patience: int = 10
    log_interval : int = 100
    val_interval: int = 1000
    save_interval: int = 10000

@dataclass
class TranslateConfig:
    src_train: str = '/data/USPTO/src_train.txt'
    tgt_train: str = '/data/USPTO/tgt_train.txt'
    src_valid: str = '/data/USPTO/src_valid.txt'
    tgt_valid: str = '/data/USPTO/tgt_valid.txt'
    GCN_ckpt: str = '/ckpts/GCN/GCN.pth'
    out_dir: str = '/translation'
    src_test_path: str = '/data/input/test.txt'
    annotated_templates: str = '/data/beamsearch_template_list.txt'
    filename: str = 'test'
    GCN_num_sampling: int = 10
    inf_max_len: int = 256
    nbest: int = 10
    beam_size: int = 10

@dataclass
class GCN_TrainConfig:
    train: str = '/data/USPTO/src_train.txt'
    valid: str = '/data/USPTO/src_valid.txt'
    test: str = '/data/USPTO/src_test.txt'
    batch_size: int = 256
    dim: int = 256
    n_conv_hidden: int = 1
    n_mlp_hidden: int = 3
    dropout: float = 0.1
    lr: float =  0.0004
    epochs: int = 100
    patience: int = 5
    save_path: str = '/ckpts/GCN'
    


@dataclass
class MCTSConfig:
    src_train: str = '/data/USPTO/src_train.txt'
    tgt_train: str = '/data/USPTO/tgt_train.txt'
    src_valid: str = '/data/USPTO/src_valid.txt'
    tgt_valid: str = '/data/USPTO/tgt_valid.txt'
    n_step: int = 200
    max_depth: int = 10
    in_smiles_file: str = '/data/input/init_smiles_drd2.txt'
    out_dir: str = '/mcts_out'
    ucb_c: float = 1/math.sqrt(2)
    reward_name: str = 'DRD2'  # 'DRD2' or 'QED'
    ckpt_Transformer: str = '/ckpts/Transformer/ckpt_conditional.pth'
    ckpt_GCN: str = '/ckpts/GCN/GCN.pth'
    beam_width:int = 10
    nbest:int = 10
    exp_num_sampling:int = 10
    rollout_depth:int = 2
    roll_num_sampling:int = 5

@dataclass
class Config:
    prep: PreProcess = PreProcess()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    translate: TranslateConfig = TranslateConfig()
    GCN_train: GCN_TrainConfig = GCN_TrainConfig()
    mcts: MCTSConfig = MCTSConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
