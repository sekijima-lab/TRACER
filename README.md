# TRACER: Molecular Optimization Using Conditional Transformer for Reaction-Aware Compound Exploration with Reinforcement Learning

This repository contains the source code of TRACER, a framework for molecular optimization with synthetic pathways. TRACER integrates a conditional Transformer model trained on chemical reactions with Monte Carlo Tree Search (MCTS) for efficient exploration of the chemical space. For more details, please refer to the [paper](https://www.nature.com/articles/s42004-025-01437-x).


## Installation

To set up the environment for running TRACER, follow these steps to create a conda environment with the necessary dependencies:

1. Clone this repository:
   ```
   git clone https://github.com/sekijima-lab/TRACER.git
   cd TRACER
   ```

2. Create a conda environment using the provided `env.yml`:
   ```
   conda env create -f env.yml
   ```

3. Activate the conda environment:
   ```
   conda activate tracer
   ```

## Setup Environment Variable

To ensure that the modules provided by TRACER can be imported, you need to add the paths to the PYTHONPATH environment variable. Please follow these steps:

At the top directory of the TRACER, run the following command to execute the `set_up.sh` script:

```
source set_up.sh
```

This command will set the `PYTHONPATH` environment variable to include the necessary directories.

Please note that you need to run this command every time you start a new terminal session.


## Download Model Parameters

Please download trained weights for the Transformer from [Figshare here](https://figshare.com/articles/software/Weights_of_conditional_unconditional_Transformer/25853551), and place the weights in the `ckpts/Transformer/` directory.

Then, the directory substructure is as follows:


```
.
├── ckpts/
│   ├── GCN/
│   │    └── GCN.pth
│   └── Transformer/
│        ├── ckpt_conditional.pth
│        └── ckpt_unconditional.pth
└── ...
```


## Configuration

TRACER uses Hydra for managing the configuration of experiments. 

You can modify the configuration file (`config/config.py`) to adjust the hyperparameters and settings for training and molecular generation.

## (optional) Training the Transformer and GCN Model

The weights used in the paper are provided at Figshare.

If you would like to train the model using other training datasets, please refer to the following procedure.

1. To train the Transformer model on chemical reactions, run `scripts/transformer_train.py`:
   ```
   python scripts/transformer_train.py
   ```

2. To train the Graph Convolutional Network (GCN) for predicting applicable reaction templates, run `scripts/gcn_train.py`:
   ```
   python scripts/gcn_train.py
   ```

The trained weights will be saved in the `ckpts` directory.

## Structural Optimization using MCTS

To generate optimized compounds using MCTS and the trained models, run `scripts/mcts.py`:
```
python scripts/mcts.py
```

The generated compounds and their synthesis routes will be saved in the `mcts_out` directory.


## Directory structure

```
.
├── README.md    
├── LICENSE
├── data/ 
│   ├── input/        # Input SMILES of starting materials of MCTS
│   ├── QSAR/         # Dataset for QSAR model training
│   └── USPTO/        # Curated dataset based on USPTO 1k TPL [1]
├── Model/               
│   ├── GCN/          # Code for GCN 
│   ├── QSAR/         # Pickle file of the QSAR models
│   └── Transformer/  # Code for Transformer
├── Utils/            # Utility functions
├── scripts/          # Code for running model training and compound generation
├── ckpts/            # The weights of trained Transformer and GCN models
├── translation/      # Output directory for Transformer inference experiments
├── mcts_out/         # Output directory for MCTS experiment results
├── env.yml           # The conda environment configuration file
├── set_up.sh         # Shell script to set up the $PYTHONPATH
└── config/           # Configuration file

```

## References

[1] Schwaller, P.; Probst, D.; Vaucher, A. C.; Nair, V. H.; Kreutter, D.; Laino, T.; Reymond, J.-L. Mapping the Space of Chemical Reactions Using Attention-Based Neural Networks. *Nat. Mach. Intell.* **2021**, *3*, 144–152, DOI: 10.1038/s42256-020-00284-w
