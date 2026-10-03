# TRACER: Molecular Optimization Using Conditional Transformer for Reaction-Aware Compound Exploration with Reinforcement Learning

This repository contains the source code of TRACER, a framework for molecular optimization with synthetic pathways. TRACER integrates a conditional Transformer model trained on chemical reactions with Monte Carlo Tree Search (MCTS) for efficient exploration of the chemical space. For more details, please refer to the [paper](https://www.nature.com/articles/s42004-025-01437-x).

For a comprehensive description of the TRACER framework itself, please refer to the dedicated TRACER paper.

Nakamura, S., Yasuo, N. & Sekijima, M. Molecular optimization using a conditional transformer for reaction-aware compound exploration with reinforcement learning. Commun Chem 8, 40 (2025). DOI: https://doi.org/10.1038/s42004-025-01437-x

## Maintained branch and validation scope

`main` is retained for the historical paper-reproduction version. This
`security/modern-runtime` branch updates the runtime and is **not intended to
be merged into main**. The validated scope is macOS ARM64 CPU float32, the bundled
GCN/QSAR models and the published conditional Transformer. This does not establish
reproduction of the author's Linux/CUDA environment or full training runs.

Old-compatible math is the default: historical attention scaling, scalar-libm
softmax, eight-lane LayerNorm statistics and separate BatchNorm affine operations.
The compiled extension uses system libm and current tensor operators; it does not
load a historical PyTorch library. First-order gradients are supported. AMP,
non-CPU float32 and higher-order derivatives are not supported in this mode.
Use `TRACER_OLD_COMPATIBLE=0` for standard current PyTorch kernels and optionally
`TRACER_DEVICE=cuda`; GPU execution has not been validated.
Runtime/version/mode information is printed, saved alongside training outputs and
included in newly saved Transformer checkpoints. See [validation report](validation/REPORT.md).

## Installation

Create a new environment; leave historical research environments unchanged.
Python 3.12.15 and a C compiler are required for the validated setup.

```bash
uv --no-config venv --python /path/to/python3.12.15 .venv
source .venv/bin/activate
uv --no-config pip install -r requirements.txt
uv --no-config pip install --no-build-isolation --no-deps -e .
uv --no-config pip check
export PYTHONPATH="$PWD:$PWD/Model${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -p 'test_*.py' -v
```

The complete lock has 46 PyPI dependencies, including PyTorch 2.14.1, PyG
2.8.0.post1, NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 and RDKit 2026.3.6.
The local compiled runtime is installed separately. torchtext is replaced by an
ordered JSON vocabulary and native tensor padding. The original environment is
archived in [validation/legacy-environment.md](validation/legacy-environment.md).

## Setup Environment Variable

To ensure that the modules provided by TRACER can be imported, you need to add the paths to the PYTHONPATH environment variable. Please follow these steps:

At the top directory of the TRACER, run the following command to execute the `set_up.sh` script:

```
source set_up.sh
```

This command will set the `PYTHONPATH` environment variable to include the necessary directories.

Please note that you need to run this command every time you start a new terminal session.


## Download Model Parameters

The author's [Figshare checkpoint files](https://figshare.com/articles/software/Weights_of_conditional_unconditional_Transformer/25853551)
contain OmegaConf objects and cannot be loaded by the maintained tensor-only
loader. Convert the fixed published files **once**, offline, in an isolated
historical Python 3.10 reference environment. The conversion script requires an
exact source SHA256 match and writes only learned tensor weights. There is no
unsafe loading fallback in the updated generation/training code.

```bash
/path/to/reference/bin/python validation/convert_checkpoint.py \
  /path/to/ckpt_conditional.pth \
  ckpts/Transformer/ckpt_conditional.runtime.pth \
  --reference-repo /path/to/historical-TRACER
```

See [validation/REPRODUCE.md](validation/REPRODUCE.md) for the isolated reference
setup and hashes. The GCN checkpoint is bundled unchanged. QSAR inference uses
converted numerical `.npz` forests, not sklearn pickle. Their learned trees and
leaf probabilities are preserved; no retraining is involved. Historical pickle
files remain on main and in Git history. The unconditional file was converted
and hashed, but unconditional generation is outside the recorded comparisons.

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
