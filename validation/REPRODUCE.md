# Reproduce the CPU comparisons

Install the maintained isolated runtime using the top-level README. Do not alter
an existing historical research environment. Compilation needs a C compiler.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python validation/benchmark.py --repo "$PWD" --out /tmp/tracer-gcn --seed 1729
python validation/transformer_benchmark.py --repo "$PWD" \
  --checkpoint ckpts/Transformer/ckpt_conditional.runtime.pth \
  --out /tmp/tracer-transformer
python validation/compare.py --gcn /tmp/tracer-gcn --transformer /tmp/tracer-transformer
python validation/mcts_benchmark.py --repo "$PWD" \
  --checkpoint ckpts/Transformer/ckpt_conditional.runtime.pth \
  --out /tmp/tracer-mcts.json
python validation/cli_smoke.py --repo "$PWD" --out /tmp/tracer-cli-smoke
```

The CLI smoke needs a new empty output directory and writes small training
checkpoints in ignored directories. It uses two DataLoader workers, so OS shared
memory/process permissions are required. It performs actual Transformer and GCN
training in both modes and verifies saved weights, optimizer state and runtime
records. This is not a full training run.

## Independent historical reference

Extract Git commit `c9c7e314d467cdac6f4388f3a4ecbf80159bc135` into a separate
checkout. Create a **new** Python 3.10.19 environment and install `legacy-pins.txt`.
The reconstructed RDKit version is 2022.9.5 instead of Linux Conda 2022.03.2;
see `REPORT.md`. The reference has known advisories and is used only with fixed
trusted repository data/checkpoints. It is not a maintained deployment runtime.

Download the author's Figshare files listed in `checkpoint-conversion.json`.
The conversion script checks the exact source SHA256 before using trusted legacy
pickle loading. It must be run in the historical environment with the original
config module; it is not an automatic fallback in maintained loaders.

```bash
/path/to/reference/bin/python validation/convert_checkpoint.py \
  /path/to/ckpt_conditional.pth \
  ckpts/Transformer/ckpt_conditional.runtime.pth \
  --reference-repo /path/to/historical-TRACER
```

Run the same benchmark scripts with the old interpreter, `--repo` pointing to
that unchanged historical checkout, and `--legacy` added. Transformer/MCTS old
runs use the original author checkpoint. Baseline arrays and new outputs should
be placed in separate directories. CPU thread count is fixed to one in scripts.

Supply full arrays with `compare.py --legacy-gcn /path/to/old/numeric.npz
--legacy-transformer /path/to/old-transformer/numeric.npz` for complete gradient
and weight comparison. Without these arguments, the committed reference contains
4,096 regularly spaced gradient/weight entries with indices; forward/preprocessing
arrays and generated sequences are complete. Full-array maxima were verified
locally and recorded. Repeat GCN with seeds 19 and 73 as well as 1729.

## QSAR conversion

The numeric forests already ship in this branch. To independently reconstruct
them, use the historical trusted pickle models from main and the sklearn 1.2.2
reference environment:

```bash
/path/to/reference/bin/python validation/export_qsar.py \
  --source /path/to/historical-TRACER/Model/QSAR --out /tmp/qsar-converted
```

A separate sklearn 1.1.1 environment was used to validate CXCR4's original saved
version. The conversion does not retrain trees and does not accept arbitrary
pickle files in the maintained runtime. Hashes are in `qsar-conversion.json`.
