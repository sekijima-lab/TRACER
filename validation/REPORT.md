# TRACER CPU security runtime validation

## Branch policy and current decision

Main stays at the historical version; the updated runtime is a separate branch.
The restored implementation passes the frozen numerical criteria in the
recorded CPU comparisons. This branch is maintained separately and is not
intended to merge into the historical main branch.

The independent original source is commit
`c9c7e314d467cdac6f4388f3a4ecbf80159bc135`. Its Python 3.10.19/PyTorch 2.0.1,
torchtext 0.15.2, PyG 2.3.0, sklearn 1.2.2, NumPy 1.23.5 and pandas 1.5.3
CPU reference substitutes RDKit 2022.9.5 for Linux Conda 2022.03.2 and Python
3.10.19 for 3.10.10. Newly resolved transitive packages also differ. This is not
the author's exact CUDA environment. CXCR4's pickle records sklearn 1.1.1;
a separate 1.1.1 environment independently confirmed its reference probabilities.

## Model and preprocessing provenance

- Same bundled GCN weights and three QSAR forests; no retraining of saved models.
- Author Figshare conditional/unconditional files: publisher MD5 checked, source
  and converted SHA256 recorded in `checkpoint-conversion.json`. Only tensor
  state is retained for runtime checkpoints. Fixed-hash trusted conversion is
  isolated from maintained loading, which always uses `weights_only=True`.
- All 1,053 vocabulary tokens/order, special and unknown IDs and padding match
  the original torchtext implementation, reconstructed from the complete bundled
  training/validation corpus. Persistence is JSON rather than pickled native vocab.
- 725 unique molecules from three QSAR test files: Morgan radius 3/2048-bit
  fingerprints exact. Three forest probability maxima differ by at most 2.22e-16
  (criterion 1e-12); CXCR4 native sklearn 1.1.1 agrees within the same tolerance.
- Recorded graph node/edge features and batch masks for 32 molecules are exact.

## Numerical differences and their causes

The criteria were recorded **before** the first old/new comparison. They have not
been loosened. Failed intermediate measurements are retained for audit purposes.

Standard GCN logits differed by 1.72e-5, and one Adam step weights by 7.91e-4,
exceeding criteria 1e-5 and 1e-4. BatchNorm statistics and same-input native
backward results matched exactly, but the affine operation order differed.
Restoring `alpha = invstd * weight`, `beta = bias - mean * alpha`, followed by
separate `x * alpha + beta`, reproduces the old forward pass. All recorded hidden
outputs through the final linear layer then matched. The remaining upstream
gradient difference was 7.45e-9 at log-softmax. Scalar libm expf/logf and the
historical eight-lane sum order reproduce that final operation and its derivative.
The previous near-zero gradient sign differences were amplified by Adam; they
are removed by fixing the originating math rather than ignoring small gradients.

Transformer standard logits differed by 6.10e-5; selecting current math SDPA
alone reduced this to 4.20e-5, still above the 1e-5 criterion. Same-input linear
layers matched. Historical eight-lane cascade/Welford moments reproduce every
recorded LayerNorm mean and inverse standard deviation exactly. LayerNorm's
separate affine operations and attention's **division of Q and K** by the fourth
root of head dimension are restored. Current math SDPA uses a different scaling
operation; merely choosing its backend did not restore the original arithmetic.
Attention uses historical scalar-libm softmax and current tensor matrix operations.
No global monkeypatch or old PyTorch binary is used.

## Restored comparisons

- GCN: 32 real molecular graphs. Output, loss, full first gradients and complete
  weights after one Adam step are exact in the restored baseline comparison.
  Seeds 1729/19/73 all match exactly in `gcn-three-seed-comparison.json`.
  The actual training script’s double-log-softmax CrossEntropy path also matches
  loss, full gradients and full Adam weights exactly (`gcn-crossentropy-comparison.json`).
- Conditional Transformer: eight real validation reactions; logits and argmax
  exact. Training loss maximum difference 4.77e-7, full gradient 7.51e-6 and
  warmup Adam weights 2.98e-8, below criteria 1e-4. Learning rate exact.
  Dropout is disabled for this deterministic full-model training comparison.
- Three inputs: actual greedy and beam-width-3/top-2 decoding outputs exact.
- MCTS comparison: two initial molecules, three seeds, two steps and one rollout
  depth, using actual saved models and DRD2 rewards. Molecules, synthesis routes
  and accept/reject results match; intermediate reward differences were <=5.55e-17.
  The final restored run passes all six conditions (`mcts-final-comparison.json`).

Full arrays are retained in the maintenance workspace; compact gradient/weight
references contain 4,096 regularly spaced elements, with indices. All forward
arrays are retained in full. Full-array maxima, not just compact samples, support
these conclusions. The scripts and references permit independent repeat checks.

## Security and runtime checks

- Python 3.12.15, 46 pinned PyPI distributions and the local compiled runtime.
  OSV official name/version snapshot matches **0 of 46 updated versions** and
  3 of 37 reconstructed reference versions. This is not a complete audit of
  historical Conda/native libraries, Python or application code. Original
  downstream Conda patches were not assessed.
- 14 tests: ordered vocabulary/JSON and padding; 725-molecule forest reference;
  object NPZ rejection; GCN output/rank reference and mode override; unsupported
  checkpoint pickle rejection and safe runtime-metadata roundtrip; LayerNorm statistics/output/gradients; actual
  old masked attention reference; C buffer rejection/log-softmax derivatives;
  and three existing tqdm compatibility/security tests.
- Actual small-data CLI training for Transformer and GCN, both math modes,
  checkpoint saving and runtime metadata. CLI completion is not full-training
  numerical equivalence. Fresh installation/dependency check and final restored
  tests/comparisons are recorded in the maintenance audit logs.

## Limits

Validated macOS ARM64 CPU float32 and first-order derivatives only. A C compiler
is required. System libm and reduction behavior may differ on other platforms.
CUDA/GPU, Linux, AMP, higher-order derivatives, full 500,000-step Transformer or
100-epoch GCN training, unconditional generation and complete dataset MCTS
benchmarks are not validated. No claim of universal bitwise reproducibility is made.
No GitHub CI workflow is added or executed. Paper reproduction should continue to
use main with the originally required environment, model weights and data.

Primary implementation references: PyTorch v2.0.1
[BatchNorm](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cpu/batch_norm_kernel.cpp),
[LayerNorm](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cpu/layer_norm_kernel.cpp),
[moments](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cpu/moments_utils.h),
[softmax](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/cpu/SoftMaxKernel.cpp),
[attention](https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/native/transformers/attention.cpp).
