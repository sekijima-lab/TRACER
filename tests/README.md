# tqdm security update validation

tqdm **4.65.0 → 4.66.4** addresses
[CVE-2024-34062 / GHSA-g7vv-2v7x-gj9p](https://github.com/advisories/GHSA-g7vv-2v7x-gj9p).
The CLI previously evaluated Python expressions in some optional arguments.
The upstream fix starts at 4.66.3; 4.66.4 is the next available conda-forge
release and its noarch package supports the existing Python >=3.7 requirement.
TRACER uses the Python iterator API; no exploitable CLI use was identified in
the inspected code. This removes the vulnerable installed CLI defensively.

In a new Python 3.10 virtual environment:

```sh
python -m pip install -r tests/requirements-tqdm.txt
python tests/test_tqdm_compatibility.py
python -m pip check
```

On macOS arm64 / Python 3.10.19, the `tqdm` and `tqdm.auto` iterators preserve
five SMILES exactly, including ordering/stereochemistry, and handle empty
input. CLI piping preserves all input bytes and reports the correct count.
Both compatibility checks pass with 4.65.0 and 4.66.4.
The added harmless CLI expression (integer/string arithmetic only) is
accepted by 4.65.0 and rejected by 4.66.4. All three updated tests and
dependency checks pass; OSV returned no advisories for 4.66.4 on 2026-10-02.

To compare the baseline, install only `tqdm==4.65.0` in a second environment
and run with `TQDM_BASELINE=1` to skip the expected failing security check.

Only the tqdm pin changes. No model or numerical dependency changes.
The suite covers the iterator API used by `scripts/beam_search.py` and
`scripts/gcn_train.py`, without importing their GPU/model dependencies.
Full Linux/CUDA environment solving, model training, generation and docking
are not verified. Other known dependency advisories remain.
