# Thesis Experiment Environment

This file records the retained environment used by the thesis experiment
scripts. The version snapshot was captured from `envs/rewi26` on 2026-07-14;
the same environment path appears in the archived SLURM launch scripts and
logs.

## Critical versions

- Python 3.12.10
- PyTorch 2.9.1+cu128
- CUDA runtime 12.8
- cuDNN 9.10.2 (PyTorch reports 91002)
- Transformers 4.57.3
- PEFT 0.18.1
- JiWER 4.0.0
- NumPy 2.2.5
- SciPy 1.17.0
- timm 1.0.24

`environment-lock.txt` pins the complete Python package set. Its SHA-256 is:

```text
24cfedcafecd62ccf3eb6cacde454ef662d09520337956184d93cb9c0b3ebfe2
```

Representative canonical HWR-GPT SLURM logs record an NVIDIA Tesla
V100-PCIE-32GB, NVIDIA driver 570.211.01, and CUDA 12.8. Job-specific logs are
authoritative for the device used by an individual fold; numerical wall-clock
comparisons are not reported in the thesis.

The canonical configurations load GPT-2 and ByT5-small from
`assets/hf_models/`. Because the retained local ByT5 metadata does not preserve
a reliable upstream commit identifier, `model-assets.sha256`
is the authoritative revision manifest for both local model/tokenizer trees.
