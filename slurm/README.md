# SLURM launch scripts (FAU NHR cluster)

These sbatch/shell scripts were used to run the thesis experiments on the FAU
Woody/TinyGPU cluster (V100 / RTX 3080 / A100 partitions). They are kept for
provenance and contain cluster-specific paths and module names; they are **not**
required to reproduce results.

The portable equivalents are:

```bash
python train_cv.py -c <config.yaml>   # all folds sequentially
python main.py     -c <config.yaml>   # one fold (idx_fold 0..4)
python evaluate.py -c <config.yaml>   # 5-fold aggregation + MACs/params
```

Most scripts follow the same pattern as `train.sbatch`: copy the YAML to a temp
dir, patch `idx_fold` from `$SLURM_ARRAY_TASK_ID`, patch `dir_work`/`dir_dataset`
from the `TRAIN_YAML` and `DATASET` environment variables, set `HF_HUB_OFFLINE=1`,
and run `main.py`.
