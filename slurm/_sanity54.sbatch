#!/bin/bash
#SBATCH -J sc-sanity
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p v100,rtx3080,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
set -uo pipefail
set -x
WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RES=$WORK/imu-hwr/results/hwr2
OUT=$WORK/imu-hwr/results/_sanity54
mkdir -p "$OUT"
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate $WORK/imu-hwr/envs/rewi26
PY=$WORK/imu-hwr/envs/rewi26/bin/python3
cd "$PROJ"; export PYTHONPATH="$PROJ:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "PY=$PY"; "$PY" --version

NOISE=$RES/Baseline-AR-XS-InputCorruption-uniform/ar_transformer_xs__onhw_wi_word_rh/fold_0/0
ARONLY=$RES/Baseline-AR-XS-blconv_b/ar_transformer_xs__onhw_wi_word_rh/fold_0/0
YN=$(ls "$NOISE"/train_*.yaml | head -1); YA=$(ls "$ARONLY"/train_*.yaml | head -1)
echo "YN=$YN"; echo "YA=$YA"

echo "#### F2 TF-gap +Noise OnHW-WI fold_0 (OLD tf=11.10 ar=13.88) ####"
"$PY" eval_tf_gap.py -c "$YN" --checkpoint "$NOISE/checkpoints/best_cer.pth" --out "$OUT/tfgap.json" --device cuda
echo "#### F1 single-corruption both models seed=0 ####"
"$PY" eval_single_corruption.py -c "$YN" --checkpoint "$NOISE/checkpoints/best_cer.pth" --out "$OUT/sc_noise.json"  --seed 0 --device cuda
"$PY" eval_single_corruption.py -c "$YA" --checkpoint "$ARONLY/checkpoints/best_cer.pth" --out "$OUT/sc_aronly.json" --seed 0 --device cuda
echo "#### DONE ####"
