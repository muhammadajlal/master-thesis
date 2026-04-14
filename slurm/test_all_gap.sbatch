#!/bin/bash
#SBATCH -J test-gap-all
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#SBATCH -p rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH -t 08:00:00

set -euo pipefail
set -x

# ---- Configuration ----
FOLD=${FOLD:-0}

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RESULTS=$WORK/imu-hwr/results/hwr2

# Setup conda
module load python/3.12-conda 2>/dev/null || true
source /apps/python/3.12-conda/etc/profile.d/conda.sh
ENV_PREFIX=/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26
conda activate "$ENV_PREFIX"
PY_BIN="$ENV_PREFIX/bin/python3"

export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HF_HOME=$PROJ/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$PROJ"

# ---- Experiment list ----
# Format: "RESULT_SUBDIR|SHORT_NAME"
declare -a EXPERIMENTS=(
  "Ablations-MMLM/GPT-2/Hybrid/H1_hybrid_mlp|H1_hybrid_mlp"
  "Ablations-MMLM/GPT-2/Hybrid-Contrastive/J1_contrastive_mlp|J1_contrastive_mlp"
  "Ablations-MMLM/GPT-2/Hybrid-Contrastive/J2_contrastive_mlp|J2_contrastive_mlp"
  "Ablations-MMLM/GPT-2/Hybrid/K1_ctc_mse|K1_ctc_mse"
  "Ablations-MMLM/GPT-2/Hybrid/K2_ctc_posterior|K2_ctc_posterior"
  "Ablations-MMLM/GPT-2/Hybrid-Contrastive/K3_ec_loss|K3_ec_loss"
  "K4_sea_contrastive|K4_sea_contrastive"
  "Ablations-MMLM/GPT-2/Hybrid/K5_kv_multiview|K5_kv_multiview"
)

for ENTRY in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r RESULT_SUBDIR SHORT_NAME <<< "$ENTRY"

  for DS_DIR in "vlm__onhw_wi_word_rh" "vlm__wi_word_hw6_meta"; do
    EXP_DIR="${RESULTS}/${RESULT_SUBDIR}/${DS_DIR}/fold_${FOLD}/${FOLD}"
    CKPT="${EXP_DIR}/checkpoints/best_cer.pth"
    CONFIG=$(ls "${EXP_DIR}"/train_*.yaml 2>/dev/null | head -1 || true)

    if [[ ! -f "$CKPT" || -z "$CONFIG" ]]; then
      echo "SKIP: ${SHORT_NAME} / ${DS_DIR} (missing checkpoint or config)"
      continue
    fi

    echo ""
    echo "========== ${SHORT_NAME} / ${DS_DIR} =========="

    # Create a temporary test config that loads the checkpoint and exports predictions
    TMP_CONFIG=$(mktemp /tmp/test_XXXXXX.yaml)
    cp "$CONFIG" "$TMP_CONFIG"

    # Patch the config for test mode
    $PY_BIN -c "
import yaml
with open('$TMP_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['test'] = True
cfg['epoch'] = 1
cfg['checkpoint'] = '$CKPT'
cfg['export_val_full'] = True
cfg['idx_fold'] = $FOLD
cfg['dir_work'] = '$EXP_DIR'
cfg['device'] = 'cuda'
with open('$TMP_CONFIG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"

    echo "Config: $TMP_CONFIG"
    echo "Checkpoint: $CKPT"

    $PY_BIN main.py -c "$TMP_CONFIG" || echo "FAILED: ${SHORT_NAME} / ${DS_DIR}"

    rm -f "$TMP_CONFIG"

    echo "Done: ${SHORT_NAME} / ${DS_DIR}"
  done
done

echo ""
echo "========== ALL DONE =========="
echo "Exports saved alongside checkpoints in results/hwr2/.../exports/"
