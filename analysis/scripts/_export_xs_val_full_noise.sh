#!/usr/bin/env bash
# Export val_full_fold{k}_epoch0.json for the AR-only and AR + noise
# injection HWRFormer (ar_transformer_xs) checkpoints on OnHW WI word,
# all 5 folds. Mirrors _export_xs_val_full.sh; the noise-injection
# checkpoint is treated as an AR-only model so the export filename
# carries no _ar suffix.
#
# Run from work/REWI_work:
#   bash analysis/scripts/_export_xs_val_full_noise.sh
set -uo pipefail

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RES=$WORK/imu-hwr/results/hwr2

if command -v conda >/dev/null 2>&1; then
    if [[ -f /apps/python/3.12-conda/etc/profile.d/conda.sh ]]; then
        # shellcheck disable=SC1091
        source /apps/python/3.12-conda/etc/profile.d/conda.sh
        conda activate "$WORK/imu-hwr/envs/rewi26" 2>/dev/null || true
    fi
fi
PY_BIN="${PY_BIN:-$WORK/imu-hwr/envs/rewi26/bin/python}"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
cd "$PROJ"

declare -a AR_RUNS=(
    "Baseline-AR-XS-blconv_b|ar_transformer_xs__onhw_wi_word_rh"
    "Baseline-AR-XS-blconv_b|ar_transformer_xs__onhw_wd_word_rh"
    "Baseline-AR-XS-blconv_b|ar_transformer_xs__wi_word_hw6_meta"
    "Baseline-AR-XS-blconv_b|ar_transformer_xs__wi_sent_hw6_meta"
)
declare -a NOISE_RUNS=(
    "Baseline-AR-XS-InputCorruption-uniform|ar_transformer_xs__onhw_wi_word_rh"
    "Baseline-AR-XS-InputCorruption-uniform|ar_transformer_xs__onhw_wd_word_rh"
    "Baseline-AR-XS-InputCorruption-uniform|ar_transformer_xs__wi_word_hw6_meta"
    "Baseline-AR-XS-InputCorruption-uniform|ar_transformer_xs__wi_sent_hw6_meta"
)

run_one() {
    local label="$1"; local dir_root="$2"; local arch="$3"; local fold="$4"
    local model_root="${RES}/${dir_root}/${arch}"
    local fold_dir="${model_root}/fold_${fold}"
    local idx_dir="${fold_dir}/${fold}"
    local ckpt="${idx_dir}/checkpoints/best_cer.pth"
    local exports_dir="${fold_dir}/exports"
    mkdir -p "${exports_dir}"

    local base_yaml
    base_yaml=$(ls -1 "${idx_dir}"/train_*.yaml 2>/dev/null | head -1)
    if [[ -z "${base_yaml}" ]]; then
        echo "[skip] ${label} fold ${fold}: no train_*.yaml in ${idx_dir}"
        return 1
    fi
    if [[ ! -f "${ckpt}" ]]; then
        echo "[skip] ${label} fold ${fold}: missing checkpoint ${ckpt}"
        return 1
    fi

    local expected="${exports_dir}/val_full_fold${fold}_epoch0.json"
    if [[ -f "${expected}" ]]; then
        echo "[done] ${label} fold ${fold}: ${expected}"
        return 0
    fi

    local patched
    patched=$(mktemp --suffix=.yaml)
    "${PY_BIN}" - "${base_yaml}" "${patched}" "${ckpt}" <<'PY'
import sys, yaml
src, dst, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["test"] = True
cfg["export_val_full"] = True
cfg["epoch"] = 1
cfg["freq_eval"] = 1
cfg["checkpoint"] = ckpt
cfg.pop("tokenizer_obj", None)
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=True)
print("patched", dst)
PY

    echo "[run]  ${label} fold ${fold}: ${expected}"
    "${PY_BIN}" main.py -c "${patched}"
    local rc=$?
    rm -f "${patched}"
    if [[ ${rc} -ne 0 ]]; then
        echo "[fail] ${label} fold ${fold}: main.py exit ${rc}"
        return ${rc}
    fi
    if [[ ! -f "${expected}" ]]; then
        echo "[warn] ${label} fold ${fold}: export missing after run (${expected})"
        return 1
    fi
    echo "[ok]   ${label} fold ${fold}: ${expected}"
    return 0
}

OK=0; FAIL=0
for entry in "${AR_RUNS[@]}"; do
    dir_root="${entry%%|*}"; arch="${entry##*|}"
    for f in 0 1 2 3 4; do
        if run_one "ar" "${dir_root}" "${arch}" "${f}"; then
            OK=$((OK+1))
        else
            FAIL=$((FAIL+1))
        fi
    done
done
for entry in "${NOISE_RUNS[@]}"; do
    dir_root="${entry%%|*}"; arch="${entry##*|}"
    for f in 0 1 2 3 4; do
        if run_one "noise" "${dir_root}" "${arch}" "${f}"; then
            OK=$((OK+1))
        else
            FAIL=$((FAIL+1))
        fi
    done
done

echo "Summary: ${OK} ok / ${FAIL} fail / 40 total"
exit $(( FAIL > 0 ? 1 : 0 ))
