#!/usr/bin/env bash
# Drive eval_tf_perturbation.py over the AR-only and AR + noise-injection
# HWRFormer xs checkpoints on OnHW WI word, 5 folds, perturbation rates
# p in {0.00, 0.05, 0.10, 0.15, 0.20}. Outputs land beside each fold's
# checkpoint as eval_tf_perturbation_p{NNN}.json. The aggregator script
# below collects all 50 result files into a single CSV.
set -uo pipefail

WORK=/home/woody/iwso/iwso214h
PROJ=$WORK/imu-hwr/work/REWI_work
RES=$WORK/imu-hwr/results/hwr2

if command -v conda >/dev/null 2>&1; then
    if [[ -f /apps/python/3.12-conda/etc/profile.d/conda.sh ]]; then
        source /apps/python/3.12-conda/etc/profile.d/conda.sh
        conda activate "$WORK/imu-hwr/envs/rewi26" 2>/dev/null || true
    fi
fi
PY_BIN="${PY_BIN:-$WORK/imu-hwr/envs/rewi26/bin/python}"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
cd "$PROJ"

declare -a RUNS=(
    "ar|Baseline-AR-XS-blconv_b|ar_transformer_xs__onhw_wi_word_rh"
    "ar|Baseline-AR-XS-blconv_b|ar_transformer_xs__onhw_wd_word_rh"
    "ar|Baseline-AR-XS-blconv_b|ar_transformer_xs__wi_word_hw6_meta"
    "ar|Baseline-AR-XS-blconv_b|ar_transformer_xs__wi_sent_hw6_meta"
    "noise|Baseline-AR-XS-InputCorruption-uniform|ar_transformer_xs__onhw_wi_word_rh"
    "noise|Baseline-AR-XS-InputCorruption-uniform|ar_transformer_xs__onhw_wd_word_rh"
    "noise|Baseline-AR-XS-InputCorruption-uniform|ar_transformer_xs__wi_word_hw6_meta"
    "noise|Baseline-AR-XS-InputCorruption-uniform|ar_transformer_xs__wi_sent_hw6_meta"
)
declare -a P_VALUES=(0.00 0.05 0.10 0.15 0.20)

run_one_eval() {
    local label="$1"; local dir_root="$2"; local arch="$3"; local fold="$4"; local p="$5"
    local idx_dir="${RES}/${dir_root}/${arch}/fold_${fold}/${fold}"
    local ckpt="${idx_dir}/checkpoints/best_cer.pth"
    local base_yaml
    base_yaml=$(ls -1 "${idx_dir}"/train_*.yaml 2>/dev/null | head -1)
    if [[ -z "${base_yaml}" ]]; then
        echo "[skip] ${label} fold ${fold} p=${p}: no train_*.yaml"
        return 1
    fi
    if [[ ! -f "${ckpt}" ]]; then
        echo "[skip] ${label} fold ${fold} p=${p}: missing checkpoint"
        return 1
    fi
    local p_int
    p_int=$(python -c "print(int(round(${p}*100)))")
    local p_tag
    printf -v p_tag "%03d" "${p_int}"
    local out_dir="${RES}/${dir_root}/${arch}/fold_${fold}"
    local out_path="${out_dir}/eval_tf_perturbation_p${p_tag}.json"
    if [[ -f "${out_path}" ]]; then
        echo "[done] ${label} fold ${fold} p=${p}: ${out_path}"
        return 0
    fi
    echo "[run]  ${label} fold ${fold} p=${p}: ${out_path}"
    "${PY_BIN}" eval_tf_perturbation.py \
        -c "${base_yaml}" \
        --checkpoint "${ckpt}" \
        --p_replace "${p}" \
        --seed $((fold * 100 + p_int)) \
        --out "${out_path}"
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[fail] ${label} fold ${fold} p=${p}: rc=${rc}"
        return ${rc}
    fi
    return 0
}

OK=0; FAIL=0
for entry in "${RUNS[@]}"; do
    label="${entry%%|*}"
    rest="${entry#*|}"
    dir_root="${rest%%|*}"
    arch="${rest##*|}"
    for f in 0 1 2 3 4; do
        for p in "${P_VALUES[@]}"; do
            if run_one_eval "${label}" "${dir_root}" "${arch}" "${f}" "${p}"; then
                OK=$((OK+1))
            else
                FAIL=$((FAIL+1))
            fi
        done
    done
done

echo "Summary: ${OK} ok / ${FAIL} fail / 200 total"
exit $(( FAIL > 0 ? 1 : 0 ))
