#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Export `val_full_fold{k}_epoch0[_ar].json` for XS AR-only + Hybrid
# on OnHW-WI and OnHW-WD, all 5 folds (2 datasets × 2 models × 5 = 20).
#
# Each per-fold run patches `test:true` and `export_val_full:true` onto
# the canonical training YAML stashed in the result dir and re-invokes
# main.py with the best_cer checkpoint. Outputs land under
# `<dir_work>/fold_<k>/exports/val_full_fold{k}_epoch0[_ar].json`,
# matching the schema thesis_dual_dataset_analysis.py reads from.
#
# Run from work/REWI_work:
#   bash analysis/scripts/_export_xs_val_full.sh
# ═══════════════════════════════════════════════════════════════════
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

# (dir_work_root, arch_suffix=dataset_dir_name) tuples
declare -a AR_RUNS=(
    "Baseline-AR-XS-blconv_b|ar_transformer_xs__onhw_wi_word_rh"
    "Baseline-AR-XS-blconv_b|ar_transformer_xs__onhw_wd_word_rh"
)
declare -a HYB_RUNS=(
    "train_element_word_hybrid_06_xs_onhw_wi|ar_transformer_xs__onhw_wi_word_rh"
    "train_element_word_hybrid_06_xs_onhw_wd|ar_transformer_xs__onhw_wd_word_rh"
)

run_one() {
    local label="$1"; local dir_root="$2"; local arch="$3"; local fold="$4"
    local model_root="${RES}/${dir_root}/${arch}"
    local fold_dir="${model_root}/fold_${fold}"
    local idx_dir="${fold_dir}/${fold}"
    local ckpt="${idx_dir}/checkpoints/best_cer.pth"
    local exports_dir="${fold_dir}/exports"
    mkdir -p "${exports_dir}"

    # Pick canonical train_*.yaml as the base.
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

    # Decide which export filename signals success.
    local out_suffix=""
    if [[ "${label}" == "hyb" ]]; then
        out_suffix="_ar"
    fi
    local expected="${exports_dir}/val_full_fold${fold}_epoch0${out_suffix}.json"
    if [[ -f "${expected}" ]]; then
        echo "[done] ${label} fold ${fold}: ${expected}"
        return 0
    fi

    # Patch YAML: set test:true, export_val_full:true, epoch:1, freq_eval:1.
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
# Drop the resolved tokenizer_obj if present so main.py rebuilds it.
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
for entry in "${HYB_RUNS[@]}"; do
    dir_root="${entry%%|*}"; arch="${entry##*|}"
    for f in 0 1 2 3 4; do
        if run_one "hyb" "${dir_root}" "${arch}" "${f}"; then
            OK=$((OK+1))
        else
            FAIL=$((FAIL+1))
        fi
    done
done

echo "Summary: ${OK} ok / ${FAIL} fail / 20 total"
exit $(( FAIL > 0 ? 1 : 0 ))
