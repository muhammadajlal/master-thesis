#!/usr/bin/env bash
# Parameterized timeout auto-resubmit monitor (v2).
#
# Same behaviour as timeout_auto_resubmit.sh but the watchlist comes
# from arguments: each argument is "job_id:sbatch_path". New jobs the
# monitor spawns are added to the watchlist automatically. State is
# shared with v1 via the same STATE_DIR so double-handling is impossible.
#
# Usage:
#   nohup bash scripts/timeout_auto_resubmit_v2.sh \
#       1726228:slurm/train_f1f2_ablation.sbatch \
#       > logs/timeout_monitor/monitor_v2.log 2>&1 &
set -euo pipefail

PROJ=/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work
cd "$PROJ"

STATE_DIR="${STATE_DIR:-$PROJ/logs/timeout_monitor}"
mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/handled_indices.tsv"
touch "$STATE_FILE"

POLL_SEC=${POLL_SEC:-900}

declare -A WATCH_SBATCH
if [[ $# -eq 0 ]]; then
    echo "ERROR: pass at least one job_id:sbatch_path argument" >&2
    exit 1
fi
for pair in "$@"; do
    job="${pair%%:*}"
    sb="${pair#*:}"
    WATCH_SBATCH[$job]="$sb"
done

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

already_handled() {
    grep -qF "${1}_${2}" "$STATE_FILE" 2>/dev/null
}

mark_handled() {
    printf '%s\t%s\t%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${1}_${2}" "handled_by" "$3" >> "$STATE_FILE"
}

pending_timeout_indices() {
    local job="$1"
    sacct -j "$job" --format=JobID,State --noheader --parsable2 2>/dev/null \
        | awk -F'|' '$1 !~ /\.(batch|extern)$/ && $2 == "TIMEOUT" {split($1, a, "_"); print a[2]}' \
        | sort -u \
        | while read -r idx; do
            already_handled "$job" "$idx" || echo "$idx"
        done
}

new_failed_indices() {
    local job="$1"
    sacct -j "$job" --format=JobID,State --noheader --parsable2 2>/dev/null \
        | awk -F'|' '$1 !~ /\.(batch|extern)$/ && $2 == "FAILED" {split($1, a, "_"); print a[2]}' \
        | sort -u \
        | while read -r idx; do
            grep -qF "${job}_${idx}_FAILED" "$STATE_FILE" 2>/dev/null || echo "$idx"
        done
}

job_state() {
    local n
    n=$(squeue -j "$1" -h --format="%T" 2>/dev/null | grep -cE "PENDING|RUNNING" || true)
    [[ "$n" -gt 0 ]] && echo "active" || echo "settled"
}

resubmit_batch() {
    sbatch --array="$2" "$1" 2>&1 | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}'
}

log "monitor v2 started; state file $STATE_FILE; poll ${POLL_SEC}s"
log "watchlist: ${!WATCH_SBATCH[*]}"

while true; do
    all_settled=1
    for job in "${!WATCH_SBATCH[@]}"; do
        state=$(job_state "$job")
        [[ "$state" == "active" ]] && all_settled=0

        mapfile -t timeouts < <(pending_timeout_indices "$job")
        if [[ "${#timeouts[@]}" -gt 0 ]]; then
            idx_csv=$(printf '%s,' "${timeouts[@]}" | sed 's/,$//')
            sbatch_path="${WATCH_SBATCH[$job]}"
            log "job $job: ${#timeouts[@]} new TIMEOUT indices ($idx_csv); resubmitting via $sbatch_path"
            new_job=$(resubmit_batch "$sbatch_path" "$idx_csv")
            if [[ -n "$new_job" ]]; then
                for idx in "${timeouts[@]}"; do
                    mark_handled "$job" "$idx" "$new_job"
                done
                WATCH_SBATCH[$new_job]="$sbatch_path"
                log "  -> new job $new_job added to watchlist"
                all_settled=0
            else
                log "  -> sbatch FAILED for $job ($idx_csv); retry next tick"
            fi
        fi

        mapfile -t failed < <(new_failed_indices "$job")
        for idx in "${failed[@]:-}"; do
            [[ -z "$idx" ]] && continue
            log "job $job: task $idx FAILED (not TIMEOUT); NOT auto-resubmitting"
            printf '%s\t%s\tFAILED_seen\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${job}_${idx}_FAILED" >> "$STATE_FILE"
        done
    done

    [[ "$all_settled" == "1" ]] && { log "all watched jobs settled; exiting"; break; }
    sleep "$POLL_SEC"
done

log "monitor v2 terminated normally"
