#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Timeout auto-resubmit monitor for Chapter 6 sweeps.
#
# Watches the specified SLURM jobs. When a task ends in TIMEOUT,
# resubmits that same array index using the associated sbatch. The
# sbatch's auto-resume-from-last.pth logic then picks up mid-training.
#
# The monitor:
#  - polls sacct every ${POLL_SEC:-900}s (15 min default)
#  - tracks which (job, index) pairs have already been resubmitted in
#    a state file, so it never re-resubmits the same task twice
#  - handles FAILED tasks by logging them and NOT resubmitting (they
#    likely need code investigation)
#  - exits when every watched job has zero PENDING or RUNNING tasks
#    left AND its own resubmission chain has settled
#
# Usage:
#   nohup bash scripts/timeout_auto_resubmit.sh > logs/monitor.log 2>&1 &
#
# Watchlist format (space-separated: job_id:sbatch_path):
#   WATCH=(job_id:sbatch_path [job_id:sbatch_path ...])
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

PROJ=/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work
cd "$PROJ"

STATE_DIR="${STATE_DIR:-$PROJ/logs/timeout_monitor}"
mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/handled_indices.tsv"
touch "$STATE_FILE"

POLL_SEC=${POLL_SEC:-900}

# Watch: original_job:associated_sbatch
# Any new sbatch spawned by the monitor is also watched (added dynamically).
declare -A WATCH_SBATCH
WATCH_SBATCH[1725240]="slurm/train_noctc_ch6.sbatch"
WATCH_SBATCH[1725241]="slurm/train_lambda02_ch6.sbatch"

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

already_handled() {
    local job="$1" idx="$2"
    grep -qF "${job}_${idx}" "$STATE_FILE" 2>/dev/null
}

mark_handled() {
    local job="$1" idx="$2" new_job="$3"
    printf '%s\t%s\t%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${job}_${idx}" "handled_by" "$new_job" >> "$STATE_FILE"
}

# Given a job ID, print TIMEOUT array indices we have not yet handled.
pending_timeout_indices() {
    local job="$1"
    sacct -j "$job" --format=JobID,State --noheader --parsable2 2>/dev/null \
        | awk -F'|' -v j="$job" '
            $1 !~ /\.(batch|extern)$/ && $2 == "TIMEOUT" {
                split($1, a, "_")
                print a[2]
            }' \
        | sort -u \
        | while read -r idx; do
            if ! already_handled "$job" "$idx"; then
                echo "$idx"
            fi
        done
}

# Given a job ID, print FAILED array indices we have not yet warned about.
new_failed_indices() {
    local job="$1"
    sacct -j "$job" --format=JobID,State --noheader --parsable2 2>/dev/null \
        | awk -F'|' -v j="$job" '
            $1 !~ /\.(batch|extern)$/ && $2 == "FAILED" {
                split($1, a, "_")
                print a[2]
            }' \
        | sort -u \
        | while read -r idx; do
            if ! grep -qF "${job}_${idx}_FAILED" "$STATE_FILE" 2>/dev/null; then
                echo "$idx"
            fi
        done
}

# Return "active" if the job still has PENDING or RUNNING tasks, "settled" otherwise.
job_state() {
    local job="$1"
    local n
    n=$(squeue -j "$job" -h --format="%T" 2>/dev/null | grep -cE "PENDING|RUNNING" || true)
    if [[ "$n" -gt 0 ]]; then
        echo "active"
    else
        echo "settled"
    fi
}

resubmit_batch() {
    local sbatch_path="$1" idx_csv="$2"
    local out
    out=$(sbatch --array="$idx_csv" "$sbatch_path" 2>&1)
    printf '%s\n' "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $4}'
}

log "monitor started; state file $STATE_FILE; poll interval ${POLL_SEC}s"
log "watchlist: ${!WATCH_SBATCH[*]}"

while true; do
    all_settled=1

    # Loop over a snapshot of keys so we can add new jobs mid-iteration.
    for job in "${!WATCH_SBATCH[@]}"; do
        state=$(job_state "$job")
        [[ "$state" == "active" ]] && all_settled=0

        # Handle TIMEOUT: resubmit indices we have not yet acted on.
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
                # Watch the new job too (it inherits the same sbatch + resume logic)
                WATCH_SBATCH[$new_job]="$sbatch_path"
                log "  -> new job $new_job watching ${#timeouts[@]} tasks; added to watchlist"
                all_settled=0
            else
                log "  -> sbatch FAILED for job $job indices $idx_csv; will retry next tick"
            fi
        fi

        # Handle FAILED: log once, do NOT resubmit (needs code investigation).
        mapfile -t failed < <(new_failed_indices "$job")
        if [[ "${#failed[@]}" -gt 0 ]]; then
            for idx in "${failed[@]}"; do
                log "job $job: task $idx FAILED (not TIMEOUT); NOT auto-resubmitting"
                printf '%s\t%s\tFAILED_seen\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${job}_${idx}_FAILED" >> "$STATE_FILE"
            done
        fi
    done

    if [[ "$all_settled" == "1" ]]; then
        log "all watched jobs settled (0 PENDING or RUNNING); exiting"
        break
    fi

    sleep "$POLL_SEC"
done

log "monitor terminated normally"
