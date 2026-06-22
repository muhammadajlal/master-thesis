#!/bin/bash
JOB=1710148
while squeue -j $JOB -h -t PD,R 2>/dev/null | grep -q "^"; do
    sleep 60
done
echo "=== 1710148 ended at $(date) ==="
sacct -j $JOB --format=JobID,State,Elapsed,ExitCode 2>&1 | head -4
echo ""
D=/home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/results/hwr2/decode_study_xs_full_hybrid/stageD2_ar_neural_shallow_B4_lw0p1__fold3
if [ -f "$D/metrics.json" ]; then
    echo "metrics.json:"
    cat "$D/metrics.json"
fi
echo ""
echo "Re-aggregating all 5 folds for stageD2 hybrid:"
/home/woody/iwso/iwso214h/imu-hwr/envs/rewi26/bin/python /home/woody/iwso/iwso214h/imu-hwr/work/REWI_work/analysis/scripts/_agg_decode_xs.py
