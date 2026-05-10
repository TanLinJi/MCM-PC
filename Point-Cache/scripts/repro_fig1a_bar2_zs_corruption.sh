#!/bin/bash
# Reproduce Figure 1(a) OpenShape — Bar 2 (orange):
#   ModelNet-C zero-shot, averaged over 7 corruptions x 5 severities = 35 settings.
#   Paper expected: 73.49
#
# Strategy:
#   - 35 jobs total; 2 GPUs (T4 x 2) -> 18 batches of 2 jobs each (last batch 1 job).
#   - Each job's stdout/stderr -> its own .log under logs/fig1a_bar2_<timestamp>/.
#   - --wandb-log NOT set; bypasses wandb entirely (zs_infer.py:40 already guarded).
#
# Usage:
#   bash Point-Cache/scripts/repro_fig1a_bar2_zs_corruption.sh
#   # or background:
#   nohup bash Point-Cache/scripts/repro_fig1a_bar2_zs_corruption.sh > fig1a_bar2.out 2>&1 &
#   tail -f fig1a_bar2.out

set -e
cd "$(dirname "$0")/.."   # cd into Point-Cache/

CKPT=weights/openshape/openshape-pointbert-vitg14-rgb/model.pt
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/fig1a_bar2_${TS}
mkdir -p "$LOG_DIR"

CORRUPTIONS=(jitter scale rotate dropout_global dropout_local add_global add_local)
SEVERITIES=(0 1 2 3 4)

# Build the full job list
JOBS=()
for cor in "${CORRUPTIONS[@]}"; do
  for sev in "${SEVERITIES[@]}"; do
    JOBS+=("${cor}_${sev}")
  done
done
N=${#JOBS[@]}
echo "[fig1a-bar2] total jobs: $N (paper: 35)"
echo "[fig1a-bar2] log dir   : $LOG_DIR"
echo "[fig1a-bar2] start time: $(date)"
echo ""

run_one () {
    local gpu=$1
    local name=$2          # e.g. add_global_2
    local logfile="$LOG_DIR/${name}.log"
    echo "  [GPU $gpu] launch: $name -> $logfile"
    CUDA_VISIBLE_DEVICES=$gpu \
    python runners/zs_infer.py \
        --config configs \
        --lm3d openshape \
        --cache-type global \
        --ckpt_path "$CKPT" \
        --dataset modelnet_c \
        --cor_type "$name" \
        --npoints 1024 \
        --oshape-version vitg14 \
        > "$logfile" 2>&1
    local final
    final=$(grep -oE 'Final\*\*\*[^:]*:[ ]*[0-9.]+' "$logfile" | tail -1 | grep -oE '[0-9.]+$')
    echo "  [GPU $gpu] done  : $name -> ${final:-FAIL}"
}

# Launch in pairs (one job per GPU per round)
for ((i=0; i<N; i+=2)); do
    j=$((i+1))
    name1=${JOBS[$i]}
    echo "[batch $((i/2+1))/18] $(date +%H:%M:%S) -- GPU0: $name1 ; GPU1: ${JOBS[$j]:-(none)}"
    run_one 0 "$name1" &
    PID0=$!
    if [ $j -lt $N ]; then
        run_one 1 "${JOBS[$j]}" &
        PID1=$!
        wait $PID0 $PID1
    else
        wait $PID0
    fi
done

echo ""
echo "[fig1a-bar2] all $N jobs done at $(date)"
echo "[fig1a-bar2] running summary..."
echo ""
python scripts/repro_fig1a_summarize.py "$LOG_DIR"
