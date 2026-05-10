#!/bin/bash
# Reproduce Figure 1(a) OpenShape — Bar 3 (purple):
#   ModelNet-C Point-Cache hierarchical TTA, averaged over 35 settings.
#   Paper expected: 76.59
#
# Differences vs bar2:
#   - runner: zs_infer.py -> model_with_hierarchical_caches.py
#   - --cache-type global -> hierarchical
#   - MUST pass --wandb-log: runner has unguarded wandb.log() at line 227.
#     We use WANDB_MODE=offline so it stays local; gitignored.
#   - per-job time ~10-15 min (build pos cache pass + online TDA pass)
#   - 35 jobs / 2 GPUs ~= 3.5-4 hours total.
#
# Usage:
#   tmux new -s bar3
#   bash Point-Cache/scripts/repro_fig1a_bar3_tta.sh
#   # detach: Ctrl-b d ; reattach: tmux attach -t bar3

set -e
cd "$(dirname "$0")/.."

CKPT=weights/openshape/openshape-pointbert-vitg14-rgb/model.pt
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/fig1a_bar3_${TS}
mkdir -p "$LOG_DIR"

CORRUPTIONS=(jitter scale rotate dropout_global dropout_local add_global add_local)
SEVERITIES=(0 1 2 3 4)

JOBS=()
for cor in "${CORRUPTIONS[@]}"; do
  for sev in "${SEVERITIES[@]}"; do
    JOBS+=("${cor}_${sev}")
  done
done
N=${#JOBS[@]}
echo "[fig1a-bar3] total jobs: $N (paper: 35)"
echo "[fig1a-bar3] log dir   : $LOG_DIR"
echo "[fig1a-bar3] start time: $(date)"
echo "[fig1a-bar3] ETA       : ~3.5-4 hours on 2x T4"
echo ""

run_one () {
    local gpu=$1
    local name=$2
    local logfile="$LOG_DIR/${name}.log"
    echo "  [GPU $gpu] launch: $name -> $logfile"
    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu \
    python runners/model_with_hierarchical_caches.py \
        --config configs \
        --lm3d openshape \
        --cache-type hierarchical \
        --ckpt_path "$CKPT" \
        --dataset modelnet_c \
        --cor_type "$name" \
        --npoints 1024 \
        --oshape-version vitg14 \
        --wandb-log \
        > "$logfile" 2>&1
    local final
    final=$(grep -oE 'Final\*\*\*[^:]*:[ ]*[0-9.]+' "$logfile" | tail -1 | grep -oE '[0-9.]+$')
    echo "  [GPU $gpu] done  : $name -> ${final:-FAIL}"
}

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
echo "[fig1a-bar3] all $N jobs done at $(date)"
echo "[fig1a-bar3] running summary..."
echo ""
python scripts/repro_fig1a_summarize.py "$LOG_DIR"
