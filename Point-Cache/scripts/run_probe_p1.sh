#!/usr/bin/env bash
# P1 feature drift probe driver (D19 post-mortem, 2026-05-11).
#
# Three independent stages, each producing a markdown report so partial
# progress is always salvageable:
#   Stage 0 (smoke):       100-sample sanity, single GPU,  ~3 min.
#   Stage 1 (scale only):  clean + scale_0..4 full,        ~12 min dual-GPU.
#   Stage 2 (full 7x5):    remaining 30 settings,          ~45 min dual-GPU.
#   Stage 3 (pollution):   CPU-only feature-space NN sim,  ~10 min.
#
# Usage:
#   STAGE=smoke   bash Point-Cache/scripts/run_probe_p1.sh   # quick pipeline test
#   STAGE=scale   bash Point-Cache/scripts/run_probe_p1.sh   # stage 1 only
#   STAGE=full    bash Point-Cache/scripts/run_probe_p1.sh   # stage 1 + 2
#   STAGE=all     bash Point-Cache/scripts/run_probe_p1.sh   # 1 + 2 + 3
#   STAGE=pollute bash Point-Cache/scripts/run_probe_p1.sh   # stage 3 only

set -e
cd "$(dirname "$0")/.."  # cd to Point-Cache/

STAGE="${STAGE:-smoke}"
PY="/root/miniconda3/envs/mcmpc/bin/python"
FEAT_DIR="reports/p1_features"

mkdir -p "$FEAT_DIR"

# --------- helper -----------------------------------------------------------
extract_dual_gpu () {
    # $1 = space-separated cor_types list
    # Splits the list across GPU 0 and GPU 1 (half-half) and runs in parallel.
    local CORS=( $1 )
    local N=${#CORS[@]}
    local HALF=$(( (N + 1) / 2 ))
    local GPU0=( "${CORS[@]:0:$HALF}" )
    local GPU1=( "${CORS[@]:$HALF}" )

    echo "[probe-p1] GPU0 will do: ${GPU0[*]}"
    echo "[probe-p1] GPU1 will do: ${GPU1[*]}"

    LOG_DIR="reports/p1_features/_logs_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"

    if [ ${#GPU0[@]} -gt 0 ]; then
        (CUDA_VISIBLE_DEVICES=0 $PY runners/probe_p1_feature_drift.py \
            --cor_types ${GPU0[*]} \
            --output_dir "$FEAT_DIR" --device 0 \
            > "$LOG_DIR/gpu0.log" 2>&1) &
        PID0=$!
    fi
    if [ ${#GPU1[@]} -gt 0 ]; then
        (CUDA_VISIBLE_DEVICES=1 $PY runners/probe_p1_feature_drift.py \
            --cor_types ${GPU1[*]} \
            --output_dir "$FEAT_DIR" --device 0 \
            > "$LOG_DIR/gpu1.log" 2>&1) &
        PID1=$!
    fi
    [ -n "${PID0:-}" ] && wait "$PID0" && echo "[probe-p1] GPU0 done -> $LOG_DIR/gpu0.log"
    [ -n "${PID1:-}" ] && wait "$PID1" && echo "[probe-p1] GPU1 done -> $LOG_DIR/gpu1.log"
    tail -n 20 "$LOG_DIR"/gpu*.log
}

# --------- stages -----------------------------------------------------------

if [ "$STAGE" = "smoke" ]; then
    echo "[probe-p1] === STAGE smoke (100 samples) ==="
    CUDA_VISIBLE_DEVICES=0 $PY runners/probe_p1_feature_drift.py \
        --cor_types clean scale_2 \
        --output_dir "reports/p1_features_smoke" \
        --max_samples 100 --batch_size 32 --device 0
    $PY scripts/aggregate_p1.py \
        --feat_dir reports/p1_features_smoke \
        --output_md  reports/P1_smoke.md \
        --output_json reports/P1_smoke.json
    echo "[probe-p1] smoke done. See reports/P1_smoke.md"
    exit 0
fi

if [ "$STAGE" = "scale" ] || [ "$STAGE" = "full" ] || [ "$STAGE" = "all" ]; then
    echo "[probe-p1] === STAGE 1: scale-only feature drift ==="
    SCALE_CORS="clean scale_0 scale_1 scale_2 scale_3 scale_4"
    extract_dual_gpu "$SCALE_CORS"
    $PY scripts/aggregate_p1.py \
        --feat_dir "$FEAT_DIR" \
        --reference clean \
        --output_md  reports/P1_scale_drift.md \
        --output_json reports/P1_scale_drift.json
    echo "[probe-p1] Stage 1 done -> reports/P1_scale_drift.md"
fi

if [ "$STAGE" = "full" ] || [ "$STAGE" = "all" ]; then
    echo "[probe-p1] === STAGE 2: full 7-family x 5-severity feature drift ==="
    # 6 families * 5 severities = 30 settings (clean already done in stage 1)
    # families: jitter, rotate, dropout_local, dropout_global, add_local, add_global
    FULL_CORS=""
    for fam in jitter rotate dropout_local dropout_global add_local add_global; do
        for sev in 0 1 2 3 4; do
            FULL_CORS+=" ${fam}_${sev}"
        done
    done
    extract_dual_gpu "$FULL_CORS"
    $PY scripts/aggregate_p1.py \
        --feat_dir "$FEAT_DIR" \
        --reference clean \
        --output_md  reports/P1_full_drift.md \
        --output_json reports/P1_full_drift.json
    echo "[probe-p1] Stage 2 done -> reports/P1_full_drift.md"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "pollute" ]; then
    echo "[probe-p1] === STAGE 3: anchor pollution simulation ==="
    $PY scripts/anchor_pollution_sim.py \
        --feat_dir "$FEAT_DIR" \
        --output_md reports/P1_pollution_sim.md \
        --output_json reports/P1_pollution_sim.json
    echo "[probe-p1] Stage 3 done -> reports/P1_pollution_sim.md"
fi

echo "[probe-p1] DONE."
