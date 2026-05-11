#!/bin/bash
# D19 P4-fast-track: hierarchical + ICP-CD on ModelNet-C scale (5 severities)
#
# Runs two passes:
#   A) hier baseline   (equivalent to bar3 scale slice, reproduction sanity)
#   B) hier + ICP-CD   (new geometric cache term)
#
# Stages:
#   STAGE=smoke  : scale_2 only, --max_samples 50, single GPU, ~3-5 min
#                  Goal: confirm nothing crashes, see per-sample ICP-CD timing
#                  and rough accuracy delta on first 50 samples.
#   STAGE=full   : scale_0..scale_4 by default, both GPUs in parallel, ~90-120 min total
#                  Goal: final D19 numbers for scale 5-severity comparison.
#
# Env knobs (D19 v0.1.3+):
#   SEVERITIES   : space-separated severity ids for STAGE=full   (default: "0 1 2 3 4")
#                  e.g. SEVERITIES="2"   -> single-severity full (~20 min, ~2k samples)
#                       SEVERITIES="1 2" -> two severities only
#   MAX_SAMPLES  : --max_samples for STAGE=smoke                  (default: 50)
#
# Usage:
#   STAGE=smoke                  bash Point-Cache/scripts/eval_p4_scale_icpcd.sh
#   STAGE=full                   bash Point-Cache/scripts/eval_p4_scale_icpcd.sh
#   STAGE=full SEVERITIES="2"   bash Point-Cache/scripts/eval_p4_scale_icpcd.sh
#
# Outputs:
#   logs/p4_scale_icpcd_<STAGE>_<TS>/*.log
#   logs/p4_scale_icpcd_<STAGE>_<TS>/summary.txt

set -e
cd "$(dirname "$0")/.."

STAGE=${STAGE:-smoke}
CKPT=weights/openshape/openshape-pointbert-vitg14-rgb/model.pt
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/p4_scale_icpcd_${STAGE}_${TS}
mkdir -p "$LOG_DIR"

echo "[p4-scale-icpcd] stage  : $STAGE"
echo "[p4-scale-icpcd] log dir: $LOG_DIR"
echo "[p4-scale-icpcd] start  : $(date)"

# ---- Common runner invocation ---------------------------------------------
run_one () {
    # $1 = gpu ; $2 = cor_type ; $3 = extra flags (as single string) ; $4 = tag
    local gpu=$1
    local cor=$2
    local extra=$3
    local tag=$4
    local logfile="$LOG_DIR/${cor}__${tag}.log"
    echo "  [GPU $gpu] $tag | $cor -> $logfile"
    WANDB_MODE=offline CUDA_VISIBLE_DEVICES=$gpu \
    /root/miniconda3/envs/mcmpc/bin/python -u runners/model_with_hierarchical_icpcd.py \
        --config configs \
        --lm3d openshape \
        --cache-type hierarchical \
        --ckpt_path "$CKPT" \
        --dataset modelnet_c \
        --cor_type "$cor" \
        --npoints 1024 \
        --oshape-version vitg14 \
        --wandb-log \
        $extra \
        > "$logfile" 2>&1
}

# ---- Shared X-ray analysis function ---------------------------------------
# Used by both smoke and full stages to inspect entropy-vs-error distributions
# and per-bin geom impact. Accepts globs so full can aggregate over severities.
print_xray_block () {
    # $1 = label (e.g. "smoke" or "full-aggregate")
    # $2 = baseline log glob (e.g. "$LOG_DIR/scale_*__hier_baseline.log")
    # $3 = geom    log glob (e.g. "$LOG_DIR/scale_*__hier_plus_geom.log")
    local label=$1
    local fb_glob=$2
    local fg_glob=$3
    # bash globs expand to arrays at runtime
    local files_b=( $fb_glob )
    local files_g=( $fg_glob )
    if [ ${#files_b[@]} -eq 0 ] || [ ! -f "${files_b[0]}" ]; then
        echo "[$label] (X-ray skipped: no baseline log files match $fb_glob)"
        return
    fi
    if ! grep -qh '\[sample-info\]' "${files_b[@]}" 2>/dev/null; then
        echo "[$label] (X-ray skipped: no [sample-info] lines; --log_sample_info not enabled)"
        return
    fi
    echo ""
    echo "[$label] === [X-ray] baseline entropy distribution & error rate by bin ==="
    echo "  bin                 |  baseline n |  base_err% | geom n |  geom_err% |  Δerr (geom - base)"
    echo "  --------------------+-------------+------------+--------+------------+-------------------"
    for bin in "0.00 0.05" "0.05 0.10" "0.10 0.15" "0.15 0.20" "0.20 0.30" "0.30 1.00"; do
        set -- $bin
        local lo=$1 hi=$2
        local row_b=$(grep -h '\[sample-info\]' "${files_b[@]}" | awk -v lo="$lo" -v hi="$hi" '
          {match($0, /ent=[0-9.]+/); e=substr($0, RSTART+4, RLENGTH-4)+0;
           match($0, /correct=[0-9]/); c=substr($0, RSTART+8, 1)+0;
           if(e>=lo && e<hi) {n++; if(c==0) err++}}
          END{if(n==0){print "0 0.0"} else{printf "%d %.1f", n, 100.0*err/n}}')
        local row_g=$(grep -h '\[sample-info\]' "${files_g[@]}" | awk -v lo="$lo" -v hi="$hi" '
          {match($0, /ent=[0-9.]+/); e=substr($0, RSTART+4, RLENGTH-4)+0;
           match($0, /correct=[0-9]/); c=substr($0, RSTART+8, 1)+0;
           if(e>=lo && e<hi) {n++; if(c==0) err++}}
          END{if(n==0){print "0 0.0"} else{printf "%d %.1f", n, 100.0*err/n}}')
        local bn=$(echo $row_b | awk '{print $1}'); local be=$(echo $row_b | awk '{print $2}')
        local gn=$(echo $row_g | awk '{print $1}'); local ge=$(echo $row_g | awk '{print $2}')
        local delta=$(awk -v g="$ge" -v b="$be" 'BEGIN{printf "%+.1f", g-b}')
        printf "  [%.2f, %.2f)        |    %4d     |   %5s    |  %4d  |   %5s    |   %s\n" $lo $hi $bn $be $gn $ge $delta
    done
    echo ""
    echo "[$label] === [X-ray] aggregate baseline error rate (entropy < / >= threshold) ==="
    for thr in 0.10 0.15 0.20 0.30; do
        grep -h '\[sample-info\]' "${files_b[@]}" | awk -v thr="$thr" '
          {match($0, /ent=[0-9.]+/); e=substr($0, RSTART+4, RLENGTH-4)+0;
           match($0, /correct=[0-9]/); c=substr($0, RSTART+8, 1)+0;
           if(e<thr) {nl++; if(c==0) el++} else {nh++; if(c==0) eh++}}
          END{
            rl = (nl>0) ? 100.0*el/nl : 0
            rh = (nh>0) ? 100.0*eh/nh : 0
            printf "  threshold=%.2f:  low-ent  n=%4d err=%5.1f%%  |  high-ent n=%4d err=%5.1f%%  |  ratio=%.2fx\n", thr, nl, rl, nh, rh, (rl>0?rh/rl:0)
          }'
    done
    echo ""
    echo "[$label] === [X-ray] geom impact (gate=PASS samples only) ==="
    paste <(grep -h '\[sample-info\]' "${files_b[@]}") <(grep -h '\[sample-info\]' "${files_g[@]}") | \
      awk '
        {
          b_part = $0; g_part = $0
          if (match($0, /\[sample-info\][^\t]*\t/)) {
              b_part = substr($0, 1, RLENGTH-1)
              g_part = substr($0, RLENGTH+1)
          }
          if (match(b_part, /correct=[0-9]/)) bc = substr(b_part, RSTART+8, 1)+0
          if (match(g_part, /correct=[0-9]/)) gc = substr(g_part, RSTART+8, 1)+0
          if (match(g_part, /gate=PASS/)) {
              npass++
              if (bc==0 && gc==1) rescued++
              if (bc==1 && gc==0) broken++
          } else {
              nskip++
          }
          total++
        }
        END {
          printf "  total %d  gate_PASS=%d  gate_SKIP=%d\n", total, npass, nskip
          printf "  on PASS samples:  rescued (base wrong → geom right) = %d  broken (base right → geom wrong) = %d  net = %+d\n", rescued, broken, rescued-broken
        }'
    echo ""
    echo "[$label] === [X-ray] per-bin gate=PASS net (D19 §9.1.5 verification) ==="
    paste <(grep -h '\[sample-info\]' "${files_b[@]}") <(grep -h '\[sample-info\]' "${files_g[@]}") | \
      awk '
        BEGIN{split("0.10 0.15 0.20 0.30 1.00", edges, " "); ne=5}
        {
          b_part = $0; g_part = $0
          if (match($0, /\[sample-info\][^\t]*\t/)) {
              b_part = substr($0, 1, RLENGTH-1)
              g_part = substr($0, RLENGTH+1)
          }
          if (match(g_part, /ent=[0-9.]+/)) e = substr(g_part, RSTART+4, RLENGTH-4)+0
          if (match(b_part, /correct=[0-9]/)) bc = substr(b_part, RSTART+8, 1)+0
          if (match(g_part, /correct=[0-9]/)) gc = substr(g_part, RSTART+8, 1)+0
          if (! match(g_part, /gate=PASS/)) next
          for (i=1; i<ne; i++) {
              lo = edges[i]+0; hi = edges[i+1]+0
              if (e>=lo && e<hi) {
                  npass[i]++
                  if (bc==0 && gc==1) rescued[i]++
                  if (bc==1 && gc==0) broken[i]++
                  break
              }
          }
        }
        END {
          for (i=1; i<ne; i++) {
              n  = (npass[i]+0); r = (rescued[i]+0); br = (broken[i]+0)
              printf "  bin [%.2f, %.2f):  PASS n=%4d  rescued=%d  broken=%d  net=%+d\n", edges[i]+0, edges[i+1]+0, n, r, br, r-br
          }
        }'
}

# ---- STAGE: smoke ---------------------------------------------------------
if [ "$STAGE" = "smoke" ]; then
    MAX_SAMPLES=${MAX_SAMPLES:-50}
    echo "[smoke] scale_2 only, $MAX_SAMPLES samples each, dual-GPU parallel"
    echo "[smoke]   GPU 0 : hier_baseline"
    echo "[smoke]   GPU 1 : hier_plus_geom (with --log_geom_timing)"

    # Row A: hier baseline (geom disabled) on GPU 0
    run_one 0 "scale_2" "--max_samples $MAX_SAMPLES --log_sample_info" "hier_baseline" &
    PID0=$!

    # Row B: hier + ICP-CD (geom enabled, timing on) on GPU 1
    run_one 1 "scale_2" "--max_samples $MAX_SAMPLES --enable_geom_cache --log_geom_timing --log_sample_info" "hier_plus_geom" &
    PID1=$!

    wait $PID0 $PID1

    echo ""
    echo "[smoke] === results ==="
    for tag in hier_baseline hier_plus_geom; do
        f="$LOG_DIR/scale_2__${tag}.log"
        final=$(grep -oE 'Final\*\*\*[^:]*:[ ]*[0-9.]+' "$f" | tail -1 | grep -oE '[0-9.]+$' || echo "FAIL")
        echo "  scale_2 $tag : $final  (log: $f)"
    done
    echo ""
    echo "[smoke] === entropy gating stats (hier_plus_geom) ==="
    f="$LOG_DIR/scale_2__hier_plus_geom.log"
    grep '\[geom-gating\]' "$f" || echo "  (no gating line; gating disabled)"
    echo ""
    echo "[smoke] === ICP-CD timing stats (hier_plus_geom, GATE=PASS only) ==="
    if grep -q 'GATE=PASS' "$f"; then
        grep 'GATE=PASS' "$f" \
            | awk -F 'icpcd_ms=' '{print $2}' | awk '{print $1}' \
            | awk 'BEGIN{n=0;s=0;mn=1e9;mx=0} {n++; s+=$1; if($1<mn)mn=$1; if($1>mx)mx=$1} END{if(n>0)printf "  n=%d  mean=%.1f ms  min=%.1f ms  max=%.1f ms\n", n, s/n, mn, mx}'
    elif grep -q 'geom-timing' "$f"; then
        grep 'geom-timing' "$f" \
            | awk -F 'icpcd_ms=' '{print $2}' | awk '{print $1}' \
            | awk 'BEGIN{n=0;s=0;mn=1e9;mx=0} {n++; s+=$1; if($1<mn)mn=$1; if($1>mx)mx=$1} END{if(n>0)printf "  n=%d  mean=%.1f ms  min=%.1f ms  max=%.1f ms\n", n, s/n, mn, mx}'
    else
        echo "  (no geom-timing logs; check $f)"
    fi
    print_xray_block "smoke" "$LOG_DIR/scale_2__hier_baseline.log" "$LOG_DIR/scale_2__hier_plus_geom.log"
    echo ""
    echo "[smoke] done at $(date). If timings look reasonable, run:"
    echo "       STAGE=full bash $(basename $0)"
    exit 0
fi

# ---- STAGE: full ----------------------------------------------------------
if [ "$STAGE" = "full" ]; then
    # Allow env-driven sub-set, e.g. SEVERITIES="2" for single-severity full (D19 v0.1.3 plan)
    if [ -n "$SEVERITIES" ]; then
        # Convert space-separated string to bash array
        read -ra SEVERITIES <<< "$SEVERITIES"
    else
        SEVERITIES=(0 1 2 3 4)
    fi
    echo "[full] severities=(${SEVERITIES[*]}), both rows, dual-GPU parallel"
    echo "[full] ETA ~$(( ${#SEVERITIES[@]} * 18 )) min (~18 min/severity, dual-GPU)"


    # Build job list: 10 jobs total (5 severities x 2 rows)
    JOBS=()
    for sev in "${SEVERITIES[@]}"; do
        JOBS+=("scale_${sev}__hier_baseline")
        JOBS+=("scale_${sev}__hier_plus_geom")
    done
    N=${#JOBS[@]}

    run_job () {
        local gpu=$1 ; local key=$2
        # key form: "scale_<sev>__<tag>" with DOUBLE underscore separator.
        # cut -d_ produces an empty 3rd field on `__`, hence using bash param-expansion
        # (% / ##) to split on the literal `__` boundary. (Bug fix: previous cut version
        # gave tag="_hier_plus_geom", silently making the if-test false and dropping
        # --enable_geom_cache on the geom row → entire 2026-05-11 15:05 full run was a no-op.)
        local cor=${key%__*}                              # scale_<sev>
        local tag=${key##*__}                             # hier_baseline | hier_plus_geom
        # Always enable --log_sample_info so D19 §9.1.5 X-ray verification works on full data
        local extra="--log_sample_info"
        if [ "$tag" = "hier_plus_geom" ]; then
            extra="$extra --enable_geom_cache"
        fi
        run_one "$gpu" "$cor" "$extra" "$tag"
    }

    for ((i=0; i<N; i+=2)); do
        j=$((i+1))
        k1=${JOBS[$i]}
        echo "[batch $((i/2+1))/$((N/2))] $(date +%H:%M:%S) -- GPU0: $k1 ; GPU1: ${JOBS[$j]:-(none)}"
        run_job 0 "$k1" &
        PID0=$!
        if [ $j -lt $N ]; then
            run_job 1 "${JOBS[$j]}" &
            PID1=$!
            wait $PID0 $PID1
        else
            wait $PID0
        fi
    done

    echo ""
    echo "[full] === results table ==="
    printf "  %-10s | %-14s | %-16s | %-10s\n" "severity" "hier_baseline" "hier_plus_geom" "delta"
    echo "  -----------+----------------+------------------+-----------"
    for sev in "${SEVERITIES[@]}"; do
        fa="$LOG_DIR/scale_${sev}__hier_baseline.log"
        fb="$LOG_DIR/scale_${sev}__hier_plus_geom.log"
        a=$(grep -oE 'Final\*\*\*[^:]*:[ ]*[0-9.]+' "$fa" | tail -1 | grep -oE '[0-9.]+$' || echo "NaN")
        b=$(grep -oE 'Final\*\*\*[^:]*:[ ]*[0-9.]+' "$fb" | tail -1 | grep -oE '[0-9.]+$' || echo "NaN")
        d=$(awk -v a="$a" -v b="$b" 'BEGIN{ if(a=="NaN"||b=="NaN") print "NaN"; else printf "%+.2f", b-a }')
        printf "  %-10s | %-14s | %-16s | %-10s\n" "scale_$sev" "$a" "$b" "$d"
    done | tee "$LOG_DIR/summary.txt"

    # D19 §9.1.5 verification: aggregate X-ray over all severities
    # Use bash-glob to match every severity that ran in this STAGE=full invocation
    print_xray_block "full-aggregate" "$LOG_DIR/scale_*__hier_baseline.log" "$LOG_DIR/scale_*__hier_plus_geom.log" \
        | tee -a "$LOG_DIR/summary.txt"

    echo ""
    echo "[full] done at $(date)"
    echo "[full] summary saved to $LOG_DIR/summary.txt"
    exit 0
fi

# ---- STAGE: debug (v0.1.1 diagnostic, single card, no parallel needed) ----
if [ "$STAGE" = "debug" ]; then
    echo "[debug] scale_2, 5 samples, geom-only run, prints per-source logits magnitudes"
    # Single card sufficient: 1 job, < 30 s after warmup (per user_preferences No.10 exception)
    run_one 0 "scale_2" "--max_samples 5 --enable_geom_cache --log_geom_timing --geom_debug_steps 5" "hier_plus_geom_debug"
    f="$LOG_DIR/scale_2__hier_plus_geom_debug.log"
    echo ""
    echo "[debug] === geom-stats lines ==="
    grep '^\[geom-stats' "$f" || echo "  (none)"
    echo ""
    echo "[debug] === per-sample logit magnitudes ==="
    grep '^\[logit-mags' "$f" || echo "  (none)"
    echo ""
    echo "[debug] === argmax shifts ==="
    grep '^\[pred-diff' "$f" || echo "  (none)"
    echo ""
    echo "[debug] full log: $f"
    exit 0
fi

echo "[p4-scale-icpcd] ERROR: unknown STAGE='$STAGE' (expected: smoke | debug | full)"
exit 1
