#!/bin/bash
# ============================================================================
# MCP-3D Data + Weights Download Script
# ----------------------------------------------------------------------------
# Downloads:
#   AAAI Phase (W1-W16):
#     [P1] ModelNet-C (corruption benchmark, priority)
#     [P1] ScanObjectNN-C (corruption benchmark, priority)
#     [P1] ModelNet-40 (clean control for E+1)
#     [P1] ScanObjectNN (clean control for E+1)
#     [P1] OpenShape PointBERT-vitg14-rgb weights
#     [P1] CLIP-ViT-bigG-14-laion2B weights (used by OpenShape)
#   CVPR Phase (W17-W28, optional via env var):
#     [P2] Sim2Real-SONN
#     [P2] Objaverse-LVIS
#     [P2] ULIP-2 weights
#
# Usage:
#   bash download_data.sh                 # AAAI essentials only
#   PHASE=cvpr bash download_data.sh      # everything
#   ONLY=mc,sonn_c,weights bash ...       # selective
# ============================================================================

set -e

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/MCP-Point-Cache}"
PC_ROOT="$PROJECT_ROOT/Point-Cache"
DATA_DIR="$PC_ROOT/data"
WEIGHTS_DIR="$PC_ROOT/weights"
HF_MIRROR="https://hf-mirror.com"

mkdir -p "$DATA_DIR" "$WEIGHTS_DIR"

PHASE="${PHASE:-aaai}"
ONLY="${ONLY:-}"

# ----------------------------------------------------------------------------
# Helper: check if a target should be downloaded
# ----------------------------------------------------------------------------
should_download() {
    local key="$1"
    if [ -n "$ONLY" ]; then
        echo "$ONLY" | grep -q -w "$key" && return 0 || return 1
    fi
    return 0
}

# ----------------------------------------------------------------------------
# Helper: skip if directory already populated (>= min_files)
# ----------------------------------------------------------------------------
already_present() {
    local dir="$1"
    local min_files="${2:-1}"
    [ -d "$dir" ] && [ "$(find "$dir" -type f 2>/dev/null | head -n "$min_files" | wc -l)" -ge "$min_files" ]
}

# ----------------------------------------------------------------------------
# Datasets via existing Point-Cache python scripts
# ----------------------------------------------------------------------------

echo "==========================================================="
echo "[1/8] ModelNet-C (corruption benchmark)"
echo "==========================================================="
if should_download "mc"; then
    if already_present "$DATA_DIR/modelnet_c" 5; then
        echo "[SKIP] ModelNet-C already present at $DATA_DIR/modelnet_c"
    else
        cd "$PC_ROOT"
        python scripts/data_download_scripts/download_mc.py
    fi
else
    echo "[SKIP] ModelNet-C (not in ONLY filter)"
fi

echo "==========================================================="
echo "[2/8] ScanObjectNN-C (corruption benchmark)"
echo "==========================================================="
if should_download "sonn_c"; then
    if already_present "$DATA_DIR/sonn_c" 5; then
        echo "[SKIP] ScanObjectNN-C already present"
    else
        cd "$PC_ROOT"
        python scripts/data_download_scripts/download_sonn_c.py
    fi
else
    echo "[SKIP] ScanObjectNN-C"
fi

echo "==========================================================="
echo "[3/8] ModelNet-40 (clean, for E+1 control)"
echo "==========================================================="
if should_download "mn40"; then
    if already_present "$DATA_DIR/modelnet40" 5; then
        echo "[SKIP] ModelNet-40 already present"
    else
        cd "$PC_ROOT"
        python scripts/data_download_scripts/download_modelnet40.py
    fi
else
    echo "[SKIP] ModelNet-40"
fi

echo "==========================================================="
echo "[4/8] ScanObjectNN (clean, for E+1 control)"
echo "==========================================================="
if should_download "scanobjnn"; then
    if already_present "$DATA_DIR/scanobjnn" 5; then
        echo "[SKIP] ScanObjectNN already present"
    else
        cd "$PC_ROOT"
        python scripts/data_download_scripts/download_scanobjnn.py
    fi
else
    echo "[SKIP] ScanObjectNN"
fi

# ----------------------------------------------------------------------------
# Pre-trained weights (OpenShape PointBERT-vitg14-rgb)
# ----------------------------------------------------------------------------
echo "==========================================================="
echo "[5/8] OpenShape PointBERT-vitg14-rgb weights"
echo "==========================================================="
if should_download "openshape_weights"; then
    OPENSHAPE_DIR="$WEIGHTS_DIR/openshape/openshape-pointbert-vitg14-rgb"
    OPENSHAPE_PT="$OPENSHAPE_DIR/model.pt"
    if [ -f "$OPENSHAPE_PT" ]; then
        echo "[SKIP] OpenShape PointBERT-vitg14-rgb already at $OPENSHAPE_PT"
    else
        mkdir -p "$OPENSHAPE_DIR"
        echo "Downloading OpenShape PointBERT-vitg14-rgb model.pt (about 1.5GB)..."
        wget -c --show-progress \
            "$HF_MIRROR/OpenShape/openshape-pointbert-vitg14-rgb/resolve/main/model.pt" \
            -O "$OPENSHAPE_PT"
    fi
else
    echo "[SKIP] OpenShape weights"
fi

# ----------------------------------------------------------------------------
# Pre-trained weights (CLIP-ViT-bigG-14-laion2B for OpenShape text encoder)
# ----------------------------------------------------------------------------
echo "==========================================================="
echo "[6/8] CLIP-ViT-bigG-14-laion2B (text encoder for OpenShape)"
echo "==========================================================="
if should_download "clip_bigg"; then
    CLIP_DIR="$WEIGHTS_DIR/openshape/clip-vit-bigg-14"
    CLIP_PT="$CLIP_DIR/open_clip_pytorch_model.bin"
    if [ -f "$CLIP_PT" ]; then
        echo "[SKIP] CLIP-ViT-bigG-14 already at $CLIP_PT"
    else
        mkdir -p "$CLIP_DIR"
        echo "Downloading CLIP-ViT-bigG-14 (about 10GB, this is large; expect 10-30 min)..."
        wget -c --show-progress \
            "$HF_MIRROR/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin" \
            -O "$CLIP_PT"
    fi
else
    echo "[SKIP] CLIP-ViT-bigG-14"
fi

# ----------------------------------------------------------------------------
# CVPR phase optional downloads
# ----------------------------------------------------------------------------
if [ "$PHASE" = "cvpr" ]; then
    echo "==========================================================="
    echo "[7/8] Objaverse-LVIS (open-vocabulary, CVPR phase)"
    echo "==========================================================="
    if should_download "objaverse_lvis"; then
        if already_present "$DATA_DIR/objaverse_lvis" 5; then
            echo "[SKIP] Objaverse-LVIS already present"
        else
            cd "$PC_ROOT"
            python scripts/data_download_scripts/download_objaverse_lvis.py
        fi
    fi

    echo "==========================================================="
    echo "[8/8] ULIP-2 weights (CVPR cross-backbone)"
    echo "==========================================================="
    if should_download "ulip2_weights"; then
        ULIP_DIR="$WEIGHTS_DIR/ulip"
        ULIP_PT="$ULIP_DIR/pointbert_ulip2.pt"
        if [ -f "$ULIP_PT" ]; then
            echo "[SKIP] ULIP-2 weights already at $ULIP_PT"
        else
            mkdir -p "$ULIP_DIR"
            echo "Downloading ULIP-2 PointBERT weights..."
            wget -c --show-progress \
                "$HF_MIRROR/datasets/auniquesun/Point-PRC/resolve/main/pretrained-weights/ulip-2/pointbert_ulip2.pt" \
                -O "$ULIP_PT"
            # text encoder shared with ULIP
            wget -c --show-progress \
                "$HF_MIRROR/datasets/auniquesun/Point-PRC/resolve/main/pretrained-weights/ulip/image-text-encoder/slip_base_100ep.pt" \
                -O "$ULIP_DIR/slip_base_100ep.pt"
        fi
    fi
else
    echo ""
    echo "[INFO] PHASE=$PHASE; skipping CVPR-only downloads (Objaverse-LVIS, ULIP-2)"
    echo "[INFO] To download those later: PHASE=cvpr bash download_data.sh"
fi

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "==========================================================="
echo "Download summary"
echo "==========================================================="
du -sh "$DATA_DIR"/*/ 2>/dev/null || echo "  (no datasets yet)"
echo ""
du -sh "$WEIGHTS_DIR"/*/ 2>/dev/null || echo "  (no weights yet)"
echo ""
echo "Done. Next step:"
echo "  python generate_paraphrase.py --datasets modelnet40,scanobjectnn"
