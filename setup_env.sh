#!/bin/bash
# ============================================================================
# MCP-3D Environment Setup Script
# ----------------------------------------------------------------------------
# Target: 2x T4 GPU (16GB each), CUDA 11.6, Linux
# Base: Point-Cache compatible env + MCP-3D specific additions
#   * ICP registration: pytorch3d
#   * Chamfer distance: pytorch3d (chamfer_distance) or chamferdist (fallback)
#   * Fallback ICP: open3d
#   * LLM paraphrase: openai-style API (DeepSeek)
# ============================================================================

set -e  # exit on any failure

ENV_NAME="${MCP3D_ENV:-mcmpc}"
PYTHON_VERSION="3.9"  # matching user-created mcmpc env; Point-Cache officially uses 3.8.16 but 3.9 is compatible
TORCH_VERSION="1.12.0"
CUDA_VERSION="cu116"

echo "==========================================================="
echo "[1/7] Creating conda environment: $ENV_NAME (python $PYTHON_VERSION)"
echo "==========================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Create env if not exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[INFO] Environment '$ENV_NAME' already exists. Skipping creation."
    echo "[INFO] To recreate, run: conda env remove -n $ENV_NAME"
else
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# Activate env (works in both interactive shells and scripts)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==========================================================="
echo "[2/7] Installing PyTorch $TORCH_VERSION + CUDA $CUDA_VERSION"
echo "==========================================================="
pip install \
    torch==1.12.0+cu116 \
    torchvision==0.13.0+cu116 \
    torchaudio==0.12.0 \
    --extra-index-url https://download.pytorch.org/whl/cu116

echo "==========================================================="
echo "[3/7] Installing Point-Cache core dependencies"
echo "==========================================================="
pip install \
    timm==0.9.16 \
    einops==0.7.0 \
    omegaconf==2.3.0 \
    easydict==1.13 \
    pyyaml \
    h5py==3.10.0 \
    plyfile==1.0.3 \
    scipy \
    scikit-learn \
    matplotlib \
    pandas \
    tqdm \
    wandb \
    tensorboard

echo "==========================================================="
echo "[4/7] Installing CLIP / OpenCLIP for text encoders"
echo "==========================================================="
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
pip install open_clip_torch==2.24.0

echo "==========================================================="
echo "[5/7] Installing MCP-3D specific dependencies"
echo "==========================================================="
# pytorch3d for ICP and Chamfer distance (with prebuilt wheel for torch 1.12 + cu116)
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1120/download.html \
    || (echo "[WARN] pytorch3d prebuilt wheel failed, falling back to source build"; \
        pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4")

# fallback CD library if pytorch3d.chamfer is too slow
pip install chamferdist || echo "[WARN] chamferdist install failed (optional, will use pytorch3d)"

# open3d as ICP fallback library
pip install open3d==0.17.0 || echo "[WARN] open3d install failed (optional, used only if pytorch3d ICP fails)"

# DeepSeek API (uses openai-compatible interface)
pip install openai==1.30.0

# Sklearn for PCA principal axis pre-alignment
pip install scikit-learn

echo "==========================================================="
echo "[6/7] Installing dassl (Point-Cache dependency)"
echo "==========================================================="
DASSL_DIR="${DASSL_DIR:-/root/autodl-tmp/dassl}"
if [ ! -d "$DASSL_DIR" ]; then
    git clone https://github.com/auniquesun/dassl.git "$DASSL_DIR"
fi
cd "$DASSL_DIR"
pip install -r requirements.txt 2>/dev/null || true
python setup.py develop
cd -

echo "==========================================================="
echo "[7/7] Verification"
echo "==========================================================="
python -c "
import torch
import torchvision
print(f'torch: {torch.__version__}')
print(f'torchvision: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GB')
"

python -c "
import sys
def check(name, import_str):
    try:
        exec(import_str)
        print(f'[OK] {name}')
    except Exception as e:
        print(f'[FAIL] {name}: {e}')
        sys.exit(1)

check('clip', 'import clip')
check('open_clip', 'import open_clip')
check('timm', 'import timm')
check('einops', 'import einops')
check('omegaconf', 'from omegaconf import OmegaConf')
check('pytorch3d', 'from pytorch3d.ops import iterative_closest_point, knn_points')
check('pytorch3d.chamfer', 'from pytorch3d.loss import chamfer_distance')
check('openai', 'import openai')
check('sklearn', 'from sklearn.decomposition import PCA')
print()
print('All MCP-3D dependencies verified successfully.')
"

echo ""
echo "==========================================================="
echo "Environment '$ENV_NAME' is ready."
echo "==========================================================="
echo "Next steps:"
echo "  1. conda activate $ENV_NAME"
echo "  2. bash download_data.sh           # download datasets and weights"
echo "  3. python generate_paraphrase.py   # generate DeepSeek paraphrases"
echo ""
