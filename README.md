# MCM-PC: Multi-Cache Matrix for 3D Point Cloud Test-time Adaptation

> Research workspace combining **Point-Cache** (CVPR'25) and **MCP/MCP++** (ICCV'25) toward a 3D extension that we call **MCP-3D**: a 2 × 3 cache matrix (hierarchy × function) with vMF-based text anchors and ICP/Chamfer-aware geometric distance.

This repository is a working monorepo for both code and documentation of the project. Pretrained weights and benchmark data are **not** committed; use the included scripts to fetch them.

---

## Repository Layout

```
MCM-PC/
├── Point-Cache/           # fork of auniquesun/Point-Cache (upstream commit 09ba57e)
│   ├── runners/
│   │   ├── zs_infer.py            # zero-shot baseline
│   │   ├── model_with_global_cache.py
│   │   ├── model_with_hierarchical_caches.py
│   │   └── model_with_mcp3d.py    # NEW: MCP-3D method skeleton (this work)
│   ├── scripts/
│   │   └── data_download_scripts/ # NEW: per-dataset HF-mirror downloaders
│   ├── weights/                   # gitignored; populated via download scripts
│   ├── data/                      # gitignored; populated via download scripts
│   ├── clip/                      # CLIP utilities (checkpoints/ gitignored)
│   ├── configs/, datasets/, models/, llm/, utils/, ...
│   └── ...
├── MCP/                   # fork of CenturyChen/ICCV25-MCP (upstream commit f98c2da)
│   └── ...                # parked for now; focus is Point-Cache baseline
├── docs/
│   ├── MCP3D_feasibility_and_proposal.md
│   ├── MCP3D_framework.md
│   ├── MCP3D_full_proposal.md
│   ├── MCP3D_full_proposal_v2.md
│   ├── NAVIGATION.md
│   └── concepts/                  # detailed module write-ups
│       ├── 00_overview.md
│       ├── 01_vmf_anchor.md
│       ├── 02_icp_cd_distance.md
│       ├── 03_compactness_diagnosis.md
│       ├── 04_2x3_memory_matrix.md
│       └── README.md
├── figures/
│   ├── mcp3d_framework_overview.mmd
│   └── mcp3d_framework_overview.svg
├── papers/                # reference PDFs (gitignored if size matters)
├── setup_env.sh           # one-shot conda env setup (mcmpc)
├── download_data.sh       # batched dataset downloader (calls scripts/)
├── generate_paraphrase.py # LLM-based class-name paraphrase generator
├── progress.txt           # weekly milestone tracker
└── .gitignore             # unified for the whole monorepo
```

---

## Upstream Provenance

This repository was assembled from two upstream public repositories:

| Subdirectory   | Upstream                                          | Commit pinned |
|----------------|---------------------------------------------------|---------------|
| `Point-Cache/` | https://github.com/auniquesun/Point-Cache         | `09ba57e` (2026-03-13) |
| `MCP/`         | https://github.com/CenturyChen/ICCV25-MCP         | `f98c2da` (2025-11-10) |

Inner `.git` directories were removed in favor of a unified monorepo history. Run `git diff` against the original upstream commits to identify our contributions.

---

## Quick Start

### 1. Environment

```bash
bash setup_env.sh
conda activate mcmpc
```

The environment ships with PyTorch 1.12 + CUDA 11.6, `pytorch3d` 0.7.2 (cu113 wheel), `dassl`, `open_clip`, `open3d`, and the OpenAI SDK.

### 2. Data

Either run the master script:

```bash
bash download_data.sh
```

or pick individual datasets (mirrored on `hf-mirror.com`):

```bash
cd Point-Cache
python scripts/data_download_scripts/download_modelnet40.py
python scripts/data_download_scripts/download_mc.py            # ModelNet-C corruptions
python scripts/data_download_scripts/download_m40c.py          # ModelNet40-C corruptions
python scripts/data_download_scripts/download_shapenet.py      # ShapeNet-C corruptions
python scripts/data_download_scripts/download_scanobjnn.py
python scripts/data_download_scripts/download_sonn_c.py
python scripts/data_download_scripts/download_omniobject3d.py
python scripts/data_download_scripts/download_objaverse_lvis.py
```

### 3. Weights

```bash
cd Point-Cache/weights
python download_openshape_weights.py
python download_uni3d_weights.py
```

### 4. Run zero-shot baseline (smoke test)

```bash
cd Point-Cache
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
python runners/zs_infer.py \
  --config configs \
  --lm3d openshape \
  --cache-type global \
  --ckpt_path weights/openshape/openshape-pointbert-vitg14-rgb/model.pt \
  --dataset modelnet_c --cor_type add_global_2 --npoints 1024 \
  --oshape-version vitg14 \
  --wandb-log
```

Expected output: `Final Zero-shot test accuracy: 71.47` on `add_global_2` with OpenShape PointBERT-ViT-g/14.

---

## Experiment Logging (Weights & Biases)

This project uses **wandb in offline mode** as the default. All runs save metrics to `Point-Cache/wandb/offline-run-<timestamp>/` without contacting any cloud service, so no account or API key is required to develop locally.

```bash
# Always export this once per shell session (or add to your ~/.bashrc):
export WANDB_MODE=offline
```

After a run, the local artifact looks like:

```
Point-Cache/wandb/offline-run-20260509_214035-gutbtkon/
├── files/                # config, requirements, system info
├── logs/                 # debug, internal logs
└── run-gutbtkon.wandb    # binary log of all wandb.log() calls
```

To **upload** offline runs to a wandb cloud dashboard later (e.g. for sharing with collaborators):

```bash
wandb login                                           # one-time, paste API key from https://wandb.ai/authorize
wandb sync Point-Cache/wandb/offline-run-*            # batch-sync everything
```

To **switch to online mode** instead (logs stream to cloud in real time):

```bash
unset WANDB_MODE
wandb login
```

To **disable wandb entirely**, simply omit the `--wandb-log` flag from the command — the runners run identically without it.

---

## Status

- [x] **W1**: environment ready (`mcmpc` conda env, ICP+CD CUDA kernel verified)
- [x] **W1**: data 31 GB and weights 27 GB locally cached
- [ ] **W2**: Point-Cache baseline reproduction (in progress)
- [ ] **W3**: MCP-3D method implementation (vMF anchors, ICP-CD, etc.)

See `progress.txt` for the full week-by-week tracker.

---

## Documents

- **High-level proposal**: `docs/MCP3D_full_proposal_v2.md`
- **Feasibility analysis**: `docs/MCP3D_feasibility_and_proposal.md`
- **Module concepts**: `docs/concepts/00_overview.md` and following

---

## Citing Upstream Work

If you use this code, please also cite the original works:

```bibtex
@inproceedings{sun2025pointcache,
  title={Point-Cache: Test-time Dynamic and Hierarchical Cache for Robust and Generalizable Point Cloud Analysis},
  booktitle={CVPR},
  year={2025}
}

@inproceedings{chen2025mcp,
  title={Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models},
  booktitle={ICCV},
  year={2025}
}
```

---

## License

Code under `Point-Cache/` and `MCP/` retains the licenses of their respective upstream repositories. Original additions in this repository (notably `runners/model_with_mcp3d.py`, `scripts/data_download_scripts/`, all `docs/` content, `setup_env.sh`, `download_data.sh`, `generate_paraphrase.py`) are released under the same terms as the upstream Point-Cache repository unless stated otherwise.
