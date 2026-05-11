"""P1 feature drift probe (D19 post-mortem, 2026-05-11).

Goal
----
Quantify whether PointBERT (OpenShape ViT-bigG-14) features actually degrade under
ModelNet-C corruptions, and whether *scale* is unusually severe vs. other corruption
families. This is the diagnostic that D17 / D19 *assumed* but never directly
measured. The output decides between H1 (feature failure), H2 (anchor pollution),
H3 (both).

Method
------
For each `cor_type` in args.cor_types:
  1. Load `data/modelnet_c/<cor_type>.h5` -> (N, 1024, 3) float32 + label
  2. Apply OpenShape convention: swap y/z axes, append RGB=0.4 -> (N, 1024, 6)
  3. Forward through `lm3d_model` in fp16 with batch_size B
  4. L2-normalize global features
  5. Save `feat_<cor_type>.npy` of shape (N, 1280) plus a shared `label.npy`.

Aggregation is done by a separate CPU script (`scripts/aggregate_p1.py`).

CLI
---
python runners/probe_p1_feature_drift.py \
    --cor_types clean scale_2 \
    --output_dir reports/p1_features \
    --batch_size 32 --device 0

Notes
-----
* Idempotent: skips cor_types whose feat file already exists, unless --force.
* Self-contained: does not depend on the runner / cache infrastructure.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

# Make `from models import openshape` work no matter where we cd to.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent  # Point-Cache/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models import openshape  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='P1 feature drift probe')
    parser.add_argument('--cor_types', nargs='+', required=True,
                        help="List of cor_types (h5 file basenames without .h5), "
                             "e.g. clean scale_2 jitter_3")
    parser.add_argument('--modelnet_c_root', default='data/modelnet_c',
                        help='Directory containing <cor_type>.h5 files.')
    parser.add_argument('--output_dir', default='reports/p1_features',
                        help='Where to dump feat_<cor_type>.npy and label.npy.')
    parser.add_argument('--oshape_version', default='vitg14',
                        choices=['vitg14', 'vitl14'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--npoints', type=int, default=1024)
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device id (default 0).')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing feat files.')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='If > 0, only process the first N samples '
                             '(use for pipeline smoke).')
    parser.add_argument('--openshape_config',
                        default='models/openshape/config.yaml')
    parser.add_argument('--openshape_weights_dir',
                        default='weights/openshape')
    return parser.parse_args()


def load_openshape_minimal(args):
    """Minimal OpenShape loader (mirrors utils.load_openshape but does not need
    the text encoder, since we only extract 3D features)."""
    device = f'cuda:{args.device}'
    cfg = OmegaConf.load(args.openshape_config)
    # OmegaConf.merge expects DictConfig — overrides via vars() of args.
    # `make()` in models/openshape/ppta.py expects cfg.cache_type and
    # cfg.n_cluster (not in config.yaml; usually injected from argparse).
    overrides = vars(args).copy()
    # `global` cache_type makes ppta.forward early-return the CLS token
    # without running per-sample sklearn KMeans (which forces batch_size=1
    # in the hier runner). We only need the global feature here, so this is
    # both faster *and* unlocks batched forward.
    overrides.setdefault('cache_type', 'global')
    overrides.setdefault('n_cluster', 3)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    model = openshape.create_openshape(cfg)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    ckpt_path = (Path(args.openshape_weights_dir) /
                 f'openshape-pointbert-{args.oshape_version}-rgb' / 'model.pt')
    print(f'[probe-p1] loading checkpoint {ckpt_path}')
    checkpoint = torch.load(str(ckpt_path), map_location='cpu')

    state_dict = OrderedDict()
    if args.oshape_version == 'vitg14':
        pat = re.compile('module.')
        for k, v in checkpoint['state_dict'].items():
            if re.search('module', k):
                state_dict[re.sub(pat, '', k)] = v
        model.load_state_dict(state_dict)
    else:  # vitl14
        pat = re.compile('pc_encoder.')
        for k, v in checkpoint.items():
            if re.search('pc_encoder', k):
                state_dict[re.sub(pat, '', k)] = v
        model.load_state_dict(state_dict)

    model.half().to(device)
    model.eval()
    return model, device


def prepare_input(pc_xyz):
    """Apply OpenShape convention to (N, P, 3) float32 numpy:
       1. swap y and z axes
       2. append rgb=0.4 channel
    Returns (N, P, 6) float32 numpy.
    """
    out = pc_xyz.copy()
    out[:, :, [1, 2]] = out[:, :, [2, 1]]
    rgb = np.ones_like(out, dtype=np.float32) * 0.4
    return np.concatenate([out, rgb], axis=-1).astype(np.float32)


def extract_one(cor_type, model, device, args):
    h5_path = Path(args.modelnet_c_root) / f'{cor_type}.h5'
    if not h5_path.exists():
        print(f'[probe-p1] WARN: {h5_path} not found, skipping {cor_type}')
        return None, None

    with h5py.File(str(h5_path), 'r') as f:
        data = f['data'][:].astype(np.float32)        # (N, 1024, 3)
        labels = f['label'][:].astype(np.int64).squeeze()  # (N,)

    if args.max_samples > 0:
        data = data[:args.max_samples]
        labels = labels[:args.max_samples]

    # ModelNet-C h5 files have npoints == 1024 already. If --npoints < that, truncate.
    if data.shape[1] > args.npoints:
        data = data[:, :args.npoints, :]

    feat6 = prepare_input(data)  # (N, P, 6)
    N = feat6.shape[0]
    out_dim = None
    feats = []

    bs = args.batch_size
    t0 = time.time()
    for i in range(0, N, bs):
        batch = feat6[i:i + bs]
        batch_t = torch.from_numpy(batch).half().to(device, non_blocking=True)
        xyz_t = batch_t[:, :, :3]
        with torch.no_grad():
            out = model(xyz_t, batch_t)
            # OpenShape `ppat` returns either CLS token (global) or
            # (CLS, patch_centers) tuple depending on cache_type config.
            # config.yaml does not set cache_type, so we accept either.
            if isinstance(out, tuple):
                global_feat = out[0]
            else:
                global_feat = out
            global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
        f_np = global_feat.cpu().float().numpy()
        feats.append(f_np)
        out_dim = f_np.shape[-1]
    elapsed = time.time() - t0
    feats = np.concatenate(feats, axis=0)  # (N, 1280)
    print(f'[probe-p1] {cor_type}: N={N} dim={out_dim} | '
          f'{elapsed:.1f}s ({elapsed * 1000 / N:.1f} ms/sample)')
    return feats, labels


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[probe-p1] cor_types: {args.cor_types}')
    print(f'[probe-p1] output_dir: {out_dir}')

    model, device = load_openshape_minimal(args)

    # Manifest for downstream aggregation reproducibility.
    manifest_path = out_dir / 'manifest.json'
    manifest = {
        'oshape_version': args.oshape_version,
        'npoints': args.npoints,
        'batch_size': args.batch_size,
        'modelnet_c_root': args.modelnet_c_root,
        'max_samples': args.max_samples,
        'cor_types_done': [],
    }
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    label_path = out_dir / 'label.npy'

    for cor_type in args.cor_types:
        feat_path = out_dir / f'feat_{cor_type}.npy'
        if feat_path.exists() and not args.force:
            print(f'[probe-p1] skip {cor_type} (already exists)')
            if cor_type not in manifest.get('cor_types_done', []):
                manifest.setdefault('cor_types_done', []).append(cor_type)
            continue

        feats, labels = extract_one(cor_type, model, device, args)
        if feats is None:
            continue

        np.save(str(feat_path), feats)
        if not label_path.exists():
            np.save(str(label_path), labels)
        else:
            # Sanity: labels must match across cor_types (paired by index).
            existing = np.load(str(label_path))
            if existing.shape != labels.shape or not np.array_equal(existing, labels):
                print(f'[probe-p1] WARN: labels for {cor_type} differ from {label_path}!')

        if cor_type not in manifest.get('cor_types_done', []):
            manifest.setdefault('cor_types_done', []).append(cor_type)
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    print(f'[probe-p1] done. cor_types in {out_dir}: '
          f'{sorted(manifest.get("cor_types_done", []))}')


if __name__ == '__main__':
    main()
