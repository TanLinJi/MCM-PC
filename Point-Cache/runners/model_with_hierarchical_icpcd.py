"""W2.5 P4-fast-track (D19): hierarchical TTA + raw ICP-CD geometric logits.

Goal: see whether adding a single geometric similarity term on top of the
existing Point-Cache hierarchical TTA can recover the ModelNet-C scale
negative gain (-0.40pp baseline) to >= 0pp (D17 Floor).

Scope (per D19):
  * scale corruption family only (5 severities)
  * v0.1 oracle: NO ROC/AUC gating, NO CD margin gating, NO z-score
  * raw ICP-CD additive logits with Tip-Adapter-style kernel
  * does NOT modify already-committed model_with_hierarchical_caches.py

Final logits formula (when --enable_geom_cache is on):
    final = clip_logits
          + alpha   * cache_logits          # global pos cache (feature)
          + alpha   * local_cache_logits    # local pos cache  (feature)
          - alpha   * neg_cache_logits      # neg cache        (feature)
          + alpha_g * geom_cache_logits     # NEW: ICP-CD geometric cache

Without --enable_geom_cache, behaviour is bit-identical to bar3.
"""
import os
import sys
import time
import operator

import wandb
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *  # noqa: F401, F403  (re-export load_models, get_logits, ...)


# ---------------------------------------------------------------------------
# Cache build / update with optional raw_pc storage
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_cache_in_advance(args, test_loader, lm3d_model, clip_weights,
                            shot_capacity, include_prob_map=False,
                            store_raw_pc=False):
    """Pre-fill cache with the first batch of zero-shot predictions.

    If `store_raw_pc=True`, additionally keeps the raw point-cloud xyz tensor
    (1, N, 3) on GPU as the LAST element of each cache item, for later ICP-CD.
    """
    if include_prob_map:
        print('*' * 10, 'Building [global] neg. cache ...', '*' * 10, '\n')
    else:
        print('*' * 10, 'Building [global] and [local] pos. cache ...', '*' * 10, '\n')

    cache, local_cache = {}, {}

    for pc, _, _, rgb in test_loader:
        # pc:  (1, n, 3) on CPU
        # rgb: (1, n, 3) on CPU
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(
            args, feature, lm3d_model, clip_weights
        )

        if include_prob_map:
            item = [pc_feats, loss, prob_map]
            local_item = [patch_centers, loss, prob_map]
        else:
            item = [pc_feats, loss]
            local_item = [patch_centers, loss]

        if store_raw_pc and not include_prob_map:
            # store raw xyz (no rgb) on GPU as fp32 for ICP precision
            raw_pc_gpu = pc.float().cuda().contiguous()  # (1, n, 3)
            item = item + [raw_pc_gpu]
            local_item = local_item + [raw_pc_gpu]

        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
                local_cache[pred].append(local_item)
        else:
            cache[pred] = [item]
            local_cache[pred] = [local_item]

        cache_num = sum(len(v) for v in cache.values())
        full_num = shot_capacity * clip_logits.size(1)
        if cache_num == full_num:
            if include_prob_map:
                print('*' * 10, 'Building [global] neg. cache is Done!', '*' * 10, '\n')
            else:
                print('*' * 10, 'Building [global] and [local] pos. cache is Done!', '*' * 10, '\n')
            break

    return cache, local_cache


@torch.no_grad()
def update_cache(cache, local_cache, pred, features_loss, shot_capacity,
                 include_prob_map=False, raw_pc=None):
    """Update cache, optionally storing raw_pc as the last element of each item.

    `features_loss` layout (matches model_with_hierarchical_caches.py):
        without prob_map: [pc_feats, patch_centers, loss]
        with prob_map:    [pc_feats, None,          loss, prob_map]
    """
    item = [features_loss[0]] + features_loss[2:]              # global
    local_item = [features_loss[1]] + features_loss[2:]        # local (only used when not include_prob_map)

    if raw_pc is not None and not include_prob_map:
        item = item + [raw_pc]
        local_item = local_item + [raw_pc]

    if pred in cache:
        if len(cache[pred]) < shot_capacity:
            cache[pred].append(item)
            if not include_prob_map:
                local_cache[pred].append(local_item)
        elif features_loss[2] < cache[pred][-1][1]:
            cache[pred][-1] = item
            if not include_prob_map:
                local_cache[pred][-1] = local_item

        cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        if not include_prob_map:
            local_cache[pred] = sorted(local_cache[pred], key=operator.itemgetter(1))
    else:
        cache[pred] = [item]
        if not include_prob_map:
            local_cache[pred] = [local_item]


# ---------------------------------------------------------------------------
# Original feature-based cache logits (unchanged from bar3)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_cache_logits(pc_feats, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    cache_keys, cache_values = [], []
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            cache_keys.append(item[0])
            if neg_mask_thresholds:
                cache_values.append(item[2])
            else:
                cache_values.append(class_index)

    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)

    if neg_mask_thresholds:
        cache_values = torch.cat(cache_values, dim=0)
        cache_values = (
            (cache_values > neg_mask_thresholds[0])
            & (cache_values < neg_mask_thresholds[1])
        ).half().cuda()
    else:
        cache_values = (
            F.one_hot(torch.tensor(cache_values, dtype=torch.int64),
                      num_classes=clip_weights.size(1))
        ).half().cuda()

    affinity = pc_feats @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits


@torch.no_grad()
def compute_local_cache_logits(patch_centers, local_cache, alpha, beta, clip_weights):
    keys, values = [], []
    for class_index in sorted(local_cache.keys()):
        for item in local_cache[class_index]:
            keys.append(item[0])
            n_cluster = item[0].shape[0]
            values.append([class_index] * n_cluster)

    keys = torch.cat(keys, dim=0).permute(1, 0)
    values = (
        F.one_hot(torch.tensor(values, dtype=torch.int64),
                  num_classes=clip_weights.size(1))
    ).half().cuda()
    values = values.view(-1, clip_weights.size(1))

    affinity = patch_centers.mean(dim=0, keepdim=True) @ keys
    local_cache_logits = ((-1) * (beta - beta * affinity)).exp() @ values
    return alpha * local_cache_logits


# ---------------------------------------------------------------------------
# NEW: ICP-CD geometric cache logits (D19 v0.1, raw, no gating)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_geom_cache_logits(query_pc, cache, alpha_g, beta_g, clip_weights,
                              estimate_scale=True, max_iter=20,
                              zero_mean=True, verbose=False):
    """Run batched ICP-Chamfer between query_pc and every anchor in `cache`,
    then aggregate via Tip-Adapter-style kernel into per-class logits.

    Args:
        query_pc:    (1, N, 3) FP32 on GPU
        cache:       dict[class_idx] -> list of items where item[-1] is anchor raw_pc (1, N, 3)
        alpha_g:     overall geom-logits weight in final fusion
        beta_g:      sharpness in exp(-(beta - beta * affinity))
        clip_weights:(d, n_cls)  -- only used to know n_cls
        estimate_scale: pass to pytorch3d.ops.iterative_closest_point
        max_iter:    ICP iterations cap

    Returns:
        geom_logits: (1, n_cls) FP16 on GPU (matches feature cache logits dtype)
    """
    from pytorch3d.ops import iterative_closest_point
    from pytorch3d.loss import chamfer_distance

    n_cls = clip_weights.size(1)
    device = query_pc.device

    # Flatten all anchors across classes into a single batch
    anchor_pcs = []
    anchor_class_idx = []
    for class_index in sorted(cache.keys()):
        for item in cache[class_index]:
            raw_pc = item[-1]                        # (1, N, 3)
            anchor_pcs.append(raw_pc)
            anchor_class_idx.append(class_index)

    if not anchor_pcs:
        return torch.zeros(1, n_cls, device=device, dtype=clip_weights.dtype)

    # (B, N, 3)  with B = sum_c |cache[c]|
    Y = torch.cat(anchor_pcs, dim=0).float().contiguous()      # anchors as ICP target
    B = Y.size(0)
    X = query_pc.float().expand(B, -1, -1).contiguous()        # query as ICP source

    # Batched ICP
    icp = iterative_closest_point(
        X=X, Y=Y,
        max_iterations=max_iter,
        estimate_scale=estimate_scale,
        allow_reflection=False,
    )
    aligned = icp.Xt                                            # (B, N, 3)

    # Per-sample Chamfer Distance (returns (B,) when batch_reduction=None)
    cd, _ = chamfer_distance(
        aligned, Y,
        batch_reduction=None,
        point_reduction='mean',
    )                                                           # (B,)

    # Convert CD (smaller=better) to affinity in (0, 1] (larger=better)
    affinity = torch.exp(-cd).to(clip_weights.dtype).view(1, B)  # (1, B)

    # One-hot values: (B, n_cls)
    cache_values = (
        F.one_hot(torch.tensor(anchor_class_idx, dtype=torch.int64),
                  num_classes=n_cls)
    ).to(clip_weights.dtype).to(device)

    # Tip-Adapter form, but with geometric affinity in [0, 1]
    geom_raw = ((-1) * (beta_g - beta_g * affinity)).exp() @ cache_values
    geom_raw = alpha_g * geom_raw

    if zero_mean:
        # B-plan: subtract per-sample mean so geom_logits expresses a *relative*
        # ranking vote across classes rather than a uniform positive bias.
        # This prevents geom from being a +constant that gets dominated by lcache.
        geom_logits = geom_raw - geom_raw.mean(dim=1, keepdim=True)
    else:
        geom_logits = geom_raw

    if verbose:
        # numerical diagnostics for D19 v0.1.1 debugging
        cd_min, cd_max, cd_mean = cd.min().item(), cd.max().item(), cd.mean().item()
        aff_min, aff_max, _ = affinity.min().item(), affinity.max().item(), affinity.mean().item()
        gr_min, gr_max, gr_mean = geom_raw.min().item(), geom_raw.max().item(), geom_raw.mean().item()
        gl_min, gl_max, gl_mean = geom_logits.min().item(), geom_logits.max().item(), geom_logits.mean().item()
        gl_top1_val, gl_top1_idx = geom_logits.max(dim=1)
        # margin between top-1 and top-2 geom_logits (does ICP-CD "vote" for one class?)
        top2_vals, _ = geom_logits.topk(2, dim=1)
        gl_margin = (top2_vals[0, 0] - top2_vals[0, 1]).item()
        # per-anchor affinity range (does ICP discriminate anchors?)
        _, aff_p50, _ = torch.quantile(
            affinity.float().flatten(), torch.tensor([0.1, 0.5, 0.9], device=device)
        ).tolist()
        zm_tag = "ZM" if zero_mean else "RAW"
        print(
            f"[geom-stats] cd[min/mean/max]={cd_min:.4f}/{cd_mean:.4f}/{cd_max:.4f}  "
            f"aff[min/p50/max]={aff_min:.4f}/{aff_p50:.4f}/{aff_max:.4f}  "
            f"gr[min/mean/max]={gr_min:.3f}/{gr_mean:.3f}/{gr_max:.3f}  "
            f"gl({zm_tag})[min/mean/max]={gl_min:.3f}/{gl_mean:.3f}/{gl_max:.3f}  "
            f"gl_top1=cls{gl_top1_idx.item()}={gl_top1_val.item():.3f}  "
            f"top1-top2_margin={gl_margin:.3f}"
        )

    return geom_logits


# ---------------------------------------------------------------------------
# Main TTA loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_test_tda(args, pos_cfg, neg_cfg, test_loader, lm3d_model, clip_weights):
    enable_geom = bool(getattr(args, 'enable_geom_cache', False))

    # --- pre-fill positive cache (also stores raw_pc when geom is on) ---
    pos_cache, pos_local_cache = build_cache_in_advance(
        args, test_loader, lm3d_model, clip_weights,
        pos_cfg['shot_capacity'], include_prob_map=False,
        store_raw_pc=enable_geom,
    )
    print('len(pos_cache):', len(pos_cache))
    print('len(pos_local_cache):', len(pos_local_cache))
    if enable_geom:
        n_anchors = sum(len(v) for v in pos_cache.values())
        print(f'>>> [geom] enabled, n_anchors = {n_anchors}, '
              f'alpha_g = {args.geom_alpha}, beta_g = {args.geom_beta}, '
              f'zero_mean = {getattr(args, "geom_zero_mean", True)}, '
              f'entropy_threshold = {getattr(args, "geom_entropy_threshold", 0.0)}, '
              f'estimate_scale = {args.geom_estimate_scale}')
    else:
        print('>>> [geom] disabled (bit-identical to bar3 hierarchical)')

    neg_cache, neg_local_cache = {}, {}

    pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
    if pos_enabled:
        pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
    if neg_enabled:
        neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta',
                                                'entropy_threshold', 'mask_threshold']}

    accuracies = []
    # C plan: entropy gating stats — count how many samples actually receive geom
    geom_ent_thresh = float(getattr(args, 'geom_entropy_threshold', 0.0))
    n_geom_applied = 0
    n_geom_skipped = 0
    for i, (pc, target, _, rgb) in enumerate(test_loader):
        feature = torch.cat([pc, rgb], dim=-1).half()
        pc_feats, patch_centers, clip_logits, loss, prob_map, pred = get_logits(
            args, feature, lm3d_model, clip_weights
        )
        target = target.cuda()
        prop_entropy = get_entropy(loss, clip_weights)

        # Stash raw_pc on GPU once for both cache update and geom logits
        raw_pc_gpu = pc.float().cuda().contiguous() if enable_geom else None

        if pos_enabled:
            update_cache(pos_cache, pos_local_cache, pred,
                         [pc_feats, patch_centers, loss],
                         pos_params['shot_capacity'],
                         include_prob_map=False,
                         raw_pc=raw_pc_gpu)

        if neg_enabled and (
            neg_params['entropy_threshold']['lower']
            < prop_entropy
            < neg_params['entropy_threshold']['upper']
        ):
            update_cache(neg_cache, neg_local_cache, pred,
                         [pc_feats, None, loss, prob_map],
                         neg_params['shot_capacity'],
                         include_prob_map=True)

        debug_now = enable_geom and i < int(getattr(args, 'geom_debug_steps', 0))

        # capture per-source logits for diagnostic comparison
        clip_only = clip_logits.clone()
        glob_cache_lg = None
        local_cache_lg = None
        neg_cache_lg = None

        final_logits = clip_logits.clone()
        if pos_enabled and pos_cache:
            glob_cache_lg = compute_cache_logits(
                pc_feats, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights
            )
            local_cache_lg = compute_local_cache_logits(
                patch_centers, pos_local_cache, pos_params['alpha'], pos_params['beta'], clip_weights
            )
            final_logits = final_logits + glob_cache_lg + local_cache_lg
        if neg_enabled and neg_cache:
            neg_cache_lg = compute_cache_logits(
                pc_feats, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights,
                (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']),
            )
            final_logits = final_logits - neg_cache_lg

        geom_gate_pass = False  # default for baseline runs (no geom)
        if enable_geom and pos_cache:
            # C plan: entropy gating — only apply geom when CLIP is unconfident
            geom_gate_pass = prop_entropy >= geom_ent_thresh

            if geom_gate_pass:
                n_geom_applied += 1
                if getattr(args, 'log_geom_timing', False):
                    torch.cuda.synchronize()
                    _t0 = time.perf_counter()
                geom_logits = compute_geom_cache_logits(
                    query_pc=raw_pc_gpu,
                    cache=pos_cache,
                    alpha_g=args.geom_alpha,
                    beta_g=args.geom_beta,
                    clip_weights=clip_weights,
                    estimate_scale=args.geom_estimate_scale,
                    max_iter=args.geom_max_iter,
                    zero_mean=getattr(args, 'geom_zero_mean', True),
                    verbose=debug_now,
                )
                final_no_geom = final_logits.clone()
                final_logits = final_logits + geom_logits
                if getattr(args, 'log_geom_timing', False):
                    torch.cuda.synchronize()
                    _dt = (time.perf_counter() - _t0) * 1000
                    n_anc = sum(len(v) for v in pos_cache.values())
                    print(f'[geom-timing] sample={i:5d}  n_anchors={n_anc:3d}  icpcd_ms={_dt:.1f}  ent={prop_entropy:.3f}  GATE=PASS')
            else:
                n_geom_skipped += 1
                geom_logits = None
                final_no_geom = final_logits  # alias
                if getattr(args, 'log_geom_timing', False):
                    print(f'[geom-timing] sample={i:5d}  n_anchors={"---":3}  icpcd_ms={"---":>5}  ent={prop_entropy:.3f}  GATE=SKIP')

            if debug_now:
                # report magnitudes + argmax shifts caused by geom term
                def _stats(t, name):
                    if t is None:
                        return f"{name}=None"
                    a_min, a_max, a_abs_mean = t.min().item(), t.max().item(), t.abs().mean().item()
                    argmax = t.argmax(dim=1).item()
                    return f"{name}[min/max/|mean|]={a_min:.3f}/{a_max:.3f}/{a_abs_mean:.3f} argmax={argmax}"
                pred_no_geom = final_no_geom.argmax(dim=1).item()
                pred_with_geom = final_logits.argmax(dim=1).item()
                tgt = target.item()
                gate_tag = "PASS" if geom_gate_pass else "SKIP"
                print(f"[logit-mags ] sample={i:3d} target={tgt} ent={prop_entropy:.3f} gate={gate_tag}  "
                      f"{_stats(clip_only, 'clip')}  "
                      f"{_stats(glob_cache_lg, 'gcache')}  "
                      f"{_stats(local_cache_lg, 'lcache')}  "
                      f"{_stats(neg_cache_lg, 'ncache')}  "
                      f"{_stats(geom_logits, 'geom')}")
                changed = '*** ARGMAX CHANGED ***' if pred_no_geom != pred_with_geom else 'argmax same'
                print(f"[pred-diff  ] sample={i:3d}  pred_no_geom={pred_no_geom}  "
                      f"pred_with_geom={pred_with_geom}  target={tgt}  {changed}")

        acc = cls_acc(final_logits, target)
        accuracies.append(acc)
        if args.wandb:
            wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

        if getattr(args, 'log_sample_info', False):
            # one-line per-sample dump for entropy-vs-error analysis (D19 v0.1.3)
            pred_idx = final_logits.argmax(dim=1).item()
            tgt = target.item()
            is_correct = int(pred_idx == tgt)
            if enable_geom:
                gate_str = "PASS" if geom_gate_pass else "SKIP"
            else:
                gate_str = "N/A"
            print(f"[sample-info] i={i:5d} ent={prop_entropy:.4f} pred={pred_idx:3d} target={tgt:3d} correct={is_correct} gate={gate_str}")

        if i % args.print_freq == 0:
            tag = '[hier+geom]' if enable_geom else '[hier]'
            print("---- {} step {} test acc: {:.2f}. ----\n".format(
                tag, i, sum(accuracies) / len(accuracies)
            ))

        if getattr(args, 'max_samples', -1) > 0 and (i + 1) >= args.max_samples:
            print(f'>>> [smoke] reached --max_samples={args.max_samples}, breaking.')
            break

    final = sum(accuracies) / len(accuracies)
    tag = '[hier+geom]' if enable_geom else '[hier]'
    if enable_geom:
        n_total = n_geom_applied + n_geom_skipped
        applied_pct = 100.0 * n_geom_applied / max(n_total, 1)
        print(f">>> [geom-gating] applied on {n_geom_applied}/{n_total} ({applied_pct:.1f}%) samples "
              f"(threshold={geom_ent_thresh:.3f})")
    print("---- ***Final*** {} test accuracy: {:.2f}. ----\n".format(tag, final))
    return final


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    # D19 flags are now part of utils.get_arguments() (with bar3-compatible defaults).
    args = get_arguments()

    set_random_seed(args.seed)

    config_path = args.config
    clip_model, lm3d_model = load_models(args)

    preprocess = None
    dataset_name = args.dataset
    print(f"Processing {dataset_name} dataset.")

    cfg = get_config_file(args, config_path, dataset_name)
    print("\nRunning dataset configurations:")
    print(cfg, "\n")

    test_loader, classnames, template = build_test_data_loader(
        args, dataset_name, args.data_root, preprocess
    )
    print(f'>>> classnames:', classnames)
    clip_weights = clip_classifier(args, classnames, template, clip_model)

    if args.wandb:
        if args.lm3d == 'openshape':
            prefix = f"[test-manual-prompts]/{args.cache_type}_cache_icpcd/{args.lm3d}-{args.oshape_version}"
        else:
            prefix = f"[test-manual-prompts]/{args.cache_type}_cache_icpcd/{args.lm3d}"
        run_name = f"{prefix}/{dataset_name}-{args.npoints}/{args.cor_type}/geom={int(args.enable_geom_cache)}"
        run = wandb.init(project="Point-TDA", config=cfg, name=run_name)

    acc = run_test_tda(args, cfg['positive'], cfg['negative'],
                       test_loader, lm3d_model, clip_weights)

    if args.wandb:
        wandb.log({f"{dataset_name}": acc})
        run.finish()


if __name__ == "__main__":
    main()
