"""Anchor Pollution Simulation (Stage 3 of P1 probe).

This script is CPU-only and tests the anchor pollution paradox *directly in
feature space*, independent of the cache implementation. The question:

  When a sample i (with feature f_corr[i]) tries to find its 1-NN class label,
  does using *corrupted* features as the anchor pool degrade accuracy compared
  to using a *clean* anchor pool?

For each cor_type we compare:

  Setting A (test-stream-as-anchor, like hier cache): query f_corr[i],
      search in {f_corr[j] : j != i}, return labels[NN].
  Setting B (clean-as-anchor, like Static-anchor / E plan): query f_corr[i],
      search in {f_clean[j] : j != i}, return labels[NN].

Delta (B - A) on top-1 accuracy = the pure cost of anchor pollution, holding
the query side fixed.

The script also slices by self-similarity quantile (a proxy for "high-ent slice"
without needing actual logits) to check whether pollution is concentrated in
the hardest samples.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--feat_dir', default='reports/p1_features')
    p.add_argument('--reference', default='clean')
    p.add_argument('--cor_types', nargs='*', default=None,
                   help='If unset, use all cor_types in manifest except reference.')
    p.add_argument('--output_md', default='reports/P1_pollution_sim.md')
    p.add_argument('--output_json', default='reports/P1_pollution_sim.json')
    return p.parse_args()


def topk_accuracy(query, pool, labels, topk=(1,), exclude_self=True):
    """For each query[i], find top-k NN in pool by cosine similarity (features
    are L2-normalised already). Return top-k label-match accuracy.

    pool[i] is treated as the same anchor as query[i] when exclude_self=True
    (i.e., they share index space). When pool != query (e.g., clean pool vs
    corrupt query), the index correspondence still holds because both are paired
    by sample index in ModelNet-C.
    """
    N = query.shape[0]
    chunk = 512
    correct = {k: 0 for k in topk}
    for i in range(0, N, chunk):
        q = query[i:i + chunk]  # (cs, D)
        sims = q @ pool.T       # (cs, N)
        if exclude_self:
            # mask diagonal block for this chunk
            local_idx = i + np.arange(sims.shape[0])
            sims[np.arange(sims.shape[0]), local_idx] = -np.inf
        for k in topk:
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            # exact ranking inside top-k
            row_idx = np.arange(top.shape[0])[:, None]
            ordered = top[row_idx, np.argsort(-sims[row_idx, top])]
            pred = labels[ordered]                # (cs, k)
            gt = labels[i:i + sims.shape[0]][:, None]
            hit = (pred == gt).any(axis=1)
            correct[k] += int(hit.sum())
    return {k: correct[k] / N for k in topk}


def per_sample_match(query, pool, labels, exclude_self=True):
    """Boolean array: did query[i]'s top-1 NN have correct class?"""
    N = query.shape[0]
    chunk = 512
    out = np.empty(N, dtype=np.bool_)
    for i in range(0, N, chunk):
        sims = query[i:i + chunk] @ pool.T
        if exclude_self:
            local_idx = i + np.arange(sims.shape[0])
            sims[np.arange(sims.shape[0]), local_idx] = -np.inf
        nn = sims.argmax(axis=1)
        out[i:i + sims.shape[0]] = labels[nn] == labels[i:i + sims.shape[0]]
    return out


def main():
    args = parse_args()
    feat_dir = Path(args.feat_dir)

    # Always discover from filesystem (manifest can be stale due to race).
    all_cors = sorted(p.stem.replace('feat_', '')
                      for p in feat_dir.glob('feat_*.npy'))

    if args.reference not in all_cors:
        raise SystemExit(f'reference {args.reference!r} not in {all_cors}')

    if args.cor_types is None:
        cor_types = [c for c in all_cors if c != args.reference]
    else:
        cor_types = args.cor_types

    print(f'[pollution-sim] reference: {args.reference}')
    print(f'[pollution-sim] cor_types: {cor_types}')

    labels = np.load(str(feat_dir / 'label.npy'))
    f_ref = np.load(str(feat_dir / f'feat_{args.reference}.npy'))

    rows = []
    for c in cor_types:
        t0 = time.time()
        f_cor = np.load(str(feat_dir / f'feat_{c}.npy'))
        # Setting A: test-stream-as-anchor (corrupt query against corrupt pool)
        accA = topk_accuracy(f_cor, f_cor, labels, topk=(1, 5))
        matchA = per_sample_match(f_cor, f_cor, labels)
        # Setting B: clean-as-anchor (corrupt query against clean pool)
        accB = topk_accuracy(f_cor, f_ref, labels, topk=(1, 5),
                             exclude_self=False)
        matchB = per_sample_match(f_cor, f_ref, labels, exclude_self=False)

        # Self-similarity slice (proxy for "easy / hard")
        # cosine(f_cor[i], f_ref[i]) — higher means feature did not drift.
        # This is the same metric as P1 stage 1/2.
        self_sim = (f_cor * f_ref).sum(axis=-1)
        # Quantile slices.
        q33 = np.percentile(self_sim, 33)
        q67 = np.percentile(self_sim, 67)
        slc_low = self_sim < q33      # most-drifted third
        slc_mid = (self_sim >= q33) & (self_sim < q67)
        slc_hi = self_sim >= q67     # least-drifted third

        def slice_acc(mask):
            if mask.sum() == 0:
                return (None, None, None)
            a = float(matchA[mask].mean())
            b = float(matchB[mask].mean())
            return a, b, b - a

        sA = slice_acc(slc_low)
        sM = slice_acc(slc_mid)
        sH = slice_acc(slc_hi)

        row = {
            'name': c,
            'accA_top1_pct': 100 * accA[1],
            'accB_top1_pct': 100 * accB[1],
            'pollution_cost_pct': 100 * (accB[1] - accA[1]),
            'accA_top5_pct': 100 * accA[5],
            'accB_top5_pct': 100 * accB[5],
            'self_sim_mean': float(self_sim.mean()),
            'low_drift_third_accA': sH[0], 'low_drift_third_accB': sH[1],
            'low_drift_third_delta': sH[2],
            'mid_drift_third_accA': sM[0], 'mid_drift_third_accB': sM[1],
            'mid_drift_third_delta': sM[2],
            'high_drift_third_accA': sA[0], 'high_drift_third_accB': sA[1],
            'high_drift_third_delta': sA[2],
        }
        rows.append(row)
        print(f"[pollution-sim] {c}: A={row['accA_top1_pct']:.2f}% "
              f"B={row['accB_top1_pct']:.2f}% "
              f"delta={row['pollution_cost_pct']:+.2f}pp "
              f"({time.time() - t0:.1f}s)")

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f'# Anchor Pollution Simulation — {ts}')
    lines.append('')
    lines.append('**Question**: holding the query fixed (corrupted features), '
                 'does using clean features as anchor pool beat using corrupted '
                 'features as anchor pool? Delta = pure cost of anchor pollution.')
    lines.append('')
    lines.append('## Overall 1-NN accuracy by anchor source')
    lines.append('')
    lines.append('| cor_type | A: corrupt-as-anchor (top-1) | '
                 'B: clean-as-anchor (top-1) | Δ = B − A | '
                 'cos(clean_i, cor_i) mean |')
    lines.append('|---|---|---|---|---|')
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['accA_top1_pct']:.2f}% | "
            f"{r['accB_top1_pct']:.2f}% | "
            f"{r['pollution_cost_pct']:+.2f}pp | "
            f"{r['self_sim_mean']:.4f} |"
        )
    lines.append('')
    lines.append('## Stratified by feature drift (1/3 tertiles of cos(clean_i, cor_i))')
    lines.append('')
    lines.append('| cor_type | tertile | accA | accB | Δ |')
    lines.append('|---|---|---|---|---|')
    for r in rows:
        for label_, key_a, key_b, key_d in [
            ('low-drift (top 1/3 cosine)', 'low_drift_third_accA',
             'low_drift_third_accB', 'low_drift_third_delta'),
            ('mid-drift', 'mid_drift_third_accA',
             'mid_drift_third_accB', 'mid_drift_third_delta'),
            ('high-drift (bottom 1/3 cosine)', 'high_drift_third_accA',
             'high_drift_third_accB', 'high_drift_third_delta'),
        ]:
            a, b, d = r[key_a], r[key_b], r[key_d]
            if a is None:
                continue
            lines.append(f'| {r["name"]} | {label_} | '
                         f'{100 * a:.1f}% | {100 * b:.1f}% | '
                         f'{100 * d:+.1f}pp |')
    lines.append('')
    lines.append('## How to interpret')
    lines.append('')
    lines.append(
        '- Δ ≈ 0 across all corruptions  →  anchor pool source does **not** '
        'matter ⇒ pollution paradox is *not* the dominant root cause.\n'
        '- Δ small (≤ +1pp) globally but +5 to +10pp on the high-drift tertile '
        '→ pollution is significant *only on hard samples*, consistent with '
        'D19 §9.2 bin-level Δerr pattern.\n'
        '- Δ ≥ +3pp globally → pollution is a primary root cause; **E plan** '
        '(static training-set anchor) is the right direction.\n'
        '- Δ < 0 (B worse than A) on some corruption → using clean features as '
        'anchor is actively harmful on that corruption (likely jitter/dropout '
        'where stream-statistics help) ⇒ E plan needs corruption-conditional '
        'logic; **D plan** (abstention only) may be safer.')
    lines.append('')

    out_md.write_text('\n'.join(lines))
    print(f'[pollution-sim] markdown -> {out_md}')

    out_json = Path(args.output_json)
    with open(out_json, 'w') as f:
        json.dump({'timestamp': ts, 'rows': rows}, f, indent=2)
    print(f'[pollution-sim] json -> {out_json}')


if __name__ == '__main__':
    main()
