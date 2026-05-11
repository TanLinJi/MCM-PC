"""Aggregate P1 feature drift probe outputs into a markdown report.

Reads `reports/p1_features/feat_<cor_type>.npy` and `label.npy`, then computes
per-pair (clean_i, corrupted_i) metrics:

  * cosine similarity     cos(f_clean, f_corr)
  * NN rank               rank of clean_i in NN list of corr_i over all clean
  * top-1 class consistency: argmax NN class == true label?

and per-setting aggregations (mean, median, percentiles, % rank=1, etc).

Outputs a markdown report under reports/.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--feat_dir', default='reports/p1_features')
    p.add_argument('--reference', default='clean',
                   help='cor_type to treat as the clean baseline.')
    p.add_argument('--output_md', default=None,
                   help='Output markdown path (default: reports/P1_drift_<ts>.md)')
    p.add_argument('--output_json', default=None,
                   help='Output json path (default: reports/P1_drift_<ts>.json)')
    p.add_argument('--rank_topk', type=int, default=20,
                   help='Compute %% rank <= K for K in this list.')
    return p.parse_args()


def load_feats(feat_dir, cor_types):
    feat_dir = Path(feat_dir)
    feats = {}
    for c in cor_types:
        p = feat_dir / f'feat_{c}.npy'
        if not p.exists():
            print(f'[aggregate-p1] WARN: missing {p}')
            continue
        feats[c] = np.load(str(p))
    return feats


def compute_pair_metrics(f_ref, f_cor, labels):
    """Compute per-sample drift metrics between paired (ref_i, cor_i)."""
    # cosine: ref and cor are already L2-normalized in the probe.
    cos = (f_ref * f_cor).sum(axis=-1)  # (N,)

    # NN rank of ref_i within all-ref space when queried by cor_i:
    #   sim_matrix[i, j] = <cor_i, ref_j>
    # rank of j=i in row i = number of j != i such that sim[i, j] > sim[i, i] + 1.
    # Do in chunks to bound memory.
    N = f_ref.shape[0]
    rank = np.empty(N, dtype=np.int32)
    top1_class_pred = np.empty(N, dtype=np.int64)
    chunk = 512
    for i in range(0, N, chunk):
        sims = f_cor[i:i + chunk] @ f_ref.T  # (cs, N)
        diag = sims[np.arange(sims.shape[0]), i + np.arange(sims.shape[0])]
        # rank = how many ref_j != i have sim > sim_ii
        higher = (sims > diag[:, None]).sum(axis=1)
        rank[i:i + chunk] = higher + 1  # 1-indexed: best=1
        top1 = sims.argmax(axis=1)
        top1_class_pred[i:i + chunk] = labels[top1]

    class_consistent = (top1_class_pred == labels).astype(np.int32)
    return {
        'cosine': cos,
        'rank': rank,
        'class_consistent': class_consistent,
    }


def aggregate(name, m, rank_topk_list=(1, 5, 10, 20)):
    cos = m['cosine']
    rank = m['rank']
    cc = m['class_consistent']
    out = {
        'name': name,
        'n': int(len(cos)),
        'cosine_mean': float(cos.mean()),
        'cosine_median': float(np.median(cos)),
        'cosine_p5': float(np.percentile(cos, 5)),
        'cosine_p25': float(np.percentile(cos, 25)),
        'cosine_p75': float(np.percentile(cos, 75)),
        'cosine_p95': float(np.percentile(cos, 95)),
        'rank_mean': float(rank.mean()),
        'rank_median': float(np.median(rank)),
        'class_consistency': float(cc.mean()),
    }
    for k in rank_topk_list:
        out[f'rank_le_{k}'] = float((rank <= k).mean())
    return out


def main():
    args = parse_args()
    ts = time.strftime('%Y%m%d_%H%M%S')

    feat_dir = Path(args.feat_dir)
    manifest_path = feat_dir / 'manifest.json'
    # Always discover from filesystem: manifest is unreliable when multiple
    # GPU workers write it concurrently (last-writer-wins erases earlier names).
    cor_types = sorted(p.stem.replace('feat_', '')
                       for p in feat_dir.glob('feat_*.npy'))
    manifest = None
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    if args.reference not in cor_types:
        raise SystemExit(f'reference {args.reference!r} not in {cor_types}')
    print(f'[aggregate-p1] cor_types: {cor_types}')
    print(f'[aggregate-p1] reference: {args.reference}')

    feats = load_feats(feat_dir, cor_types)
    labels = np.load(str(feat_dir / 'label.npy'))
    f_ref = feats[args.reference]

    rows = []
    pair_dump = {}
    for c in cor_types:
        if c == args.reference:
            # self-pair (sanity)
            t0 = time.time()
            m = compute_pair_metrics(f_ref, f_ref, labels)
            print(f'[aggregate-p1]   {c}: self-pair sanity '
                  f'({time.time() - t0:.1f}s)')
        else:
            t0 = time.time()
            m = compute_pair_metrics(f_ref, feats[c], labels)
            print(f'[aggregate-p1]   {c}: pair vs {args.reference} '
                  f'({time.time() - t0:.1f}s)')
        rows.append(aggregate(c, m))
        # keep cosine + rank per-sample for downstream plotting (small)
        pair_dump[c] = {
            'cosine': m['cosine'].tolist(),
            'rank': m['rank'].tolist(),
            'class_consistent': m['class_consistent'].tolist(),
        }

    # --- markdown report -----------------------------------------------------
    out_md = Path(args.output_md or f'reports/P1_drift_{ts}.md')
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f'# P1 feature drift probe — {ts}')
    lines.append('')
    lines.append(f'**reference**: `{args.reference}`  '
                 f'**n**: {rows[0]["n"]}  '
                 f'**oshape_version**: '
                 f'{(manifest or {}).get("oshape_version", "(unknown)")}')
    lines.append('')
    lines.append('## Summary table')
    lines.append('')
    header = ('| cor_type | n | cos mean | cos p25 | cos median | cos p75 | '
              'rank median | rank=1 % | rank≤5 % | class-consistent % |')
    sep = ('|---|---|---|---|---|---|---|---|---|---|')
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['n']} | "
            f"{r['cosine_mean']:.4f} | "
            f"{r['cosine_p25']:.4f} | "
            f"{r['cosine_median']:.4f} | "
            f"{r['cosine_p75']:.4f} | "
            f"{int(round(r['rank_median']))} | "
            f"{100 * r['rank_le_1']:.1f} | "
            f"{100 * r['rank_le_5']:.1f} | "
            f"{100 * r['class_consistency']:.1f} |"
        )
    lines.append('')

    # ---- Group by corruption family for easier reading
    fams = defaultdict(list)
    for r in rows:
        name = r['name']
        if name == args.reference:
            continue
        # split family from severity: trailing _<digit>
        if '_' in name:
            head, tail = name.rsplit('_', 1)
            if tail.isdigit():
                fams[head].append((int(tail), r))
                continue
        fams[name].append((0, r))

    if fams:
        lines.append('## By corruption family (severity within row)')
        lines.append('')
        for fam, items in sorted(fams.items()):
            items.sort()
            lines.append(f'### `{fam}`')
            lines.append('')
            lines.append(
                '| severity | cos mean | cos median | rank=1 % | rank≤5 % | class-consistent % |')
            lines.append('|---|---|---|---|---|---|')
            for sev, r in items:
                lines.append(
                    f"| {sev} | {r['cosine_mean']:.4f} | "
                    f"{r['cosine_median']:.4f} | "
                    f"{100 * r['rank_le_1']:.1f} | "
                    f"{100 * r['rank_le_5']:.1f} | "
                    f"{100 * r['class_consistency']:.1f} |"
                )
            lines.append('')

    # ---- Quick H1/H2/H3 interpretation
    lines.append('## Interpretation guideline (D17 hypotheses)')
    lines.append('')
    lines.append(
        '| Observed | Interpretation | Implies |\n'
        '|---|---|---|\n'
        '| cos mean ≥ 0.90 AND rank=1 % ≥ 80 | features barely drift | '
        'D17 wrong; root cause is anchor pollution → favour **E plan** |\n'
        '| cos mean 0.5–0.9 OR rank=1 % 30–80 | partial degradation | '
        'D17 partially right; **D plan** may suffice |\n'
        '| cos mean < 0.5 AND rank=1 % < 30 | features severely drift | '
        'D17 right + pollution amplifies → **E plan** still preferred |\n'
    )

    # ---- Anomaly check
    lines.append('## Footnote: per-corruption ranking (worst-feature-drift first)')
    lines.append('')
    worst = sorted([r for r in rows if r['name'] != args.reference],
                   key=lambda r: r['cosine_mean'])[:10]
    for r in worst:
        lines.append(f'- `{r["name"]}`: cos mean = {r["cosine_mean"]:.4f}, '
                     f'rank=1 % = {100 * r["rank_le_1"]:.1f}')
    lines.append('')

    out_md.write_text('\n'.join(lines))
    print(f'[aggregate-p1] markdown written: {out_md}')

    # --- json (with per-sample data for plotting) ----------------------------
    out_json = Path(args.output_json or f'reports/P1_drift_{ts}.json')
    summary_payload = {
        'timestamp': ts,
        'reference': args.reference,
        'rows': rows,
        'pair_per_sample': pair_dump,
        'manifest': manifest,
    }
    with open(out_json, 'w') as f:
        json.dump(summary_payload, f)  # no indent: per-sample is large
    print(f'[aggregate-p1] json written: {out_json}')


if __name__ == '__main__':
    main()
