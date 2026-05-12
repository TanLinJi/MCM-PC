# MCP-3D Project Navigation

This is the current map for `/root/autodl-tmp/MCM-PC`.

## 1. Where Documents Live

| Category | Directory | Purpose |
|---|---|---|
| Project management | `docs/project/` | SOP, navigation, progress tracker |
| Research proposals | `docs/proposals/` | historical and current proposal drafts |
| Decisions / post-mortems | `docs/decisions/` | locked decisions, failed hypotheses, pivots |
| Concept learning | `docs/concepts/` | explanations for vMF, ICP-CD, compactness, memory matrix |
| Paper drafting | `docs/paper/` | paper outline and manuscript sections |
| Experiment reports | `docs/experiments/` | human-readable summaries of results |
| HTML reports | `docs/reports/` | self-contained visual reports |
| Conversation context | `docs/context/windsurf/` | recovery notes, next steps, user preferences |
| Reference papers | `docs/references/papers/` | PDF references |
| Figures | `docs/assets/figures/` | diagrams and figure sources |

See `docs/README.md` for the placement rules for future documents.

## 2. Current Status

The project has completed W1 and W2. The Point-Cache OpenShape baseline has
been reproduced:

| Result | Ours | Paper | Status |
|---|---:|---:|---|
| Clean ModelNet40 zero-shot | 83.27 | 84.56 | within tolerance |
| ModelNet-C zero-shot 35-mean | 72.51 | 73.49 | within tolerance |
| ModelNet-C hierarchical TTA 35-mean | 75.27 | 76.59 | within tolerance |

The main W2 summary is `docs/experiments/fig1a_summary.md`.

W2.5 diagnostics have started. P1 is complete and is currently the most
important evidence:

- `docs/experiments/p1/P1_full_drift.md`
- `docs/experiments/p1/P1_pollution_sim.md`
- `docs/decisions/D20_p1_post_mortem.md`

The raw ICP-CD fast-track route was tested and rejected for `scale_2`; the next
method direction is conditional anchor switching rather than raw additive
ICP-CD logits.

## 3. Recommended Reading Order

For a quick recovery:

1. `docs/README.md`
2. `docs/project/progress.txt`
3. `docs/experiments/fig1a_summary.md`
4. `docs/decisions/D20_p1_post_mortem.md`
5. `docs/paper/01_introduction.md`

For concept learning:

1. `docs/concepts/00_overview.md`
2. `docs/concepts/01_vmf_anchor.md`
3. `docs/concepts/02_icp_cd_distance.md`
4. `docs/concepts/03_compactness_diagnosis.md`
5. `docs/concepts/04_2x3_memory_matrix.md`
6. `docs/concepts/05_mcp_three_caches.html`

For implementation:

1. `Point-Cache/runners/zs_infer.py`
2. `Point-Cache/runners/model_with_hierarchical_caches.py`
3. `Point-Cache/runners/model_with_hierarchical_icpcd.py`
4. `Point-Cache/runners/probe_p1_feature_drift.py`
5. `Point-Cache/runners/model_with_mcp3d.py` (skeleton; not the current working method)

## 4. Next Work Item

Before adding the next method implementation, sync the D20 pivot into:

- `docs/context/windsurf/decisions.md`
- `docs/context/windsurf/key_findings.md`
- `docs/context/windsurf/next_steps.md`
- `docs/project/progress.txt`

Then implement the new conditional-anchor runner and smoke-test it on `scale_2`
and `jitter_3`.
