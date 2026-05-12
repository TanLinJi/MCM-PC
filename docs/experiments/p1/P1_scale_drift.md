# P1 feature drift probe — 20260511_190949

**reference**: `clean`  **n**: 2468  **oshape_version**: vitg14

## Summary table

| cor_type | n | cos mean | cos p25 | cos median | cos p75 | rank median | rank=1 % | rank≤5 % | class-consistent % |
|---|---|---|---|---|---|---|---|---|---|
| clean | 2468 | 1.0000 | 0.9996 | 1.0000 | 1.0003 | 1 | 100.0 | 100.0 | 100.0 |
| scale_0 | 2468 | 0.9501 | 0.9309 | 0.9627 | 0.9825 | 1 | 81.6 | 93.9 | 97.9 |
| scale_1 | 2468 | 0.9400 | 0.9143 | 0.9559 | 0.9799 | 1 | 78.3 | 92.3 | 96.9 |
| scale_2 | 2468 | 0.9306 | 0.8993 | 0.9484 | 0.9767 | 1 | 75.4 | 90.2 | 95.5 |
| scale_3 | 2468 | 0.9222 | 0.8864 | 0.9401 | 0.9744 | 1 | 73.4 | 89.1 | 95.1 |
| scale_4 | 2468 | 0.9145 | 0.8743 | 0.9362 | 0.9738 | 1 | 69.8 | 85.9 | 93.3 |

## By corruption family (severity within row)

### `scale`

| severity | cos mean | cos median | rank=1 % | rank≤5 % | class-consistent % |
|---|---|---|---|---|---|
| 0 | 0.9501 | 0.9627 | 81.6 | 93.9 | 97.9 |
| 1 | 0.9400 | 0.9559 | 78.3 | 92.3 | 96.9 |
| 2 | 0.9306 | 0.9484 | 75.4 | 90.2 | 95.5 |
| 3 | 0.9222 | 0.9401 | 73.4 | 89.1 | 95.1 |
| 4 | 0.9145 | 0.9362 | 69.8 | 85.9 | 93.3 |

## Interpretation guideline (D17 hypotheses)

| Observed | Interpretation | Implies |
|---|---|---|
| cos mean ≥ 0.90 AND rank=1 % ≥ 80 | features barely drift | D17 wrong; root cause is anchor pollution → favour **E plan** |
| cos mean 0.5–0.9 OR rank=1 % 30–80 | partial degradation | D17 partially right; **D plan** may suffice |
| cos mean < 0.5 AND rank=1 % < 30 | features severely drift | D17 right + pollution amplifies → **E plan** still preferred |

## Footnote: per-corruption ranking (worst-feature-drift first)

- `scale_4`: cos mean = 0.9145, rank=1 % = 69.8
- `scale_3`: cos mean = 0.9222, rank=1 % = 73.4
- `scale_2`: cos mean = 0.9306, rank=1 % = 75.4
- `scale_1`: cos mean = 0.9400, rank=1 % = 78.3
- `scale_0`: cos mean = 0.9501, rank=1 % = 81.6
