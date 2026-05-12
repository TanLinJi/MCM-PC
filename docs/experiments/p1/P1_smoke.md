# P1 feature drift probe — 20260511_183633

**reference**: `clean`  **n**: 100  **oshape_version**: vitg14

## Summary table

| cor_type | n | cos mean | cos p25 | cos median | cos p75 | rank median | rank=1 % | rank≤5 % | class-consistent % |
|---|---|---|---|---|---|---|---|---|---|
| clean | 100 | 1.0000 | 0.9997 | 1.0000 | 1.0005 | 1 | 100.0 | 100.0 | 100.0 |
| scale_2 | 100 | 0.9217 | 0.8872 | 0.9376 | 0.9729 | 1 | 96.0 | 99.0 | 99.0 |

## By corruption family (severity within row)

### `scale`

| severity | cos mean | cos median | rank=1 % | rank≤5 % | class-consistent % |
|---|---|---|---|---|---|
| 2 | 0.9217 | 0.9376 | 96.0 | 99.0 | 99.0 |

## Interpretation guideline (D17 hypotheses)

| Observed | Interpretation | Implies |
|---|---|---|
| cos mean ≥ 0.90 AND rank=1 % ≥ 80 | features barely drift | D17 wrong; root cause is anchor pollution → favour **E plan** |
| cos mean 0.5–0.9 OR rank=1 % 30–80 | partial degradation | D17 partially right; **D plan** may suffice |
| cos mean < 0.5 AND rank=1 % < 30 | features severely drift | D17 right + pollution amplifies → **E plan** still preferred |

## Footnote: per-corruption ranking (worst-feature-drift first)

- `scale_2`: cos mean = 0.9217, rank=1 % = 96.0
