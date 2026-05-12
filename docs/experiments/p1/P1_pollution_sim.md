# Anchor Pollution Simulation — 20260511_191549

**Question**: holding the query fixed (corrupted features), does using clean features as anchor pool beat using corrupted features as anchor pool? Delta = pure cost of anchor pollution.

## Overall 1-NN accuracy by anchor source

| cor_type | A: corrupt-as-anchor (top-1) | B: clean-as-anchor (top-1) | Δ = B − A | cos(clean_i, cor_i) mean |
|---|---|---|---|---|
| add_global_0 | 89.71% | 99.96% | +10.25pp | 0.9954 |
| add_global_1 | 89.79% | 99.96% | +10.17pp | 0.9955 |
| add_global_2 | 89.87% | 99.96% | +10.09pp | 0.9954 |
| add_global_3 | 90.15% | 100.00% | +9.85pp | 0.9954 |
| add_global_4 | 89.67% | 100.00% | +10.33pp | 0.9955 |
| add_local_0 | 89.91% | 99.96% | +10.05pp | 0.9950 |
| add_local_1 | 90.44% | 99.96% | +9.52pp | 0.9949 |
| add_local_2 | 89.79% | 99.96% | +10.17pp | 0.9949 |
| add_local_3 | 89.67% | 99.92% | +10.25pp | 0.9949 |
| add_local_4 | 90.07% | 99.96% | +9.89pp | 0.9950 |
| dropout_global_0 | 89.34% | 99.96% | +10.62pp | 0.9864 |
| dropout_global_1 | 88.82% | 99.80% | +10.98pp | 0.9761 |
| dropout_global_2 | 89.02% | 99.51% | +10.49pp | 0.9557 |
| dropout_global_3 | 87.60% | 96.96% | +9.36pp | 0.9083 |
| dropout_global_4 | 77.19% | 75.89% | -1.30pp | 0.7646 |
| dropout_local_0 | 87.84% | 97.69% | +9.85pp | 0.9437 |
| dropout_local_1 | 87.56% | 95.87% | +8.31pp | 0.9020 |
| dropout_local_2 | 83.39% | 91.53% | +8.14pp | 0.8543 |
| dropout_local_3 | 80.96% | 83.43% | +2.47pp | 0.8054 |
| dropout_local_4 | 76.86% | 70.18% | -6.69pp | 0.7485 |
| jitter_0 | 89.06% | 97.69% | +8.63pp | 0.9062 |
| jitter_1 | 86.91% | 88.09% | +1.18pp | 0.7962 |
| jitter_2 | 84.32% | 59.48% | -24.84pp | 0.6976 |
| jitter_3 | 81.93% | 30.75% | -51.18pp | 0.6201 |
| jitter_4 | 78.20% | 15.52% | -62.68pp | 0.5671 |
| rotate_0 | 89.95% | 99.92% | +9.97pp | 0.9942 |
| rotate_1 | 89.42% | 100.00% | +10.58pp | 0.9859 |
| rotate_2 | 88.41% | 99.76% | +11.35pp | 0.9606 |
| rotate_3 | 85.70% | 96.92% | +11.22pp | 0.9152 |
| rotate_4 | 83.83% | 90.19% | +6.36pp | 0.8579 |
| scale_0 | 85.25% | 97.93% | +12.68pp | 0.9501 |
| scale_1 | 84.85% | 96.88% | +12.03pp | 0.9400 |
| scale_2 | 84.44% | 95.46% | +11.02pp | 0.9306 |
| scale_3 | 83.91% | 95.06% | +11.14pp | 0.9222 |
| scale_4 | 82.70% | 93.27% | +10.58pp | 0.9145 |

## Stratified by feature drift (1/3 tertiles of cos(clean_i, cor_i))

| cor_type | tertile | accA | accB | Δ |
|---|---|---|---|---|
| add_global_0 | low-drift (top 1/3 cosine) | 90.2% | 99.9% | +9.7pp |
| add_global_0 | mid-drift | 90.2% | 100.0% | +9.8pp |
| add_global_0 | high-drift (bottom 1/3 cosine) | 88.7% | 100.0% | +11.3pp |
| add_global_1 | low-drift (top 1/3 cosine) | 89.4% | 99.9% | +10.4pp |
| add_global_1 | mid-drift | 89.5% | 100.0% | +10.5pp |
| add_global_1 | high-drift (bottom 1/3 cosine) | 90.4% | 100.0% | +9.6pp |
| add_global_2 | low-drift (top 1/3 cosine) | 90.6% | 99.9% | +9.3pp |
| add_global_2 | mid-drift | 89.7% | 100.0% | +10.3pp |
| add_global_2 | high-drift (bottom 1/3 cosine) | 89.3% | 100.0% | +10.7pp |
| add_global_3 | low-drift (top 1/3 cosine) | 90.1% | 100.0% | +9.9pp |
| add_global_3 | mid-drift | 90.5% | 100.0% | +9.5pp |
| add_global_3 | high-drift (bottom 1/3 cosine) | 89.9% | 100.0% | +10.1pp |
| add_global_4 | low-drift (top 1/3 cosine) | 90.6% | 100.0% | +9.4pp |
| add_global_4 | mid-drift | 88.8% | 100.0% | +11.2pp |
| add_global_4 | high-drift (bottom 1/3 cosine) | 89.7% | 100.0% | +10.3pp |
| add_local_0 | low-drift (top 1/3 cosine) | 90.1% | 100.0% | +9.9pp |
| add_local_0 | mid-drift | 89.6% | 99.9% | +10.3pp |
| add_local_0 | high-drift (bottom 1/3 cosine) | 90.1% | 100.0% | +9.9pp |
| add_local_1 | low-drift (top 1/3 cosine) | 90.2% | 99.9% | +9.7pp |
| add_local_1 | mid-drift | 90.0% | 100.0% | +10.0pp |
| add_local_1 | high-drift (bottom 1/3 cosine) | 91.2% | 100.0% | +8.8pp |
| add_local_2 | low-drift (top 1/3 cosine) | 89.8% | 99.9% | +10.1pp |
| add_local_2 | mid-drift | 89.1% | 100.0% | +10.9pp |
| add_local_2 | high-drift (bottom 1/3 cosine) | 90.4% | 100.0% | +9.6pp |
| add_local_3 | low-drift (top 1/3 cosine) | 89.0% | 99.8% | +10.8pp |
| add_local_3 | mid-drift | 89.1% | 100.0% | +10.9pp |
| add_local_3 | high-drift (bottom 1/3 cosine) | 90.9% | 100.0% | +9.1pp |
| add_local_4 | low-drift (top 1/3 cosine) | 89.4% | 99.9% | +10.4pp |
| add_local_4 | mid-drift | 89.3% | 100.0% | +10.7pp |
| add_local_4 | high-drift (bottom 1/3 cosine) | 91.5% | 100.0% | +8.5pp |
| dropout_global_0 | low-drift (top 1/3 cosine) | 89.8% | 100.0% | +10.2pp |
| dropout_global_0 | mid-drift | 91.1% | 100.0% | +8.9pp |
| dropout_global_0 | high-drift (bottom 1/3 cosine) | 87.1% | 99.9% | +12.8pp |
| dropout_global_1 | low-drift (top 1/3 cosine) | 91.5% | 99.6% | +8.1pp |
| dropout_global_1 | mid-drift | 88.8% | 100.0% | +11.2pp |
| dropout_global_1 | high-drift (bottom 1/3 cosine) | 86.1% | 99.8% | +13.6pp |
| dropout_global_2 | low-drift (top 1/3 cosine) | 92.9% | 99.8% | +6.9pp |
| dropout_global_2 | mid-drift | 88.2% | 99.9% | +11.7pp |
| dropout_global_2 | high-drift (bottom 1/3 cosine) | 86.0% | 98.9% | +12.9pp |
| dropout_global_3 | low-drift (top 1/3 cosine) | 93.7% | 99.9% | +6.1pp |
| dropout_global_3 | mid-drift | 87.9% | 99.8% | +11.8pp |
| dropout_global_3 | high-drift (bottom 1/3 cosine) | 81.1% | 91.2% | +10.1pp |
| dropout_global_4 | low-drift (top 1/3 cosine) | 87.0% | 97.9% | +10.9pp |
| dropout_global_4 | mid-drift | 75.5% | 85.6% | +10.0pp |
| dropout_global_4 | high-drift (bottom 1/3 cosine) | 69.1% | 43.9% | -25.2pp |
| dropout_local_0 | low-drift (top 1/3 cosine) | 88.5% | 99.9% | +11.4pp |
| dropout_local_0 | mid-drift | 87.8% | 100.0% | +12.2pp |
| dropout_local_0 | high-drift (bottom 1/3 cosine) | 87.2% | 93.1% | +5.9pp |
| dropout_local_1 | low-drift (top 1/3 cosine) | 88.6% | 99.6% | +11.0pp |
| dropout_local_1 | mid-drift | 87.6% | 99.9% | +12.3pp |
| dropout_local_1 | high-drift (bottom 1/3 cosine) | 86.5% | 88.0% | +1.5pp |
| dropout_local_2 | low-drift (top 1/3 cosine) | 86.4% | 99.4% | +13.0pp |
| dropout_local_2 | mid-drift | 82.9% | 97.6% | +14.7pp |
| dropout_local_2 | high-drift (bottom 1/3 cosine) | 80.9% | 77.4% | -3.4pp |
| dropout_local_3 | low-drift (top 1/3 cosine) | 85.8% | 99.3% | +13.5pp |
| dropout_local_3 | mid-drift | 81.0% | 93.0% | +11.9pp |
| dropout_local_3 | high-drift (bottom 1/3 cosine) | 76.1% | 57.8% | -18.3pp |
| dropout_local_4 | low-drift (top 1/3 cosine) | 79.3% | 96.1% | +16.8pp |
| dropout_local_4 | mid-drift | 76.6% | 74.3% | -2.3pp |
| dropout_local_4 | high-drift (bottom 1/3 cosine) | 74.7% | 40.0% | -34.7pp |
| jitter_0 | low-drift (top 1/3 cosine) | 90.7% | 99.9% | +9.2pp |
| jitter_0 | mid-drift | 89.9% | 99.5% | +9.7pp |
| jitter_0 | high-drift (bottom 1/3 cosine) | 86.6% | 93.6% | +7.0pp |
| jitter_1 | low-drift (top 1/3 cosine) | 88.6% | 98.0% | +9.4pp |
| jitter_1 | mid-drift | 87.9% | 93.4% | +5.5pp |
| jitter_1 | high-drift (bottom 1/3 cosine) | 84.2% | 72.6% | -11.5pp |
| jitter_2 | low-drift (top 1/3 cosine) | 85.9% | 90.9% | +5.0pp |
| jitter_2 | mid-drift | 82.9% | 64.0% | -19.0pp |
| jitter_2 | high-drift (bottom 1/3 cosine) | 84.2% | 23.4% | -60.7pp |
| jitter_3 | low-drift (top 1/3 cosine) | 78.3% | 65.2% | -13.1pp |
| jitter_3 | mid-drift | 81.0% | 22.2% | -58.8pp |
| jitter_3 | high-drift (bottom 1/3 cosine) | 86.5% | 5.2% | -81.3pp |
| jitter_4 | low-drift (top 1/3 cosine) | 74.1% | 35.6% | -38.5pp |
| jitter_4 | mid-drift | 78.3% | 9.7% | -68.6pp |
| jitter_4 | high-drift (bottom 1/3 cosine) | 82.2% | 1.5% | -80.7pp |
| rotate_0 | low-drift (top 1/3 cosine) | 90.1% | 99.9% | +9.8pp |
| rotate_0 | mid-drift | 90.6% | 99.9% | +9.3pp |
| rotate_0 | high-drift (bottom 1/3 cosine) | 89.2% | 100.0% | +10.8pp |
| rotate_1 | low-drift (top 1/3 cosine) | 89.4% | 100.0% | +10.6pp |
| rotate_1 | mid-drift | 90.7% | 100.0% | +9.3pp |
| rotate_1 | high-drift (bottom 1/3 cosine) | 88.1% | 100.0% | +11.9pp |
| rotate_2 | low-drift (top 1/3 cosine) | 89.4% | 100.0% | +10.6pp |
| rotate_2 | mid-drift | 89.0% | 99.8% | +10.7pp |
| rotate_2 | high-drift (bottom 1/3 cosine) | 86.7% | 99.5% | +12.8pp |
| rotate_3 | low-drift (top 1/3 cosine) | 89.2% | 99.9% | +10.7pp |
| rotate_3 | mid-drift | 87.5% | 100.0% | +12.5pp |
| rotate_3 | high-drift (bottom 1/3 cosine) | 80.4% | 90.8% | +10.4pp |
| rotate_4 | low-drift (top 1/3 cosine) | 88.1% | 99.9% | +11.8pp |
| rotate_4 | mid-drift | 85.4% | 98.7% | +13.2pp |
| rotate_4 | high-drift (bottom 1/3 cosine) | 77.9% | 71.8% | -6.1pp |
| scale_0 | low-drift (top 1/3 cosine) | 89.9% | 99.9% | +9.9pp |
| scale_0 | mid-drift | 85.2% | 99.6% | +14.4pp |
| scale_0 | high-drift (bottom 1/3 cosine) | 80.6% | 94.2% | +13.6pp |
| scale_1 | low-drift (top 1/3 cosine) | 86.5% | 99.9% | +13.4pp |
| scale_1 | mid-drift | 86.9% | 99.3% | +12.4pp |
| scale_1 | high-drift (bottom 1/3 cosine) | 81.1% | 91.4% | +10.3pp |
| scale_2 | low-drift (top 1/3 cosine) | 87.0% | 99.9% | +12.9pp |
| scale_2 | mid-drift | 87.4% | 98.6% | +11.2pp |
| scale_2 | high-drift (bottom 1/3 cosine) | 78.9% | 87.9% | +9.0pp |
| scale_3 | low-drift (top 1/3 cosine) | 87.2% | 99.8% | +12.5pp |
| scale_3 | mid-drift | 86.8% | 98.2% | +11.5pp |
| scale_3 | high-drift (bottom 1/3 cosine) | 77.7% | 87.1% | +9.4pp |
| scale_4 | low-drift (top 1/3 cosine) | 87.2% | 99.8% | +12.5pp |
| scale_4 | mid-drift | 84.7% | 97.9% | +13.1pp |
| scale_4 | high-drift (bottom 1/3 cosine) | 76.1% | 82.1% | +6.0pp |

## How to interpret

- Δ ≈ 0 across all corruptions  →  anchor pool source does **not** matter ⇒ pollution paradox is *not* the dominant root cause.
- Δ small (≤ +1pp) globally but +5 to +10pp on the high-drift tertile → pollution is significant *only on hard samples*, consistent with D19 §9.2 bin-level Δerr pattern.
- Δ ≥ +3pp globally → pollution is a primary root cause; **E plan** (static training-set anchor) is the right direction.
- Δ < 0 (B worse than A) on some corruption → using clean features as anchor is actively harmful on that corruption (likely jitter/dropout where stream-statistics help) ⇒ E plan needs corruption-conditional logic; **D plan** (abstention only) may be safer.
