# §3 Method (草稿 v0.1, 2026-05-12)

> **状态**：早期方法草稿。当前只落地 C1 的 D22 pivot：conditional anchor switching。
>
> **证据来源**：P1 feature drift probe 与 anchor pollution simulation，见 `docs/experiments/p1/P1_full_drift.md`、`docs/experiments/p1/P1_pollution_sim.md`、`docs/decisions/D22_p1_anchor_pollution_pivot.md`。
>
> **证据强度**：preliminary mechanism + method hypothesis。scale / jitter 上已有 oracle-level 机制证据；真实 end-to-end runner 仍待 W2.5 next step 验证。
>
> **协议约束**：clean test anchors and labeled source prototypes are diagnostic or upper-bound variants. The deployable main method under a strict source-free TTA protocol must construct stable anchors from class-text anchors, vMF text anchors, or high-confidence test-time evidence, not from clean test samples or labeled source samples.

---

## 3.1 Overview

We propose a diagnosis-driven test-time adaptation framework for 3D point-cloud recognition. Instead of assuming that every corrupted sample benefits from the same cache update rule, our method first estimates whether the current query feature remains reliable, then chooses the source of cache evidence accordingly. This design is motivated by P1, where affine-like corruptions such as scale preserve semantic features but suffer from test-stream anchor pollution, while displacement corruptions such as jitter substantially damage the feature itself.

The current C1 module is therefore reframed as **corruption-aware anchor source selection**. Its concrete mechanism is **conditional anchor switching**: the model switches among stable anchor sources, stream anchors, and abstention according to a reliability score computed at test time. Here, stable anchors refer to deployable anchors that are not continuously overwritten by uncertain test-stream predictions. Under a strict source-free TTA protocol, the main stable anchors should come from class-text anchors, vMF text anchors, or frozen high-confidence test-time evidence. Clean test anchors and labeled source prototypes are used only for diagnosis or upper-bound analysis.

---

## 3.2 Conditional Anchor Switching

Let `f_q` denote the query feature extracted from the current point cloud by the 3D encoder. The method computes a reliability score `r_q` from signals that are available at test time:

```text
entropy(q)              = uncertainty of the current prediction
max_proto_cos(q)        = max_c cos(f_q, p_c)
top1_top2_margin(q)     = cos(f_q, p_top1) - cos(f_q, p_top2)
```

Here `p_c` is a stable class anchor for class `c`. In oracle analysis, `p_c` can be approximated by a clean reference or a labeled source prototype to isolate the effect of anchor pollution. In the deployable strict source-free setting, `p_c` must instead be instantiated by a text-derived anchor, a vMF text anchor, or a carefully frozen high-confidence test-time anchor. The score does not have to be a single scalar in the final system; early experiments will compare entropy, max prototype cosine, and prototype margin as separate gates.

The anchor source selector uses three branches:

| branch | condition | anchor source | purpose |
|---|---|---|---|
| Reliable | high prototype similarity and clear margin | stable anchor source | avoid test-stream anchor pollution |
| Uncertain | middle reliability region | stream anchor | retain Point-Cache adaptation when stable anchors are not clearly safe |
| Unreliable | low prototype similarity or high uncertainty | abstention | skip cache evidence and fall back to text logits / baseline-safe output |

The rationale is directly tied to P1. On `scale_2`, features remain close to their clean counterparts (`cos mean = 0.9306`, class-consistency `95.5%`), while replacing corrupted stream anchors with clean anchors improves 1-NN top-1 accuracy from `84.44%` to `95.46%`. This does not mean that the deployable method may use clean test anchors. Rather, it shows that a non-polluted anchor source is valuable when the query feature remains semantically reliable. In contrast, on `jitter_3`, clean-anchor accuracy drops to `30.75%` while corrupt-stream-anchor accuracy remains `81.93%`, showing that even a clean oracle anchor can be harmful when the query feature has drifted away from the clean semantic manifold. Thus, the method must select the anchor source conditionally rather than using a single anchor pool for all corruptions.

---

## 3.3 Stable Anchor Sources

A stable anchor source is a fixed or slowly updated class-level reference that is protected from noisy test-stream overwrite. Unlike stream anchors, it is not blindly updated by every incoming test sample and therefore is less vulnerable to anchor pollution.

The diagnostic source-available prototype uses one labeled class prototype:

```text
p_c = normalize(mean({ f_i : y_i = c, i in training set }))
```

However, this prototype is not the strict source-free main method. It should be treated as an ablation or upper bound unless the paper explicitly opens a source-available setting. The deployable strict TTA variants are:

1. **Text anchor**: construct class anchors from class names and prompt templates.
2. **vMF text anchor**: estimate a more stable text direction from multiple prompts with uncertainty-aware concentration.
3. **Frozen high-confidence anchor**: accept only very high-confidence test samples into a protected anchor pool, then freeze or update conservatively.

The clean reference oracle is not a deployable method because it uses clean versions of test samples. It is used only to measure whether anchor source can explain the failure mechanism. Similarly, labeled source prototypes should be reported separately from strict source-free TTA results.

---

## 3.4 Stream Anchors

Stream anchors are the original Point-Cache anchors collected from the test stream. They are adaptive because they represent the current test distribution, but they can also become polluted: if an earlier sample is incorrectly classified and written into the cache, later samples may retrieve it as a misleading neighbor.

The uncertain branch keeps stream anchors because P1 does not imply that stable anchors dominate everywhere. In jitter and heavy dropout regimes, the corrupted stream may still contain useful local statistics, while clean or source-derived anchors can pull the query toward an incorrect manifold region.

---

## 3.5 Abstention

Abstention means skipping cache-based evidence for the current sample. It does not mean the model refuses to predict. The model still uses text logits or a baseline-safe output path, but it avoids adding anchor votes that are likely to amplify error.

This branch is required by the negative P1 evidence on jitter. When feature drift is severe, neither stable anchors nor polluted stream anchors should be trusted blindly. Abstention provides a safety mechanism that aims to preserve baseline performance under high-drift corruptions.

---

## 3.6 Planned Validation

The next implementation should start with a minimal runner, `Point-Cache/runners/model_with_conditional_anchor.py`, and run two smoke tests before any full benchmark:

| setting | purpose | pass condition |
|---|---|---|
| `scale_2`, 50 samples | verify that a non-polluted stable-anchor branch helps when feature remains reliable | at least hierarchical baseline |
| `jitter_3`, 50 samples | verify that the selector avoids clean/source-anchor collapse | no obvious drop below hierarchical baseline |

If both smoke tests pass, the next stage is `scale_0..4`, followed by the full 35-setting ModelNet-C evaluation.
