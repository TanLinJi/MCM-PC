# §1 Introduction (草稿 v0.1, 2026-05-11)

> **状态**：早期草稿。所有带 `[需补]` / `[待 P*]` 占位符的部分等具体探针实验完成后回填。
>
> **写作约束**：
> - **D16**：现实问题是 distribution shift；ModelNet-C / ScanObjectNN-C 的 corruptions 仅作为可控代理 / 压力测试。不能直接等同于现实漂移本身。
> - **D17**：diagnosis-driven TTA 是候选主卖点；在 P1 / P2 / P5 / feature-vs-geometry ROC 完成前，论文中只能写成 *preliminary observation* / *hypothesis*；证据齐备后才升级为 *contribution*。
>
> **更新触发**：W2.5 P1 / P2 / P5 完成后，回填本节的 motivation 段与 contribution C1-C3 末段。

---

## 1.1 动机段（Motivation）

3D 点云识别在自动驾驶、机器人、AR/VR 等场景中需要面对**部署时的分布偏移** (distribution shift)：传感器噪声、采样点缺失、姿态变化、尺度归一化误差等都会让训练分布之外的输入主导推理流。**测试时适配** (Test-Time Adaptation, TTA) 旨在不修改模型权重的前提下，在推理阶段动态调整模型行为。为可控地评估鲁棒性，社区构造了 ModelNet-C / ScanObjectNN-C 等鲁棒性 benchmark，将上述现实因素抽象成 jitter / dropout / rotate / scale 等手工 corruption families，作为代理任务和压力测试。

近期免训练 TTA 方法 (Point-Cache [需补 ref], TDA [需补 ref], DMN [需补 ref]) 通过缓存历史样本特征实现快速适配，并在 ModelNet-C 等 benchmark 上取得了较强基线 (Point-Cache hierarchical 在 35-setting 平均上达到 76.59% [需补 ref])。然而我们在复现实验中观察到一个未被以往工作单独讨论的 benchmark 现象 (本文 §4.1, F1)：

> **在 ModelNet-C 的 scale corruption family 下，Point-Cache hierarchical TTA 反而比零样本基线退化 -0.40pp** (5 个 severity 中 4 个为负增益)。

我们将这一现象称为 *negative adaptation under controlled corruption*——TTA 不仅没能弥补分布偏移，反而在某些 stress test 下加重了误差。需要强调的是：本论文将该结果定位为 benchmark 压力测试中的失败案例，而非对现实世界尺度漂移的直接等价声明。

---

## 1.2 我们的视角：diagnosis-driven TTA

现有许多 TTA / cache 方法主要从「提出新模块以提升精度」的角度组织叙事。本文采取一个不同的视角：**先从根因上诊断当前 SOTA 方法在哪些 benchmark corruption 下、为什么会出现 negative adaptation，再据此设计修复机制**。

> **We provide a diagnosis-driven analysis of negative adaptation in cache-based 3D TTA, and use the diagnosis to motivate a geometry-aware multi-memory design.**

具体而言，我们围绕 cache-based TTA 的三个关键决策面分别构造诊断：

1. **特征侧诊断**：在受控 corruption 下，3D-VLM (Vision-Language Model) 的 feature distance 是否仍然可靠？我们用 *feature distance failure probe* (P1) 检验同物体 clean-corrupted pair 的余弦相似度与最近邻 rank 是否退化 [待 P1]。
2. **方法共性诊断**：scale 退化是 hierarchical 缓存的特例，还是 cache-family 的共性？我们通过 *cross-method probe* (P5) 比较 Zero-Shot / Global-Cache / Hierarchical-Cache 在 scale corruption 下的逐 severity 表现 [待 P5]。
3. **记忆侧诊断**：测试流是否会让 memory 自身漂移？我们离线度量 memory **紧致性** (compactness, Φ(c)) 与 per-class 精度的相关性 (Pearson r) [待 P2]。

诊断结果指引我们设计**几何感知的多记忆框架** (geometry-aware multi-memory framework)，针对三类失败模式分别引入正交修复信号。

### 1.2.1 Preliminary observation (D19 fast-track, 2026-05-11)

[*preliminary*；P1 / P5 / feature-vs-geometry ROC 证据齐备前不升级为 contribution claim，遵循 D17 / 写作约束 §1 顶部第二条]

作为 §1.2 第一项 (特征侧诊断) 的初步证据，我们在 ModelNet-C `scale_2` 全集 (n=2468) 上观测了 Point-Cache hierarchical baseline 的 *entropy distribution vs. baseline error rate*：

| 阈值 | low-entropy err% | high-entropy err% | ratio |
|---|---|---|---|
| 0.10 | 6.1 (n=1568) | **49.2 (n=900)** | **8.04x** |
| 0.15 | 11.4 (n=1850) | 53.1 (n=618) | 4.65x |
| 0.20 | 14.4 (n=2051) | 58.5 (n=417) | 4.07x |
| 0.30 | 18.6 (n=2308) | 68.8 (n=160) | 3.70x |

进一步按 entropy bin 切片 (bin width = 0.05–0.10)，**baseline error rate 在六个 bin 上严格单调递增** (2.9% → 21.1% → 40.8% → 41.8% → 52.1% → 68.8%)。

这一观察表明：在 controlled corruption 下，3D-VLM 的 prediction entropy 是 baseline 错误率的强单调代理。该结论提供两条路径价值：

1. *Diagnostic*：它把 baseline 的失败样本集中地暴露在 high-entropy region，这是 §1.2 三个诊断的统一 entry point——P1 可以重点采样 high-ent slice 检验 feature 是否还可靠；P5 可以按 entropy 比较多种 cache 方法在同一 slice 的表现；feature-vs-geometry ROC 也可以在 high-ent slice 上区分 feature 与 geometry 信号。
2. *Architectural*：它支持以 entropy 为 gate 切分 "low-ent trust feature / high-ent invoke alternate evidence" 的双轨设计 (entropy-conditional fusion)，这是 §3 中 C1 ICP-CD 几何信号融合策略的形式根据。

**注意事项 (D16)**：此 ratio 来自 `scale_2` 单 corruption family 的 single severity，**不能**直接外推到现实世界尺度漂移；扩展到 multi-corruption / multi-method 是 §4 实验侧的工作 (W4 / W6)。

**对应实验源**：`docs/decisions/D19_design_rationale.md` §9.2.2；原始 log 位于 `Point-Cache/logs/p4_scale_icpcd_full_20260511_150549/`。

[*figure placeholder F-prelim-1*]: bin-wise baseline error rate bar chart with 95% CI；x = entropy bin, y = err%；overlay scatter of per-sample (entropy, correct) for visual density.
[*table placeholder T-prelim-1*]: 上述 4-行 ratio table 的扩展版（在 W4 主实验后扩到 7 corruption × 5 severity）。

### 1.2.2 Preliminary observation: feature drift vs. anchor pollution (P1 probe, 2026-05-11)

[*preliminary*；与 §1.2.1 的 entropy ratio 同档证据，作为 §1.2 第一项与第二项的直接量化基础]

§1.2 第一项的诊断需要回答：当 ModelNet-C corruption 让 hier baseline 倒退时，下游错误是来自 **PointBERT feature 本身退化**（H1），还是 **test-stream-as-anchor 池被错预测污染**（H2）？我们在 ModelNet-C 全集 (n=2468) 上提取每个 sample 的 PointBERT global feature，按对齐 index 与 clean reference 配对，得到两组 paired-sample 指标：

- **Feature drift**：`cos(f_clean[i], f_corr[i])` 的均值与 NN-rank 分布。
- **Anchor pollution Δ**：1-NN top-1 acc 在 *corrupt-as-anchor*（A）vs *clean-as-anchor*（B）两种 anchor 池下的差值，holding query 不变。Δ = B − A 即 "纯 pollution 代价"。

**关键观测**（节选自 `docs/experiments/p1/P1_full_drift.md` + `P1_pollution_sim.md`）：

| corruption family @ sev=2 | cos mean | class-cons % | Δ pollution (pp) |
|---|---|---|---|
| add_global / add_local | ≈ 0.995 | ≈ 100 | +10.1 |
| rotate | 0.961 | 99.8 | +11.4 |
| dropout_global | 0.956 | 99.5 | +10.5 |
| **scale** | **0.931** | **95.5** | **+11.0** |
| dropout_local | 0.854 | 91.5 | +8.1 |
| jitter | 0.698 | 59.5 | **−24.8** |

两个对立 regime：

1. **Affine-like corruption (scale / rotate / add\_\*)**：feature **几乎不漂** (cos > 0.93, class-consistency > 95%)，但 anchor pool 切换给出 **+10 ~ +13pp** 的纯增益。在 *scale* 这一最受关注的 family 上，**H1 被 falsify、H2 被 confirm**。这与 §1.2.1 entropy 单调性互补：entropy 切片告诉我们 baseline 错误集中在 high-ent，P1 进一步告诉我们这些 high-ent 错误的 root cause 是 anchor pollution 而非 feature failure。
2. **Displacement corruption (jitter / 重度 dropout)**：feature 大幅退化 (cos < 0.70, class-consistency < 60%)，且 clean anchor 反而是 **有害** 的 (Δ ≤ −24.8pp)。在这些 corruption 上 H1+H2 同时成立 (H3)，且 anchor switching 不是有效解药。

**对方法设计的影响**：§1.3 C1 原本以 "geometry-as-feature-backup" 为 framing；P1 数据表明这只在 displacement family 上 mechanistically reasonable。在 affine family 上，更直接的 remedy 是 **anchor source switching**：把 1-NN 池从易污染的 test stream 切到更稳定的 anchor source。clean reference 和 labeled source prototype 只提供 oracle / upper-bound 证据；strict source-free 主方法需要由 text / vMF anchor 或高置信测试时证据实现该 stable anchor source。

**对应实验源**：`docs/decisions/D22_p1_anchor_pollution_pivot.md` §3-§7；原始数据 `docs/experiments/p1/P1_{scale,full}_drift.md` 与 `P1_pollution_sim.md`。

**注意事项 (D16 / protocol)**：表中 +Δ 是 *oracle simulation*，用于隔离 anchor pollution 机制；它不表示 strict TTA 方法可以访问 clean test samples。若使用 labeled source prototypes，也必须作为 source-available ablation 或 upper bound 单独报告。主方法在 strict source-free TTA 协议下应使用 text / vMF anchors 或高置信测试时证据构造 stable anchor source。

[*figure placeholder F-prelim-2*]: scatter — x = `cos(f_clean, f_corr)`, y = pollution Δ；按 corruption family 着色；W4 主实验前用 `P1_*.json` 中 per-sample 数据出图。

---

## 1.3 主要贡献（Contributions）

我们的贡献既包括方法模块，也包括方法之上的 *diagnostic framing*：

- **C1（核心方法）：corruption-aware anchor source selection。**  
  P1 诊断显示，ModelNet-C scale / rotate / add\_\* 等 affine-like corruption 下的主要瓶颈不是 feature failure，而是 test-stream anchor pollution；相反，jitter / heavy dropout 下 feature 本身会显著漂移，盲目使用 clean/source anchor 会造成严重负迁移。因此我们将 C1 从原先的 "geometry-as-feature-backup" 重构为 **conditional anchor switching**：根据样本可靠性在 stable anchor source、stream anchor 与 abstention 之间切换。clean anchor 与 labeled source prototype 只作为诊断或上界；strict source-free TTA 主方法使用 text / vMF anchors 或高置信测试时证据构造稳定锚点 (本文 §3.4)。

- **C2（数学严谨化）：vMF 文本锚点。**  
  在理想单峰假设下，Point-Cache 的 `normalize(Σ_i t_i)` 与 vMF mean direction 的 MLE 等价；但它是一个无先验、无不确定性建模的点估计，对 prompt 噪声和多语义模式脆弱。我们以 von Mises-Fisher 分布的 Maximum a Posteriori (MAP) 估计构造文本锚点，引入 prompt concentration κ 与先验 anchor 的 Bayesian shrinkage，使锚点在 noisy / mixed prompt distributions 下更稳健 (本文 §3.3)。

- **C3（系统整合）：2×3 多记忆矩阵。**  
  把 confidence / compactness / boundary 三类功能与 global / local 两类表征正交组合为 6 个独立 memory cell，并以 z-score 归一化做跨格融合。boundary cell 用于推理 logits 的 calibration（不是训练时的 negative sample），其位置与权重通过 leave-one-out 消融决定 (本文 §3.5, §4.4)。

- **C4（科学发现）：3D 紧致性诊断。**  
  我们在 3D TTA 中复现并扩展了 2D MCP [需补 ref] 中关于 memory compactness 与精度强相关的发现。该诊断为离线分析工具，不参与无监督推理；其作用是解释 memory drift 与 negative adaptation 之间的机制关联 (本文 §4.5)。

> **整体贡献串联**：先用诊断说明「为何会有 negative adaptation」，再用 C1 / C2 / C3 / C4 构成一条「文本锚点 → 几何证据 → 多记忆融合 → 离线机制解释」的修复链路。

---

## 1.4 结果与定位（Results, Preview）

实验上，我们以 OpenShape PointBERT-ViT-g/14 [需补 ref] 为主干，在 ModelNet-C 和 ScanObjectNN-C 上评估 35-setting 平均与逐 corruption family 表现。性能目标依据 D 决策表 (`docs/proposals/MCP3D_feasibility_and_proposal.md` 附录) 拆为三档：

- **Floor**：35-mean ≥ +0.5pp，scale 列 ≥ 0pp（即把 negative adaptation 拉回非负）。
- **Target**：35-mean +1~+3pp，scale 列 +1~+3pp（主卖点档）。
- **Stretch**：35-mean > +3pp，scale 列 > +5pp（范式级改进 / Oral 候选）。

[待 W4 / W6 主实验回填具体数字]

---

## 1.5 本节关键约束（写作 guardrail）

- 不能把 jitter / dropout / rotate / scale 直接等同为现实世界分布偏移；它们是 benchmark corruption families，是诊断工具 (D16)。
- 不能宣称 Point-Cache 的普通平均在数学上错误；只能说它是无先验、无不确定性建模的点估计，在 noisy / mixed prompt 下脆弱 (G2)。
- 不能宣称 ICP-CD 在所有 corruption 上都优于 feature distance；只能在 P1 / 跨类 ROC 提供证据的区域使用 (G1, C1 安全门控)。
- diagnosis-driven 卖点在 P1 / P2 / P5 / ROC 证据齐备前为 *preliminary observation*；齐备后才升级为正式 contribution (D17)。

---

## 1.6 待办

- [x] ~~W2.5 P1 / P2 完成后回填 1.1 数字与 1.2 第一项 / 第三项的诊断结论~~ (P1 已完成于 §1.2.2)
- [ ] W2.5 P5 完成后回填 1.2 第二项的跨方法对比图
- [ ] **§1.2.1 ratio table 扩展到 7 corruption × 5 severity** (W4 主实验后)，把 preliminary 提升为 supporting evidence
- [ ] **§1.2.1 figure placeholder F-prelim-1**：W4 后用 matplotlib 出图
- [ ] **§1.2.2 figure placeholder F-prelim-2**：用 `Point-Cache/reports/P1_*.json` per-sample 数据出 cos vs Δ-pollution 散点图
- [x] **§1.3 C1 reframe**：基于 §1.2.2 数据，把 C1 从 "geometry-as-feature-backup" 重新表述为 "corruption-aware anchor source selection"（D22；详见 `docs/decisions/D22_p1_anchor_pollution_pivot.md` §6-§7）
- [ ] W4 oracle 完成后回填 1.4 的预览数字与三档判定
- [ ] 与 §2 Related Work / §3 Method / §4 Experiments 的术语对齐 (anchor / cache cell / corruption family / negative adaptation)
- [ ] 7 处 [需补 ref] 引用补全
