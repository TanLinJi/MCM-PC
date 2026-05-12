# MCP-3D 原稿漏洞日志

> **用途**：在用户复习 `MCM-PC/docs/*` 期间，每发现一个逻辑/证据/实验/写作漏洞就追加一条。
>
> **原则**：
> - 当场记录，**不打断用户提问节奏**
> - 每条必须给"出处（文件 + 行号）+ 漏洞类型 + 严重度 + 修补方向"
> - 修补方向要可执行（具体到加哪段实验/改哪段叙事）
> - 漏洞修复后**移到下方"已修复"区**，不删除（保留思维演化轨迹）

---

## 漏洞类型分类

| 代号 | 类型 | 说明 |
|---|---|---|
| **L** | Logic | 推理跳跃 / 因果不成立 / 自相矛盾 |
| **E** | Evidence | 假设无实验/文献支撑 |
| **X** | eXperiment | 实验设计不足以支撑结论 |
| **W** | Writing | 表达模糊 / 定义不清 / 概念混用 |
| **S** | Scope | 范围过大或过小 / 工作量与产出不匹配 |
| **C** | Comparison | 与 baseline / 相关工作的区分度不足 |

## 严重度

- 🔴 **Blocker**：不修补论文必被拒（审稿人秒挑）
- 🟡 **Major**：影响说服力但可在 rebuttal 救回
- 🟢 **Minor**：润色级，影响读者体验不影响接收

---

## 待修复漏洞

### G1 — C1 (ICP-CD) "几何距离比特征余弦更稳" 是假设无证据

- **类型**：E (Evidence) | **严重度**：🔴 Blocker
- **出处**：`docs/concepts/02_icp_cd_distance.md` 全篇 + `docs/proposals/MCP3D_full_proposal_v2.md` 提到 C1 motivation 段
- **问题**：原稿用"直觉论证"——"特征空间是软的，几何空间是硬的"——但没有数据证明 feature distance 在 benchmark corruption 下确实失败，也没有证明 ICP-CD 能恢复 Point-Cache 的退化。万一 feature 并未失败，或 ICP-CD 在相似异类上也失败，整个 C1 motivation 倒塌。
- **修补方向**：W4 主实验前先做三类证据：(1) P1 feature invariance probe，证明同物体 clean/corrupted feature cosine 或 rank 下降；(2) P5 跨方法验证，确认 scale 负增益是 hierarchical 特性还是 cache-family 共性；(3) feature-vs-geometry ROC/AUC + 相似异类 class-pair 分析，决定 ICP-CD 只能辅助还是能进入主预测。论文 §3.4 motivation 段必须引用真实数据，并写入 safe gating：低 AUC / 小 CD margin 时降低几何权重或 fallback 到 feature/text。

### G2 — C2 (vMF) vs 简单"几十文本向量取平均" 区分度不足

- **类型**：C (Comparison) | **严重度**：🟡 Major
- **出处**：`docs/concepts/01_vmf_anchor.md`
- **问题**：高维空间 (d≥256) 下 vMF MAP 估计与简单算术平均的角距离差异可能 < 1°，审稿人会问"换名字"。更关键的是，在理想单峰 vMF 假设下，`normalize(mean)` 本身就是 mean direction 的 MLE，不能在论文里简单宣称普通平均"数学上错误"。
- **修补方向**：附录 A 补 d=64/128/256/512/768 的数值仿真，并把理论论点改成：普通归一化均值是无先验、无不确定性建模、等权重的点估计；vMF-MAP 的优势是 noisy / mixed prompt distributions 下的球面概率建模、先验收缩和 prompt concentration 建模。论文需证明"更稳健"而非"普通平均必错"。

### G3 — C3 (2×3 矩阵) "6 格子" 是先验设计无消融

- **类型**：X (Experiment) | **严重度**：🟡 Major
- **出处**：`docs/concepts/04_2x3_memory_matrix.md`
- **问题**：原稿说"4-6 格子有效"无数据支撑。审稿人秒挑"为什么 2×3 不是 1×3 不是 2×2 不是 3×3"。
- **修补方向**：W5 实验阶段做 6 格 leave-one-out 消融。原稿先注明"格子数由 W5 消融实验决定，初版 6 格留有删减空间"。

### G5 — 紧致度发现的论文定位暧昧

- **类型**：W (Writing) | **严重度**：🟢 Minor
- **出处**：`docs/concepts/03_compactness_diagnosis.md`
- **问题**：原稿没说清紧致度是"独立科学贡献"还是"C2 的辅助 motivation"——影响 §5 章节归属和摘要写法。
- **修补方向**：决策"紧致度作为独立 contribution C5 写在 §5.1 (Discussion 头条)，而非 §3 (Method) 之内"。需要 W9 实验完成后回来确认。

### G8 — F1 (scale 退化) 跨方法验证缺失

- **类型**：X (Experiment) | **严重度**：🟡 Major
- **出处**：`docs/experiments/fig1a_summary.md` + 即将更新的 `02_related_work.md` §2.1 末段
- **问题**：F1 现象目前只在 OpenShape + hierarchical cache 上观察到一次。论文 §2.1 "特征余弦类方法共性"的写法需要更广证据。
- **修补方向**：W2.5 P5 跑 global cache + (可选) ULIP-2 backbone 在 scale 5 个 severity 上的对照实验，1-2 小时成本，决定 §2.1 末段写"hierarchical 特性"还是"特征余弦类共性"。

---

## 已修复漏洞

### G4 — "+1~+3pp" 拆解三档（已修复，2026-05-10 22:40）

- **原严重度**：🟡 Major
- **修复落点**：`docs/proposals/MCP3D_feasibility_and_proposal.md` 附录新增§“性能目标三档声明”
- **修复内容**：三档表格（Floor ≥+0.5pp / Target +1.0-2.5pp / Stretch ≥+3pp）× paper framing × 对应 Green/Yellow/Red 判据。外部 anchor：TPT +1.2 / TDA +0.8 / Point-Cache +3.1。内部 ceiling：W4 oracle 实验定。三档决策规则：基础→投 3DV/WACV；目标→投 AAAI；卓越→AAAI Oral + 期刊扩展。

### G6 — boundary memory 作用澄清（已修复，2026-05-10 22:40）

- **原严重度**：🟡 Major
- **修复落点**：`docs/concepts/04_2x3_memory_matrix.md` 新增 §5.x 常见误解澄清段
- **修复内容**：显式拆解两个误解（“负样本” + “logits 负号是排斥”）+ 真实机制（存特征/top-k 软标签/entropy）+ 与 Point-Cache neg cache 的关系（base 版近似同；若 A1 消融增益<0.5%，升级为基于 logits 梯度不确定性）+ 数学形式（inference time 惩罚项，不是 loss）。

### G7 — vMF 不在点云侧的三理由（已修复，2026-05-10 22:40）

- **原严重度**：🟢 Minor
- **修复落点**：`docs/concepts/01_vmf_anchor.md` 常见问题段新增 Q4
- **修复内容**：三条理由 — (1) 样本量不对称：文本~30 paraphrase 需参数化拟合，点云 cache ~100-500 样本经验分布即可；(2) 算力：vMF MAP 需 Bessel 数值迭代，文本端一次性 OK，点云端每 batch 重拟合爆炸；(3) 维度：文本 d=512 中维拟合稳，点云 d=1280 高维 vMF 相对优势衰减。

### G9 — benchmark corruption 与现实分布偏移混写（已修复，2026-05-11 10:30）

- **原严重度**：🟡 Major
- **修复落点**：`docs/concepts/00_overview.md`、`docs/paper/02_related_work.md`、`docs/reports/2026-05-10_concepts_overview.html`、`docs/reports/2026-05-11_concepts_detailed.html`、`docs/reports/2026-05-10_review_session.html`、`key_findings.md`、`decisions.md`
- **修复内容**：明确 `jitter / dropout / rotate / scale` 是 ModelNet-C / ScanObjectNN-C 中人为设计的 corruption families，是现实 distribution shift 的可控代理和压力测试；F1 的 scale 负增益只能写成 benchmark stress-test 下的失败案例，不能直接等价为现实世界尺度漂移。

---

## 复习时间线

- 2026-05-10 19:46 文件创建。
- 2026-05-10 21:10 录入 G1-G8 八条漏洞。G1/G6/G8 由本次复习直接发现；G2-G5/G7 在故事 arc 讲解过程中暴露。
- 2026-05-10 22:40 批量修补 G4/G6/G7 三条“立即可修的写作类漏洞”。当前待修复 5 条：G1/G2/G3/G5/G8。
- 2026-05-11 10:30 根据用户反馈修补 G9：禁止把人工 corruption 直接写成现实世界分布偏移本身。当前待修复仍为 5 条：G1/G2/G3/G5/G8。
