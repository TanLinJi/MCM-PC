# MCP-3D 关键发现库

> **用途**：实验数据 → 论文 contribution / 失败案例 / narrative 调整的因果链
>
> **维护原则**：每条 finding 必须有"出处（哪个实验跑出来）+ 数字 + 解读 + 影响哪个设计点"。
> 任何会话都应先读这个文件，再讨论 MCP-3D 的方法论调整。

---

## F1（2026-05-10）：ModelNet-C scale corruption 上 Point-Cache hierarchical TTA 反而退化 ⭐

**出处**
- 实验：Figure 1(a) bar 3 复现（35-setting 全跑）
- 数据源：`Point-Cache/logs/fig1a_bar3_20260510_152446/scale_*.log`
- 汇总：`docs/experiments/fig1a_summary.md` 第四节

**数字**

| sev | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| zero-shot | 80.23 | 78.81 | 78.57 | 77.92 | 76.82 | 78.47 |
| **TTA** | **79.70** | 79.17 | **78.16** | **77.19** | **76.13** | **78.07** |
| Δ (TTA-ZS) | **-0.53** | +0.36 | **-0.41** | **-0.73** | **-0.69** | **-0.40** |

**5 个 severity 里 4 个为负**——TTA 在 ModelNet-C 的人工 scale corruption family 上不仅没帮忙，还小幅伤害了精度。

**解释边界**
- 这是一条 benchmark stress-test finding，不是对现实世界分布偏移的直接等价声明。
- 论文中应写成：真实问题是 distribution shift；scale 是手工构造的可控代理，用于诊断全局几何变化下的 failure mode。

**机制解释**
- Point-Cache 的 cache 检索基于"特征余弦相似度"
- 在该人工 scale corruption protocol 下，整体几何变化可能偏移 feature manifold（不是局部噪声，而是全局形变代理）
- 历史 cache 里的 key 反而误导预测 → TTA 减分

**对 MCP-3D 的影响**
- ⭐ 这是 **C1 (ICP-CD 几何距离) 最有 promise 的 attack point**
- ICP-CD 对全局 scale 不变（chamfer distance 在归一化点云上）→ 应能矫正这种 failure mode
- → W2.5 P2 探针 + W4 主实验都要在 scale_2/3/4 上重点验证 C1 的恢复能力
- → 论文 §4 Method motivation 段引用本数字时，必须表述为"benchmark corruption 下的特征空间不足证据"

**风险/反例**
- 如果 W4 实验显示 ICP-CD 在 scale 上**也救不回来**，要诚实地把这条放到 §5 失败案例分析

---

## F2（2026-05-10）：jitter 是 OpenShape 的最深软肋，TTA 拯救最显著 ⭐

**出处**
- 同 F1，bar 3 复现日志
- jitter_*.log

**数字**

| sev | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| zero-shot | 79.29 | 71.39 | 59.76 | 45.75 | 32.66 | 57.77 |
| TTA | 80.06 | 75.08 | 68.80 | 54.74 | 45.06 | 64.75 |
| Δ | +0.77 | +3.69 | +9.04 | +8.99 | **+12.40** | **+6.98** |

- jitter sev=4 单 cell **+12.40pp**：35 cell 里增益最大
- jitter mean **+6.98pp**：7 corruption 里增益最大

**机制解释**
- jitter = per-point Gaussian noise，破坏特征局部稳定性
- Point-Cache 的多尺度 cache（global+local hierarchical）通过历史样本平均把噪声抹掉
- 即使如此，sev=4 的最终精度仍只有 45.06——还有大量空间

**对 MCP-3D 的影响**
- C1 (ICP-CD) 在 jitter 上要拿出"几何距离比特征余弦更稳健"的具体数字证据
- → W2.5 P1 探针 + W4 主实验都用 jitter 作 stress test
- 但 jitter 不是 C1 最有 promise 的点（scale 才是，参见 F1）；jitter 的卖点是"continuous improvement"叙事

---

## F3（2026-05-10）：rotate 上 TTA 几乎无增益 (+0.58pp)

**出处**：bar 3 复现，rotate_*.log

**数字**

| sev | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| zero-shot | 84.12 | 83.71 | 82.54 | 79.21 | 72.33 | 80.38 |
| TTA | 84.97 | 83.79 | 82.82 | 79.34 | 73.87 | 80.96 |
| Δ | +0.85 | +0.08 | +0.28 | +0.13 | +1.54 | **+0.58** |

**机制解释**
- OpenShape backbone 在预训练时见过大量旋转增广 → 已内置 rotation-awareness
- TTA 的 cache 检索逻辑对"旋转后特征仍接近原 anchor"几乎没提升空间

**对 MCP-3D 的影响 ⚠️ 重大叙事调整**
- 原计划 C1 narrative = "pose-shape disentanglement"（解耦位姿与形状）→ **死了**
- 因为 rotate 上 zero-shot 已经够好（80.38），TTA 涨不了，C1 的 ICP-CD 也不可能超过 zero-shot 太多
- → 论文 narrative 改为 **"corruption-specificity"**（D5 锁定）：C1 不是号称"旋转不变"，而是"对 scale/jitter 这类全局形变特别有效"
- → W2.5 P1 探针就是来"埋葬"原 motivation 的，不是验证它

---

## F4（2026-05-10）：dropout sev=4 cliff（论文 N+1 的失败案例素材）

**出处**：bar 3 复现，dropout_global_4.log

**数字**
- dropout_global sev=4：zero-shot 63.57 → TTA 65.19，仅 +1.62pp
- 点数从 1024 → 256（按 OpenShape 协议），形状已严重退化

**对 MCP-3D 的影响**
- 这是"任何 TTA 方法都救不回来"的极端损坏
- 论文 §5 失败案例分析章节直接用此例：当输入信息丢失到一定程度，所有方法都失效
- → 不指望 MCP-3D 在 dropout sev=4 拿亮眼数字

---

## F5（2026-05-10）：复现整体偏差 ~-1pp，系统性同向

**出处**
- bar 1 (clean): 83.27 vs 论文 84.56 → -1.29pp
- bar 2 (ZS 35-mean): 72.51 vs 论文 73.49 → -0.98pp
- bar 3 (TTA 35-mean): 75.27 vs 论文 76.59 → -1.32pp

**判定**：D9 锁定接受。三柱同向 -1pp 偏差，归因 cudnn/fp16/template-order/EMA-vs-final，不深 debug。

**对 MCP-3D 的影响**
- 我们最终论文里 baseline 数字会比 Point-Cache 原论文低 1pp 左右，这是"我们重跑环境"的事实
- 但 **TTA gain（bar3-bar2）+2.76pp ≈ 论文 +3.10pp** → 方向 + 量级正确，足够支撑相对比较
- → MCP-3D 的提升必须基于"我们重跑的 75.27"做对比，不能跨论文比较绝对数字

---

## F6（2026-05-11）：scale 的主因是 anchor pollution，不是 feature failure ⭐

**出处**
- `docs/experiments/p1/P1_full_drift.md`
- `docs/experiments/p1/P1_pollution_sim.md`
- `docs/decisions/D22_p1_anchor_pollution_pivot.md`

**关键数字**

| setting | 数字 | 解读 |
|---|---:|---|
| scale_2 feature cos mean | 0.9306 | corrupted feature 仍接近 clean feature |
| scale_2 class-consistent | 95.5% | feature space 里同类关系仍然很强 |
| scale_2 corrupt anchor top-1 | 84.44% | 使用测试流 anchor 会被污染拖累 |
| scale_2 clean anchor top-1 | 95.46% | 干净 anchor 显著更好 |
| scale_2 clean - corrupt | +11.02pp | anchor source 是主要变量 |

**结论**
- F1 里"scale 是 ICP-CD 最有希望 attack point"的解释需要修正。
- scale 仍然是重要 attack point，但 attack 的对象不是 feature failure，而是 anchor pollution。
- 论文里应将该发现写成机制贡献：cache-based 3D TTA 的负适配可由测试流 anchor 污染触发。

## F7（2026-05-11）：jitter 真正破坏 feature，stable anchor 不能全局使用

**出处**
- 同 F6。

**关键数字**

| setting | cos mean | class-consistent | clean-anchor Δ |
|---|---:|---:|---:|
| jitter_2 | 0.6976 | 59.5% | -24.84pp |
| jitter_3 | 0.6201 | 30.8% | -51.18pp |
| jitter_4 | 0.5671 | 15.5% | -62.68pp |

**结论**
- jitter 与 scale 属于不同机制区间：scale 主要是 anchor pollution，jitter 是 feature failure + anchor 选择风险。
- 不能把 clean/source/stable anchor 作为全局方法；必须做 conditional anchor switching。
- 下一步 smoke 必须同时包含 scale_2 和 jitter_3，前者验证收益，后者验证安全性。

## 待补充（占位）

- F8：W2.5 P2 紧致度-精度相关性 r 测量 → 决定 C2 narrative
- F9：W2.5 P3 T4 显存测量 → 决定要不要降级 backbone
- F10：conditional anchor switching smoke → 验证 scale_2 收益与 jitter_3 安全性
- F11：W4 ICP 残差分布（若保留 appendix 诊断）→ 定 ICP 失败阈值
- F10：W7 主实验 MCP-3D vs Point-Cache 比较 → 论文 Table 4 主结果
- ……

---

## 索引：finding × MCP-3D contribution

| finding | 影响哪个 contribution | 影响哪个论文章节 |
|---|---|---|
| F1 (scale 反向) | C1 触发点 | §4 motivation, §6 Table 4 |
| F2 (jitter 软肋) | C1 安全门控 | §4 motivation |
| F3 (rotate 无增益) | C1 narrative pivot | §3 framing |
| F4 (dropout cliff) | 失败案例 | §5.2 limitations |
| F5 (复现 -1pp) | 实验协议 | §6.1 reproduction note |
| F6 (anchor pollution) | C1 anchor switching | §1.2, §3.4 |
| F7 (jitter feature failure) | C1 abstention / safety | §3.4, §5 |
