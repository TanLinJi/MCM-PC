# Figure 1(a) 复现 — OpenShape 列三柱

> 数据源：`logs/fig1a_bar2_20260510_122450/` + `logs/fig1a_bar3_20260510_152446/`
> 复现完成时间：2026-05-10 19:05
> Backbone：OpenShape PointBERT-vitg14-rgb (paper primary)
> 协议：ModelNet-C 7 corruption × 5 severity = 35 setting，算术平均

---

## 一、三柱总览

| Bar | 含义 | 你的数 | 论文 | Δ | 状态 |
|---|---|---|---|---|---|
| 1 (绿) | clean ModelNet40 zero-shot | **83.27** | 84.56 | -1.29pp | ✓ |
| 2 (橙) | ModelNet-C zero-shot 35-mean | **72.51** | 73.49 | -0.98pp | ✓ |
| 3 (紫) | ModelNet-C Point-Cache hierarchical TTA 35-mean | **75.27** | 76.59 | -1.32pp | ✓ |

**判定**：三柱全部在 ±2pp 容忍内复现成功（D9 决策）。系统性 -1pp 偏差归因于 cudnn/fp16/template-order，不深 debug。

**TTA 增益**：bar 3 - bar 2 = **+2.76pp**（论文 +3.10pp）。

---

## 二、Bar 3（hierarchical TTA）完整 35-cell 矩阵

| corruption \ severity | sev=0 | sev=1 | sev=2 | sev=3 | sev=4 | per-cor mean |
|---|---|---|---|---|---|---|
| add_global       | 79.90 | 76.22 | 74.80 | 71.35 | 70.91 | **74.64** |
| add_local        | 77.43 | 75.45 | 73.82 | 72.12 | 70.95 | **73.95** |
| dropout_global   | 83.10 | 84.00 | 82.25 | 80.11 | 65.19 | **78.93** |
| dropout_local    | 82.25 | 80.31 | 76.66 | 71.96 | 66.86 | **75.61** |
| jitter           | 80.06 | 75.08 | 68.80 | 54.74 | 45.06 | **64.75** |
| rotate           | 84.97 | 83.79 | 82.82 | 79.34 | 73.87 | **80.96** |
| scale            | 79.70 | 79.17 | 78.16 | 77.19 | 76.13 | **78.07** |

**35-mean = 75.27**

---

## 三、Bar 2（zero-shot）完整 35-cell 矩阵（对照用）

| corruption \ severity | sev=0 | sev=1 | sev=2 | sev=3 | sev=4 | per-cor mean |
|---|---|---|---|---|---|---|
| add_global       | 78.85 | 74.72 | 71.47 | 69.57 | 68.15 | **72.55** |
| add_local        | 74.84 | 70.87 | 67.50 | 64.71 | 63.37 | **68.26** |
| dropout_global   | 83.75 | 83.06 | 81.24 | 78.61 | 63.57 | **78.05** |
| dropout_local    | 80.47 | 77.96 | 73.46 | 68.03 | 60.70 | **72.12** |
| jitter           | 79.29 | 71.39 | 59.76 | 45.75 | 32.66 | **57.77** |
| rotate           | 84.12 | 83.71 | 82.54 | 79.21 | 72.33 | **80.38** |
| scale            | 80.23 | 78.81 | 78.57 | 77.92 | 76.82 | **78.47** |

**35-mean = 72.51**

---

## 四、TTA 增益矩阵（bar 3 − bar 2）

| corruption \ severity | sev=0 | sev=1 | sev=2 | sev=3 | sev=4 | per-cor Δ |
|---|---|---|---|---|---|---|
| add_global       | +1.05 | +1.50 | +3.33 | +1.78 | +2.76 | **+2.09** |
| add_local        | +2.59 | +4.58 | +6.32 | +7.41 | +7.58 | **+5.69** |
| dropout_global   | -0.65 | +0.94 | +1.01 | +1.50 | +1.62 | **+0.88** |
| dropout_local    | +1.78 | +2.35 | +3.20 | +3.93 | +6.16 | **+3.49** |
| jitter           | +0.77 | +3.69 | +9.04 |+8.99 |+12.40 | **+6.98** |
| rotate           | +0.85 | +0.08 | +0.28 | +0.13 | +1.54 | **+0.58** |
| scale            | -0.53 | +0.36 | -0.41 | -0.73 | -0.69 | **-0.40** |

**总均值 Δ = +2.76pp**

---

## 五、关键观察 ⭐

### 5.1 jitter 是 TTA 拯救最多的 corruption（+6.98pp）

- jitter 是 OpenShape 的"软肋"：sev=4 时 zero-shot 仅 32.66
- TTA 在 sev=4 上拉回到 45.06（**+12.40pp**），是 35 个 cell 里增益最大的
- 但 jitter sev=4 仍只有 45.06，远低于其他 corruption sev=4 → **MCP-3D 的 C1 (ICP-CD)** 应在这里继续 attack：几何距离对 per-point 噪声更稳健

### 5.2 scale 上 TTA 反而退化 ⚠️ (-0.40pp)

| sev | 0 | 1 | 2 | 3 | 4 | mean |
|---|---|---|---|---|---|---|
| Δ | -0.53 | +0.36 | -0.41 | -0.73 | -0.69 | -0.40 |

5 个 severity 里 4 个为负。**这是 MCP-3D 最有 promise 的 attack point**：

- Point-Cache 的 cache 检索基于特征余弦相似度
- scale 扰动会整体偏移 feature manifold（不是局部噪声）
- 历史 cache 里的 key 反而误导预测 → TTA 减分
- **ICP-CD 几何距离对全局 scale 不变** → 应能矫正这种 failure mode

→ **W2.5 P1 探针实验**应优先在 scale 上验证 ICP-CD 的恢复能力。

### 5.3 rotate 上 TTA 几乎无增益（+0.58pp）

- 因为 OpenShape 已经是 rotation-aware
- TTA 的 cache 检索逻辑对"旋转后特征仍接近原 anchor"没什么提升空间
- → C1 narrative "pose-shape disentanglement" 在 OpenShape 上**不成立**（与 W2.5 P1 一致预测）
- → 论文叙事建议改成 "corruption-specificity"（D5 已锁）

### 5.4 dropout sev=4 的 cliff

- dropout_global sev=4：zero-shot 63.57 → TTA 65.19，仅 +1.62pp
- dropout_local sev=4：zero-shot 60.70 → TTA 66.86，+6.16pp
- 当点数从 1024 → 256（dropout_global sev=4）时，**形状已经过度退化**，TTA 也救不回多少
- → 这给 **MCP-3D 失败案例分析章节**（论文 N+1）提供素材

---

## 六、对论文 Figure 1(a) 的复现质量

```
            paper        ours        delta      tolerance       status
bar 1       84.56        83.27       -1.29pp    ±2pp            ✓ pass
bar 2       73.49        72.51       -0.98pp    ±2pp            ✓ pass
bar 3       76.59        75.27       -1.32pp    ±2pp            ✓ pass
TTA gain    +3.10pp     +2.76pp     -0.34pp    qualitative      ✓ direction match
```

**复现质量评分**：A-（系统性 -1pp 偏差但完全 in tolerance，TTA 方向 + 量级正确）。

---

## 七、对 MCP-3D 设计的启示（汇总到 chat 决策日志）

| 观察 | MCP-3D 行动 | 涉及 contribution | 涉及探针 |
|---|---|---|---|
| jitter 退化最重 + TTA 涨最多 | C1 ICP-CD 在 jitter 上重点验证 | C1 | P1 |
| scale 上 TTA 反向 | C1 ICP-CD 在 scale 上重点验证（最有 promise） | C1 | P2 |
| rotate 上 TTA 无增益 | C1 narrative 改 "corruption-specificity" | (C1 framing) | P1 |
| dropout sev=4 cliff | 失败案例素材 | N+1 写作 | P4 |

---

## 八、复现脚本清单（可重跑）

```bash
# bar 1: 单条命令，~5 min
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 \
python runners/zs_infer.py --config configs --lm3d openshape --cache-type global \
    --ckpt_path weights/openshape/openshape-pointbert-vitg14-rgb/model.pt \
    --dataset modelnet40 --npoints 1024 --oshape-version vitg14 --wandb-log

# bar 2: 35 setting × 双 T4，~90 min
bash Point-Cache/scripts/repro_fig1a_bar2_zs_corruption.sh

# bar 3: 35 setting × 双 T4 hierarchical TTA，~3.5 h
bash Point-Cache/scripts/repro_fig1a_bar3_tta.sh

# 任何 log 目录的汇总
python Point-Cache/scripts/repro_fig1a_summarize.py logs/<dir>
```

**重跑产物位置**：
- `logs/fig1a_bar2_<时间戳>/` × 35 个 .log
- `logs/fig1a_bar3_<时间戳>/` × 35 个 .log
- 本文件由人工汇总；后续如需重新生成，可写 `repro_fig1a_make_summary.py` 自动化（暂未实现）
