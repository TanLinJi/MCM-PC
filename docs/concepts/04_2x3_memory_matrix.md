# 概念 4：2×3 记忆矩阵（Memory Matrix）

> **预计阅读时间**：5 分钟
> **读完应该掌握**：6 个格子的分工、为什么这么设计、ablation 怎么做

> **这是 contribution C3**，把前面 3 个概念整合成一个统一的工程系统。

---

## 1. 为什么需要"矩阵"而不是单个记忆库

Point-Cache 已经用了**层次记忆**（global + local 两层）。MCP-3D 进一步把记忆按**两个轴**拆开：

```
轴 1（功能轴）：每个格子负责什么类型的样本
  - confidence  -> 高置信度的样本（最确定的，当类的"标杆"）
  - compactness -> 几何上最像该类的（用 ICP-CD 选出）
  - boundary    -> 决策边界附近的（模棱两可的）

轴 2（表征轴）：每个格子用什么特征空间
  - global -> 整个点云的 768 维特征
  - local  -> 5 个 patch 的局部特征（Point-Cache 现有）
```

3 个功能 × 2 个表征 = **6 个独立记忆库**

---

## 2. 矩阵的可视化布局

```
                  | global         | local
                  | (整个点云)     | (5 个 patch)
       -----------+----------------+----------------
       confidence | global_conf    | local_conf
       (置信)     | (低 entropy)   | (低 entropy patch)
       -----------+----------------+----------------
       compactness| global_comp    | local_comp     <-- 这两个用 ICP-CD
       (紧凑)     | (CD 小)        | (patch CD 小)
       -----------+----------------+----------------
       boundary   | global_bnd     | local_bnd
       (边界)     | (中 entropy)   | (中 entropy patch)
```

---

## 3. 每个格子的具体职责

### 3.1 global_conf（全局置信库）

```
负责：识别"非常确定属于该类"的整个点云
存什么：3D-VLM 编码后的 768 维全局特征
筛选标准：entropy 最低的 K 个样本（每类 K=3）
作用：作为该类的"标杆"，给查询提供主要相似度信号
```

### 3.2 local_conf（局部置信库）

```
负责：识别"局部 patch 非常确定属于该类"的样本
存什么：5 个 patch 的局部特征（来自 Point-Cache 现有的 hierarchical cache）
筛选标准：patch entropy 最低的 K 个样本
作用：补充 global，对部分遮挡的物体更鲁棒
```

### 3.3 global_comp（全局紧凑库）⭐ 新增

```
负责：识别"几何上最像该类"的整个点云
存什么：3D-VLM 编码后的 768 维全局特征 + 原始点云数据
筛选标准：用 ICP+Chamfer 距离最小的 K 个样本
作用：当 feature 距离不可靠时（旋转损坏），提供几何兜底
```

### 3.4 local_comp（局部紧凑库）⭐ 新增

```
负责：识别"局部 patch 几何上最像该类"的样本
存什么：5 个 patch 的局部特征 + 局部 patch 的原始点云
筛选标准：patch 级别的 ICP+CD 距离
作用：对局部几何变形（如某部分缺失）鲁棒
```

### 3.5 global_bnd（全局边界库）⭐ 新增

```
负责：识别"决策边界附近"的样本
存什么：768 维全局特征 + 概率分布 prob_map
筛选标准：entropy 在中等范围（既不是最低也不是最高）
作用：提供"反向信号"，告诉模型"这种特征不要相信"
       类似 Point-Cache 的 negative cache，但角度不同
```

### 3.6 local_bnd（局部边界库）

```
负责：识别"局部 patch 处于决策边界"的样本
默认状态：DISABLED（关闭）
原因：审计计划 Gap C2 推测局部边界库太弱，未必有用
ablation A1 时打开测试，看是否值得保留
```

---

## 4. 6 个格子的分工总览

| 格子 | 表征 | 功能 | 关键技术 | 默认 |
|------|------|------|----------|------|
| `global_conf` | 768 维 | 高置信 | feature 相似度 | ✅ ON |
| `local_conf` | 5 patch | 高置信 | feature 相似度 | ✅ ON |
| `global_comp` | 768 维 + raw pc | 几何相似 | **ICP+CD** | ✅ ON |
| `local_comp` | 5 patch + raw | 几何相似 | **patch ICP+CD** | ✅ ON |
| `global_bnd` | 768 维 + prob | 边界 | mid-entropy 筛选 | ✅ ON |
| `local_bnd` | 5 patch + prob | 边界 | mid-entropy 筛选 | ❌ OFF |

---

## 5. 最终预测：多源融合

把所有信号加起来：

```
final_logits = z_text * text_logits        （文本锚点贡献）
             + a_1 * global_conf_logits    （全局置信）
             + a_2 * local_conf_logits     （局部置信）
             + a_3 * global_comp_logits    （全局紧凑）
             + a_4 * local_comp_logits     （局部紧凑）
             - a_5 * global_bnd_logits     （全局边界，负号）
             - a_6 * local_bnd_logits      （局部边界，负号；若打开）
```

**为什么 boundary 用负号**：

```
boundary 格子捕捉的是"模糊样本"
  - 如果当前查询和 boundary 样本相似 -> 说明它也很模糊
  - 应该【降低】对这种特征的信心
  - 类似 Point-Cache 的 negative cache 机制
```

### 5.x 常见误解澄清（修补 G6, 2026-05-10）

新读者经常把 boundary memory 理解错，这里明确区分：

```
误解 1：boundary memory 是负样本（negative sample）
  -> 错误。负样本通常意味着用损失函数（如 InfoNCE 对比损失）
     训练时让特征【远离】负样本。
  -> 但 MCP-3D 是 TTA (Test-Time Adaptation) = training-free 范式，
     没有任何损失函数在跑，没有参数在更新。
     "负样本" 这个词在 TTA 里本质是误用。

误解 2：logits 里的负号意味着"排斥"这一类
  -> 错误。负号作用在 logits 层面，不是特征层面。
  -> 它降低的是主预测在"模糊区"的【置信度】，
     让 fusion 层知道"这个区域不要瞎自信"。
  -> 本质是【校准信号 / calibration signal】，不是【类别排斥】。

真实机制：boundary memory 存什么、怎么用
  存什么：
    - 特征向量（用于余弦相似度比较）
    - top-k 类别软标签 e.g. [chair: 0.4, sofa: 0.35, stool: 0.25]
    - entropy 值（筛选进入 boundary 的门槛）

  怎么用：
    1. 新查询 q 进来，算 q 和 M_bnd 里每个样本的余弦
    2. 若 max_sim > 阈值 → 说明 q 也处于"模糊区"
    3. 这时主预测 logits 的 top-1 值被软化（乘以 < 1 的因子）
       或 boundary 样本的软标签被加权融合进 final prediction
    4. 结果：在 3 个接近类别上分布更均匀，而不是"硬选一个"

与 Point-Cache 的 negative cache 的区别（v2 提案 Risk 表已指出）：
  - Point-Cache neg cache：prob_map 软掩码，机制相近
  - MCP-3D Boundary Memory 当前 base 版：机制近似相同（风险点）
  - 差异化方向（若 A1 消融显示 < 0.5% 增益，必须升级）：
      基于 logits 梯度方向的不确定性（而非仅 entropy 区间）
      → 这是 W7 阶段的待验证项
```

数学描述（已在 `MCP3D_full_proposal_v2.md` §4.6 展开）：

```
l_final = zeta_1 * main_prediction
        + zeta_2 * [rho * global + (1-rho) * local]  ← 正向：信心+紧凑
        - zeta_3 * Psi(h, M_bnd)                     ← 校准：boundary 惩罚项

其中 Psi 是 boundary memory 的贡献，负号表示降低置信。
关键：这个负号在【inference 时】作用，不是训练时的 loss。
```

---

## 6. 怎么调融合权重 a_i

### 6.1 W1-W7 阶段：等权重起步

```
所有 a_i = 1.0
所有 z_text = 1.0

先看看默认权重下的整体效果
```

### 6.2 W7 阶段：根据 ablation 调整

```
A1（leave-one-out ablation）：
  逐个关掉一个格子，看精度跌多少
  -> 跌得多 = 重要 = a_i 调大
  -> 不跌 = 不重要 = a_i 调小或直接关掉

A2（omega scan）：
  紧凑库内的 omega ∈ {0, 0.3, 0.5, 0.7, 1.0}
  找 ICP-CD vs feature 的最优融合权重
```

### 6.3 W8+：可能改成自适应

```
对不同损坏类型用不同权重：
  - jitter 损坏：feature 还可靠 -> 紧凑库权重小
  - rotate 损坏：feature 不可靠 -> 紧凑库权重大

自适应规则：
  根据 entropy / 不确定度切换 omega
  这是 W11+ 的优化方向
```

---

## 7. 矩阵设计的合理性论证

**审稿人会问**："为什么是 3×2 = 6 而不是其他组合？"

我们的回答：

### 7.1 功能轴为什么是 3 个

```
- confidence  -> 必要：每类总需要"标杆"样本
- compactness -> 新增：解决 feature 失败问题（C1 contribution）
- boundary    -> 必要：增强决策边界（类似 Point-Cache neg cache）

三者覆盖了 anchor 选择的三个独立维度，缺一不可
```

### 7.2 表征轴为什么是 2 个

```
- global -> Point-Cache 已有，必要
- local  -> Point-Cache 已有，必要（处理部分遮挡）

我们没有发明新的表征，只是把 Point-Cache 的两层延展到 3 个功能
```

### 7.3 为什么 6 个格子可以独立 enable/disable

```
ablation A1：leave-one-out
  逐个关掉，看精度损失
  -> 让我们知道每个格子的实际贡献
  -> 论文里写一个完整的 ablation table
```

---

## 8. 在 MCP-3D 里的实现位置

```
代码文件：Point-Cache/runners/model_with_mcp3d.py

相关类：
  - CellConfig            (Section 6) -> 单个格子的配置
  - MemoryCell            (Section 6) -> 单个格子的实现
  - MCP3DMemoryMatrix     (Section 7) -> 矩阵管理器
  - FusionWeights         (Section 9) -> 融合权重
  - fuse_multi_source_logits (Section 9) -> 融合函数

实现状态：
  - 类结构和接口：已实现
  - 单格子 logits 计算：TODO（W6-W7）
  - 主循环填充：TODO（W6-W7）
```

---

## 9. 论文里的位置

```
第 3 章 方法（Method）:
  3.5 多记忆矩阵（Multi-Memory Matrix）   <-- C3 在这里
    3.5.1 功能轴 vs 表征轴
    3.5.2 单格子记忆机制
    3.5.3 多源融合公式
    3.5.4 自适应权重（CVPR 阶段）

第 4 章 实验:
  4.4 ablation study
    Table 3：A1 leave-one-out ablation
    Table 4：A2 omega scan
```

---

## 10. 整合：MCP-3D 完整流程

```
推理时（每个测试样本）:

1. 编码：
   - 3D-VLM 编码 -> 全局特征 z（768 维）
   - 3D-VLM 编码 -> 5 个局部 patch 特征
   - CLIP 文本编码（已离线）-> vMF 锚点 a_c（每类一个）

2. 计算 6 个格子的 logits（每个格子都给出 (1, n_cls)）:
   - 4 个 conf/comp 格子：feature 相似度 -> logits
   - 2 个 bnd 格子：基于 entropy 加权 -> logits
   - compactness 格子额外：用 ICP-CD 计算几何距离

3. 融合：
   final_logits = text + sum(positive_cells) - sum(boundary_cells)
   
4. 预测：
   y_hat = argmax(final_logits)
   
5. 更新记忆库：
   根据 y_hat 把当前样本放入相应格子（满足该格子筛选标准的）
```

---

## 总结：MCP-3D 概念全景

读完 5 份概念文件后，你应该能口述：

```
背景：
  3D 点云 TTA，Point-Cache 是 SOTA，但有 3 个软肋

3 个 contribution：
  C1（ICP-CD）：补救 feature 距离失败
  C2（紧致性诊断）：揭示 TTA 失效机制
  C3（2×3 矩阵）：整合多源信号
  C4（vMF 锚点）：数学严谨化（minor）

整合架构：
  6 个记忆格子，2 轴拆分
  最终预测 = 文本锚点 + 5 个正向格子 - 1-2 个边界格子

风险：
  最关键是 W2.5 探针实验 P1（旋转不变性）和 P2（紧致性 r 值）
  探针失败 -> 重新规划
```

---

## 现在你掌握了什么

✅ 知道 MCP-3D 在解决什么问题
✅ 知道 3 个 contributions 分别是什么、为什么 novel
✅ 知道每个 contribution 的最大风险和应对方案
✅ 知道 6 个记忆格子的分工
✅ 能口述完整的方法流程

可以进入下一阶段：
- **B 文件走读**：陪你逐段读 `model_with_mcp3d.py` 的代码
- **D 快速上手指南**：写一份 5/30/60 分钟分级指南
- **E 直接动手**：开始跑 setup_env.sh + download_data.sh

请告诉我哪个最适合你下一步的需要。
