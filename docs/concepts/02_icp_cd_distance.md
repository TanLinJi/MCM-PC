# 概念 2：ICP-CD 几何距离

> **预计阅读时间**：8 分钟
> **读完应该掌握**：feature 距离为什么会失败、ICP+CD 怎么补救、最大风险是什么

> **重要性**：这是 MCP-3D 最核心的 contribution C1，也是风险最高的部分。

---

## 1. 问题：feature distance 可能撒谎

### 1.1 假设场景

你有两把椅子的点云：

```
X_1：一把扶手椅，朝向正前方
X_2：另一把扶手椅，但被旋转了 90°（朝向右边）

两者属于同一类 "chair"
```

### 1.2 正常情况（编码器表现良好）

```
3D-VLM 编码器编码两个点云：
  z_1 = encoder(X_1) -> 768 维向量
  z_2 = encoder(X_2) -> 768 维向量

cosine similarity = 0.95（很相似）

结论：feature distance 小，正确判断为同类
```

### 1.3 异常情况（编码器对旋转敏感）

```
旋转后：
  z_2_rotated = encoder(X_2_rotated)
  cosine similarity(z_1, z_2_rotated) = 0.5（远了！）

结论：feature distance 大，错误判断为不同类
```

---

## 2. 为什么 feature distance 会失败

### 2.1 编码器号称"近似旋转不变"

OpenShape / Uni3D 这些 3D-VLM 在训练时做了旋转增强（每个样本随机旋转后再用）。
理论上，编码器应该把旋转后的点云映射到接近原向量的位置。

### 2.2 但"近似不变" ≠ "完全不变"

**审计计划 Gap A2 的关键判断**：
> 3D 编码器的"旋转不变性"是**近似的**，不是完美的。
> 这是**未验证假设**——我们不知道实际差距有多大。

所以审计计划里强制 W2.5 做探针实验 P1：
```
对 100 个干净 ModelNet 样本，每个样本：
  1. 用编码器算 z_orig
  2. 旋转 5 个角度（30°, 60°, 90°, 120°, 180°）
  3. 用编码器算 z_rotated
  4. 计算 cosine similarity(z_orig, z_rotated)

如果 cosine 都 > 0.95：
  -> 编码器其实很不变，ICP-CD 没必要做
  -> contribution C1 要重新定位
如果 cosine 在 0.5 ~ 0.8：
  -> 确实有失败案例，ICP-CD 有意义
```

---

## 3. 为什么 Point-Cache 没有"几何兜底"

Point-Cache 完全依赖 3D-VLM 的 feature space。它的逻辑是：

```
1. 编码器把点云映射到 feature
2. 在 feature 空间里比较距离
3. 不管 feature 怎么得来，反正距离小就是同类
```

**问题**：当编码器在 feature 空间错误时，没有任何机制能纠正。

---

## 4. MCP-3D 方案：直接比较 3D 形状本身

### 4.1 核心想法

**除了**比较 feature 距离，**也直接比较**两个点云的 3D 几何。

具体步骤：

```
输入：两个点云 X_1（anchor 样本）和 X_2（待判断样本）

步骤 1（PCA 主轴预对齐）：
  把 X_1 和 X_2 的"主要变化方向"对齐
  这是粗对齐，O(N) 复杂度
  
步骤 2（ICP 细对齐）：
  迭代找到最优的旋转 R 和平移 t
  让 X_2 经过 R*X_2 + t 后，与 X_1 严格对齐
  
步骤 3（Chamfer Distance）：
  对齐后的 X_2 与 X_1 计算"互相最近邻距离平均"
  CD 越小 -> 两个形状越像
```

### 4.2 PCA 主轴预对齐怎么做

```
对一个点云 X（N 个点）:
  1. 计算质心 c = mean(X)
  2. 中心化 X' = X - c
  3. 计算协方差矩阵 Cov = X'.T @ X' / N  （3x3 矩阵）
  4. 特征分解 Cov = V * D * V.T
     V = (v_1, v_2, v_3)，三个主轴方向
     D = diag(lambda_1, lambda_2, lambda_3)，方差大小
  5. 用 V 旋转 X：X_aligned = V.T @ X'
     -> 现在 X 的最大变化方向沿着 x 轴

为什么有用：
  即使两个点云是"旋转过的同一物体"，PCA 后它们的主轴会重合
  这给 ICP 提供了一个不错的初始值
```

### 4.3 ICP 怎么做

```
输入：source = X_2_aligned（已经 PCA 对齐）
      target = X_1
输出：旋转 R 和平移 t，使得 R*source + t 最贴合 target

迭代步骤（重复 max_iter=20 次）：
  1. 对每个 source 点 p_i，找 target 中最近的点 q_i
  2. 求最优 R, t 使得 sum ||R*p_i + t - q_i||^2 最小
     （这个有闭式解，叫 Procrustes 问题）
  3. 应用 R, t 到 source
  4. 检查变化量，如果变化 < epsilon 就停止

我们用 pytorch3d.ops.iterative_closest_point，已经实现好了
```

### 4.4 Chamfer Distance 怎么算

```
输入：两个对齐后的点云 P 和 Q

CD(P, Q) = mean over p in P [ min over q in Q ||p - q|| ]
        + mean over q in Q [ min over p in P ||p - q|| ]

含义：
  - 第一项：P 中每个点到 Q 的最近距离
  - 第二项：Q 中每个点到 P 的最近距离
  - CD 越小 -> 两个点云越"互相覆盖"

我们用 pytorch3d.loss.chamfer_distance，已经实现好了
```

---

## 5. 关键引理：同类 vs 异类的 CD 差异

### 5.1 论文 §3.4 Lemma 1

```
对于同类的两个点云（chair vs chair）：
  ICP 对齐后，CD 接近 0（形状本来就像）

对于异类的两个点云（chair vs table）：
  ICP 对齐后，CD 仍然较大（几何形状本来就不同）
```

这是 ICP-CD 能补救 feature 失败的**理论基础**。

### 5.2 但是有个隐忧（Gap A1）

```
异类但形状相似的情况：
  扶手椅 vs 小沙发：几何上确实接近，CD 也可能很小
  实木桌 vs 实木椅子（无靠背）：座位面板 + 4 条腿，形状像
```

如果 ICP-CD 把这两类也判断为"很像"，那它就**没法区分这些边界情况**。

**审计计划 Gap A1（阻断级风险）**：必须做跨类 ROC 分析。

```
W2.5 探针实验：
  对所有类对（k 个类，C(k, 2) 个对子）：
    随机抽 50 个样本 a 来自类 A
    随机抽 50 个样本 b 来自类 B
    计算 CD(a, b) 后存起来

画 ROC 曲线：
  X 轴：CD 阈值
  Y 轴：用阈值判"同类"的精度

如果 AUC < 0.7：
  -> CD 不能区分相似类
  -> contribution C1 失败，整个 W2.5 之后方向要变
如果 AUC > 0.85：
  -> CD 有判别力，C1 站得住
```

---

## 6. 怎么把 feature 距离和几何距离组合

两个距离量纲不同（feature 是 cosine，CD 是欧氏距离），**不能直接相加**。

我们做 **z-score 归一化 + 加权融合**：

```
对每类 c，维护两个距离的"运行均值和标准差":
  mu_feat_c, sigma_feat_c
  mu_geo_c,  sigma_geo_c

计算 z-score:
  z_feat = (d_feat - mu_feat_c) / sigma_feat_c
  z_geo  = (d_geo  - mu_geo_c)  / sigma_geo_c

加权融合:
  d_combined = (1 - omega) * z_feat + omega * z_geo

其中 omega in [0, 1] 是融合权重
  omega = 0   : 只用 feature 距离（退化为 Point-Cache）
  omega = 1   : 只用几何距离
  omega = 0.5 : 等权融合
```

### 6.1 omega 怎么调

```
W4-W5 阶段：
  扫一遍 omega in {0, 0.3, 0.5, 0.7, 1.0}
  在 ModelNet-C 的 rotate_2 损坏类型上测精度
  选最优的 omega（预期 0.5 ~ 0.7）

W7 阶段：
  考虑做 omega 自适应（不同损坏类型用不同权重）
  比如 jitter 用 omega=0.3（feature 还可靠），
  rotate 用 omega=0.7（feature 不可靠）
```

---

## 7. 比喻总结

```
Feature distance 像"看 ID 卡照片对人"
  - 大部分情况能认出
  - 但化妆 / 戴口罩 / 转脸时可能认错

ICP-CD 像"让对方走过来站到你面前"
  - 物理上 align 了，看真人
  - 但长得像的双胞胎（chair vs sofa）还是分不开

MCP-3D 是"两个一起用"
  - ID 卡 + 真人 = 互补判断
  - 哪个不靠谱就靠另一个补救
```

---

## 8. 三个关键风险（来自审计计划）

| 风险编号 | 内容 | 应对 |
|----------|------|------|
| **A1（阻断级）** | ICP 后异类相似形状的 CD 也小，无法区分 | W2.5 做跨类 ROC，AUC < 0.7 触发 plan B |
| **A2（阻断级）** | 编码器其实很旋转不变，ICP-CD 没必要 | W2.5 P1 探针，cos > 0.95 触发 plan B |
| **A4** | ICP 收敛失败时怎么办 | 添加 inlier ratio 检查，失败时 fallback 到只用 feature |

---

## 9. 在 MCP-3D 里的实现位置

```
代码文件：Point-Cache/runners/model_with_mcp3d.py
相关函数：
  - pca_principal_axis_align     (Section 4，TODO W4 实现)
  - icp_then_chamfer             (Section 4，TODO W4-W5 实现)
  - icp_success_check            (Section 4，TODO W4 实现)
  - PerClassRunningStats         (Section 5，已实现)
  - compute_cell_logits          (Section 8，TODO W6-W7 实现)
```

---

## 接下来读什么

- 概念 3：`03_compactness_diagnosis.md` - 紧致性诊断（contribution C2，是科学发现而非新方法）
- 实现细节：`../../MCP3D_feasibility_and_proposal.md` 第二部分 §3.4
