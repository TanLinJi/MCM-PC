# MCP-3D：可行性诊断与修订版完整 Proposal

> 本文档独立于 `MCP3D_framework.md` / `MCP3D_full_proposal.md` / `MCP3D_full_proposal_v2.md`，在阅读完三篇参考文献（Point-Cache CVPR'25、MCP/MCP++ ICCV'25、BayesMM CVPR'26）以及 `MCP/`、`Point-Cache/` 两套官方源码之后，对你提出的研究思路给出**独立可行性诊断 + 修订版完整 Proposal**。
>
> 阅读顺序建议：
> 1. **第一部分** — 你这条思路在科研意义上是否站得住？（关键看创新性与三大风险）
> 2. **第二部分** — 修订版完整 Proposal（在第一部分诊断基础上的"可投稿版"方案）
> 3. **第三部分** — 实施路线图、风险对策、与现有 v1/v2 提案的差异说明

---

## 第一部分：可行性诊断

### 1.1 思路一句话总结

> **把 MCP/MCP++ 的"多缓存原型学习 + 残差精修"机制从 2D CLIP 迁移到 3D 点云多模态模型，使用 Point-Cache 的层级（global+local）表征作为骨架，仅借鉴 BayesMM 的"LLM 文本扩充 + 文本分布建模"模块。**

这个表述本身是清晰、合理、可执行的。下面我们逐点拆解它的"科研含金量"和"工程可行性"。

### 1.2 三篇文献交集与你的研究空白

| 维度 | Point-Cache (CVPR'25) | MCP/MCP++ (ICCV'25) | BayesMM (CVPR'26) | **你的 MCP-3D** |
|------|-----------------------|---------------------|-------------------|-----------------|
| 模态 | 3D 点云 | 2D 图像 | 3D 点云 | 3D 点云 |
| 缓存数 | 2（global + local，但功能单一：低熵正缓存 + 负缓存） | 3（entropy / align / negative） | 0（贝叶斯分布替代缓存） | **2 × 3 = 6**（层级 × 功能） |
| 选择准则 | 低熵 | 低熵 + 特征到文本中心距离 | — | **低熵 + 几何紧致性 + 边界熵** |
| 文本处理 | 多 prompt 模板（无分布建模） | CuPL prompts（无分布建模） | LLM paraphrase + 高斯分布 + MAP | **LLM paraphrase + vMF/高斯分布 + MAP** |
| 残差精修 | 无 | 有（MCP++） | 无（贝叶斯推断替代） | 有（可选模块 MCP-3D++） |
| 几何信息 | 仅特征空间 | 仅特征空间 | 仅特征空间 | **特征 + Chamfer/ICP 混合距离**（核心新点） |

**真实研究空白**：
- (a) **MCP 的多缓存机制尚未被验证在 3D 点云上**——这是工程性贡献。
- (b) **3D 点云独有的几何度量（CD/ICP）能否融入"紧致性原型选择"** — 这是真正的方法论新点。
- (c) **3D 紧致性-性能相关性是否仍成立** — 这是理论性贡献。

只有 (b) 和 (c) 才是顶刊级新点。(a) 单独不够支撑顶会。

### 1.3 创新性诊断：哪些是真新点，哪些会被审稿人质疑

#### ✅ 真正新颖、有论文卖点的部分

1. **几何感知紧凑记忆库（Compactness Memory with ICP-aligned Chamfer Distance）** ⭐⭐⭐⭐⭐
   - MCP 的 align cache 用 `||feat - text_center||` 做选择（见 `@/root/autodl-tmp/MCP-Point-Cache/MCP/mcp_runner.py:55-74`）
   - 你的关键升级：在 3D 中**特征距离不足以反映"形状紧致性"**，因为：
     - 3D-文本对齐空间比 CLIP 弱（数据量小一两个数量级）
     - 旋转损坏会导致特征位置漂移但形状不变
   - **ICP 配准 + Chamfer 距离 = 物理空间的形状紧致性度量**，这是 2D 域不存在、3D 域天然适配的全新维度。
   - **这就是论文最强的卖点。**

2. **3D 紧致性-性能相关性的系统性诊断** ⭐⭐⭐⭐
   - MCP 在 2D 上发现 r ≈ 0.82；3D 是否一致？哪些损坏类型打破了它？
   - 这是一篇可以"独立成节"的实证发现，且无人做过。
   - 如果数据支撑（rotate 下相关性低、dropout 下相关性崩溃），就能反过来**论证为什么需要几何距离作为补充**。

3. **2 × 3 记忆矩阵的正交分解** ⭐⭐⭐
   - 表征轴（global/local）× 功能轴（conf/comp/bnd）
   - 形式整洁，叙事清晰，但**单纯的笛卡尔积不构成强方法贡献**——审稿人会问"为什么不是 3 × 3 或 1 × 6？"
   - 必须用消融实验证明每个格子都有边际收益，否则会被砍。

#### ⚠️ 容易被质疑的部分

4. **"借鉴 BayesMM 文本处理"的边界** ⚠️⚠️
   - BayesMM 的 LLM 扩充 + 高斯分布 + MAP 估计是其**核心方法**，不是边角料。
   - 如果你照搬，审稿人会问："你这跟 BayesMM 的差别只是把贝叶斯推断换成了缓存？"
   - **必须做出实质改动**：
     - v2 已经修正为 vMF 分布（球面统计而非欧氏高斯）——这个修正本身是合理的、有理论依据的
     - 但要明确说："BayesMM 的高斯假设在归一化特征上有度量错配，我们改用 vMF；这不是借鉴而是修正"
     - 同时要注意：**原 BayesMM 在球面上做高斯估计能 work，说明 d 较大时高斯近似 vMF 误差不大**，所以你的"vMF 修正"可能在数值上提升有限。需要消融实验严格验证 vMF vs 高斯的实际增益。

5. **MCP 的 align cache 用 `||feat - text_center||`，你把它换成 `Ω(feat, geo)`** ⚠️
   - 这是直接的"换距离函数"工程改动。
   - 加分项：换的合理（特征→几何混合距离适配 3D）
   - 减分项：从方法贡献角度看，是渐进式而非范式式创新

6. **MCP++ 残差精修在 3D 上是否仍然有效** ⚠️⚠️⚠️
   - MCP++ 的 `loss_align` 假设 CLIP 特征空间高度对齐——3D 多模态模型（ULIP-2 / OpenShape / Uni3D）的对齐质量明显弱
   - 你的 v1 提案降权到 0.5（vs MCP++ 的 1.0），但**这个数字是经验拍脑袋的**
   - 风险：残差精修可能在 3D 上**反而退化**（在 align loss 引导下把特征推向不准的文本锚点）
   - 必要做法：A6 残差消融必须诚实报告"哪些 backbone/损坏下残差精修退化"

### 1.4 技术可行性逐项评估

#### Module 1：vMF 文本锚点 — **可行 ✅，但需精简**

- 数学上 v2 的修正（球面 SLERP + vMF MAP）是**正确的**。
- 实现复杂度低：一行代码 `(κ₀·z̄ + κ̂·m) / ||·||`。
- **不要叙述 vMF 的密度函数 + Bessel 函数全套理论**——这会让审稿人误以为这是核心贡献，而它其实是细节修正。
- 在论文中只需 1/4 页带过：「在球面上对 paraphrase 编码做加权平均后归一化；这等价于 vMF MAP 的闭式解。」

#### Module 2：Confidence Memory（低熵） — **完全可行 ✅**

- 直接复用 Point-Cache 的 `update_cache()` 逻辑（见 `@/root/autodl-tmp/MCP-Point-Cache/Point-Cache/runners/model_with_hierarchical_caches.py:71-107`）
- 几乎零工程量。

#### Module 3：Compactness Memory（ICP + Chamfer） — **可行但工程量大 ⚠️**

| 子问题 | 难度 | 对策 |
|--------|------|------|
| ICP 配准对每对样本 O(N²) 调用 | 高 | 仅在更新缓存时调用（每类 K_comp×N_samples 次），且采样到 256 点；用 `open3d` 或 `pytorch3d.ops.iterative_closest_point` 批量化 |
| 不同损坏下 ICP 收敛性差异（dropout 下大量缺失） | 中 | 设置 ICP 失败的回退（直接用原始 CD 或纯特征距离），并在 A7 中报告 ICP 成功率 |
| Chamfer 距离与特征距离尺度不一致 | 中 | v2 用 LogSumExp 融合是合理的；但更简单的方法是先做 z-score 标准化再加权（实验 A2 应对比这两种） |
| ICP 对仿射变换非旋转损坏（如 scale）的处理 | 中 | ICP 默认刚性变换；如果用 scale-aware ICP 会引入额外参数，建议保持刚性 + 先做尺度归一化 |

**关键警告**：v2 提到「对 rotate 损坏，ICP 配准后的 CD 接近零」——这只在**同类**点云之间成立。**异类**点云配准后 CD 仍会保持较大值（这才是判别性来源）。表 4.3 中"两者互补性"的描述偏乐观，实际 A2 实验可能发现：在 dropout 下 ICP 失败率>30%，使得 ω 自适应策略复杂化。

#### Module 4：Boundary Memory — **可行 ✅，但与 Point-Cache 负缓存几乎重复**

- Point-Cache 已有 negative cache 用 `prob_map` 做软掩码（见 `@/root/autodl-tmp/MCP-Point-Cache/Point-Cache/runners/model_with_hierarchical_caches.py:215-216`）
- MCP 的 negative cache 也是同样思路
- 你叫它 "Boundary Memory" 改了名字，但**机制本质相同**
- **建议**：要么并入 Point-Cache 的 negative cache 而不假装是新东西；要么真正做出区别（例如基于"梯度方向不确定性"而不仅是熵区间），否则 A1 消融会显示 boundary memory 增益微弱（< 0.5%）

#### Module 5：Hierarchical（global + local） — **可行 ✅**

- Point-Cache 已经实现，patch_centers 由 K-Means 在 transformer 中间层 token 上聚类得到
- 直接复用，零创新成本，但要在论文里坦诚标注"层级表征沿用 Point-Cache"

#### Module 6：MCP++ 残差精修 — **可行但风险高 ⚠️⚠️**

- 见 1.3 节第 6 点。**强烈建议把 MCP-3D++ 作为可选模块**，主方法 MCP-3D（base 版本）保持 training-free，与 Point-Cache 一致的部署效率。

### 1.5 风险清单（按严重性排序）

| 优先级 | 风险 | 触发条件 | 对策 |
|--------|------|---------|------|
| 🔴 高 | **创新性被定性为"3D + MCP + BayesMM = 缝合"** | 审稿人没看到 ICP-CD 的真实价值，或紧致性-性能分析数据不漂亮 | 把 §3.5 ICP-CD 紧凑记忆库 + §3.3 紧致性诊断写成两个独立的"故事"，并各做一张 figure；摆明这是 3D 域不存在于 2D 的全新工具 |
| 🔴 高 | **MCP++ 残差在 3D 上退化** | 3D-文本对齐弱 → align loss 把特征推向不准文本锚点 | 默认主方法 base 版本不含残差；MCP-3D++ 作为可选模块；在 A6 中坦诚报告失败模式 |
| 🟡 中 | **ICP 在 dropout 损坏下失败** | ω 自适应策略需精细化，否则平均增益被拉低 | 在 §3.5 设置 ICP 失败的 fallback；A2 报告各损坏下 ICP 成功率；学习一个轻量的 ω 预测器（基于点云密度+特征熵） |
| 🟡 中 | **vMF vs Gaussian 数值差异 < 0.3%** | d=512/768 时高斯近似 vMF 已经很接近 | 不把 vMF 作为核心卖点；在 A10 中作为消融的一个对比项；保留它的理由是"模长归一保证类别公平性"而非"数值精度" |
| 🟡 中 | **整体增益相对 Point-Cache 仅 +1.5~2.5%** | 改进主要在特定损坏类型 | 加 per-corruption 表格 (A7)，重点突出 rotate/scale 上 +3~5% 的强增益；坦诚 dropout 下增益小 |
| 🟢 低 | **DeepSeek API 成本与可复现性** | 多类别×40 paraphrase 调用费用 | 一次性生成并 cache 成 JSON 文件随代码发布（参考 `Point-Cache/llm/*.json` 的做法） |
| 🟢 低 | **Boundary Memory 与 Point-Cache 负缓存重复** | 审稿人指出本质相同 | 要么并入正缓存的负向支路；要么改用基于 logits 梯度方向的新选择准则 |

### 1.6 可行性结论

**整体可行性：☑️ 可行，但要把"卖点"重新分配。**

> 你目前的 v1/v2 提案把六个新点都当"创新点"列在前面，但**审稿人只会接受其中 2~3 个真正的核心贡献**。建议重新组织叙事，让 ICP-CD 紧凑记忆库 + 3D 紧致性诊断成为论文的两条主线，其它（多缓存框架、层级表征、文本 vMF）作为支撑模块。

具体调整建议见**第二部分**修订版 Proposal。

---

## 第二部分：修订版完整 Proposal

> 这一部分是在第一部分诊断的基础上**重新组织**的论文方案，相比 v1（1538 行）/ v2（264 行）：
> - 大幅精简符号系统（删掉一半冗余符号）
> - **创新点从 6 个收敛到 3 个核心 + 2 个辅助**
> - 实验方案聚焦 8 组核心消融（v1 有 16 组，过多）
> - 残差精修降级为可选扩展（MCP-3D++）
> - 新增"诚实的失败模式"章节

### 2.1 论文定位与题目

**主推题目**：
> **Geometry-Aware Multi-Memory Test-Time Adaptation for 3D Point Cloud Recognition**

**副标题选项**：
- *Where Compactness Meets Geometry: Geometry-Aware Multi-Memory TTA for 3D*
- *MCP-3D: Bridging Cache-based 2D TTA and Hierarchical 3D Recognition with ICP-aligned Compactness*

**核心叙事**（一段话讲完）：
> 现有缓存式 TTA 方法（Point-Cache）在 3D 上仅依赖低熵选择，无法刻画"几何形状紧致性"；2D 中的多缓存方法（MCP）证明了"特征紧致性"是良好原型的关键，但其特征距离选择准则迁移到 3D 时面临 3D-文本对齐质量弱、特征空间无法解耦位姿与形状的双重困难。我们提出 **MCP-3D**，引入 **ICP 配准 + Chamfer 距离的几何紧致度量**，与特征熵协同选择高质量原型，并通过系统性的紧致性-性能相关性诊断揭示 3D 与 2D 在该规律上的本质差异。

**目标投稿**：CVPR 2027 / ICCV 2027 / NeurIPS 2026（视进度）。

### 2.2 重新定位的核心贡献（3 主 + 2 辅）

| # | 贡献 | 类别 | 论文页面占比 |
|---|------|------|-------------|
| **C1** | **ICP-aligned Chamfer Distance 作为 3D 原型选择的几何紧致度量**：首次在 TTA 中将"位姿对齐后的形状距离"作为缓存样本筛选的判别量；与特征距离构成**互补**而非冗余的双度量体系。 | 主 | §3.4（约 1 页） |
| **C2** | **3D 紧致性-性能相关性的系统性诊断**：揭示该相关性在 3D 中显著弱于 2D（r ≈ 0.65±0.08 vs 2D 的 0.82±0.05），且在 rotate/dropout 下呈现损坏特异性；为引入几何度量提供实证依据。 | 主 | §3.3（约 0.6 页） |
| **C3** | **多记忆库 × 层级表征的统一框架**：将 MCP 的功能分工（confidence/compactness/boundary）与 Point-Cache 的层级表征（global/local）正交融合为 2 × 3 记忆矩阵，并验证 6 个格子中至少 4 个具有显著边际增益。 | 主 | §3.2 + §3.5（约 0.8 页） |
| C4 | **超球面 vMF 文本锚点**：将 BayesMM 的高斯先验替换为方向统计的 vMF MAP，避免欧氏高斯在归一化特征上的模长衰减偏置；这是一个数值修正而非主创新。 | 辅 | §3.6（约 0.3 页） |
| C5 | **MCP-3D++（可选）**：在 3D-文本弱对齐场景下重新设计的残差精修，降低 align loss 权重，仅更新 top-k 混淆类残差。 | 辅 | §3.7（约 0.3 页） |

> **写作策略**：Introduction 只列 C1 / C2 / C3 三点；C4 / C5 在 Method 中以"补充模块"形式简短提及，不在 contribution 列表中出现。

### 2.3 符号系统（精简版，仅保留 12 个核心符号）

| 符号 | 含义 |
|------|------|
| $X_t \in \mathbb{R}^{N_t \times 3}$ | 第 $t$ 个测试点云 |
| $\mathbf{h}^g, \mathbf{h}^\ell \in \mathbb{S}^{d-1}$ | 全局 / 局部归一化特征 |
| $\mathcal{M}^{\text{conf}}_{*}, \mathcal{M}^{\text{comp}}_{*}, \mathcal{M}^{\text{bnd}}_{*}$ | 三类记忆库（$* \in \{g, \ell\}$） |
| $\mathbf{a}_c^T, \mathbf{a}_c^V \in \mathbb{S}^{d-1}$ | 文本 / 视觉锚点 |
| $\mathbf{m}_c, \hat{\kappa}_c$ | paraphrase 集合的均值方向 + vMF 浓度 |
| $d_{\text{feat}}, d_{\text{GEO}}$ | 特征角距离 / ICP 后 Chamfer 距离 |
| $\Omega(\cdot)$ | 异构融合距离（见 §3.4） |
| $\mathcal{H}(\cdot)$ | 预测分布熵 |
| $\zeta_1, \zeta_2, \zeta_3$ | 文本 / 记忆 / 边界 logits 融合权重 |
| $\rho$ | global vs local 平衡因子 |

**与 v1/v2 的差异**：去掉 $\beta, \kappa, \gamma, \xi, \iota, \tau_0^2, \kappa_0, \omega$ 等次要超参（在论文正文中只用默认值，附录给出敏感性分析）。

### 2.4 方法（精简版）

#### §3.1 Problem Setup

给定预训练 3D-VLM $(\mathcal{E}_{3D}, \mathcal{E}_{\text{text}})$，测试流 $\{X_t\}$ 在线到达，无标签、不可改权重。目标：最大化 $\lim_{T \to \infty} \frac{1}{T}\sum_{t} \mathbb{I}[\hat{y}_t = y_t]$。

#### §3.2 Multi-Memory Architecture (Overview)

2 × 3 记忆矩阵：

|        | Confidence ($\mathcal{M}^{\text{conf}}$) | Compactness ($\mathcal{M}^{\text{comp}}$) | Boundary ($\mathcal{M}^{\text{bnd}}$) |
|--------|-------------------------------------------|-------------------------------------------|----------------------------------------|
| Global | 低熵全局特征 | ICP 紧致全局特征 + 点云 | 边界全局特征 + 软标签分布 |
| Local  | 低熵局部 patch centers | ICP 紧致局部 patch centers | — *（局部边界库消融后发现增益 < 0.3%，删去）* |

> **关键实验决策**：先做 A1 消融决定保留哪些格子；论文只展示有显著贡献的子集（避免 v1 那种"6 个全保留"的过度声明）。

#### §3.3 Compactness-Performance Correlation in 3D（理论动机，C2）

定义类内紧致性：

$$\Phi(c) = \frac{1}{N_c} \sum_{i=1}^{N_c} \cos(\mathbf{h}_i^c, \bar{\mathbf{h}}^c)$$

**实验设计**：在 ModelNet-C 7 类损坏 × 3 严重等级 = 21 个分布上，分别测量 $\Phi$ 与 TTA 增益（vs zero-shot）的 Pearson 相关 $r$，与 2D（在 ImageNet-C 上重做 MCP 实验）对比。

**预期发现**：
- 2D 平均 $r \approx 0.82$
- 3D 平均 $r \approx 0.65$（**显著弱于 2D**）
- rotate 损坏下 $r$ 仍然高（≈ 0.75）但**绝对紧致性下降**（特征位置偏移）
- dropout 损坏下 $r$ 崩溃到 ≈ 0.50（紧致性失去预测力）

**结论**：单纯依赖特征紧致性的 2D MCP 方法在 3D 上注定增益有限；必须引入正交的几何信号 → 引出 §3.4。

#### §3.4 Geometry-Aware Compactness Memory（核心创新，C1）

**Step 1: ICP 配准**

对候选样本 $X_i$ 与同类参考样本 $X_j$，求最优刚性变换：

$$(\mathbf{R}^*, \mathbf{t}^*) = \arg\min_{\mathbf{R} \in SO(3), \mathbf{t} \in \mathbb{R}^3} \|\mathbf{R} X_i + \mathbf{t} - X_j\|^2$$

实现：FPS 下采样到 $N_{\text{ICP}} = 256$ 点，调用 `pytorch3d.ops.iterative_closest_point`，5 次迭代。

**Step 2: 对齐后 Chamfer 距离**

$$d_{\text{GEO}}(X_i, X_j) = d_{\text{CD}}(\mathbf{R}^* X_i + \mathbf{t}^*,\, X_j)$$

**Step 3: 异构距离融合**（用 z-score 归一化后的线性组合，**不用 LogSumExp**）

```
d_feat_norm = (d_feat - μ_feat) / σ_feat   # batch-wise
d_geo_norm  = (d_geo  - μ_geo)  / σ_geo
Ω = ω · d_feat_norm + (1 − ω) · d_geo_norm
```

> **设计决策**：v2 用 LogSumExp 是数学优雅但**调参困难**（β 取值敏感）；z-score 后的线性组合更稳健且每个样本只引入 2 个 batch 统计量。在 A2 消融中对比两种方案。

**Step 4: 紧凑样本选择规则**

样本 $i$ 进入 $\mathcal{M}^{\text{comp}}_c$ 当且仅当：
1. $\mathcal{H}(\mathbf{p}_i) < \tau_H$（足够自信）
2. $\Omega(i, \bar{\mathcal{M}}^{\text{conf}}_c) < \text{median}(\Omega \text{ within class } c)$（双重紧致）

**Step 5: ICP 失败回退**

如果 ICP 5 次迭代后位移残差 > 阈值（默认 0.05），认为配准失败，**该样本仅依赖特征距离**（即 ω = 1）。统计 A2 中报告各损坏下的 ICP 成功率。

#### §3.5 Confidence & Boundary Memories

直接复用 Point-Cache 与 MCP 的成熟设计，论文只用 0.3 页带过：
- **Confidence**：低熵筛选 + 熵加权聚合（multi-view augmentation V=64）
- **Boundary**：中熵区间样本 + 软标签分布 → 推理时减法抑制

#### §3.6 Hypersphere Text Anchor（C4）

```
m_c = Σ z_c^(m) / ||Σ z_c^(m)||
R̄_c = ||(1/M) Σ z_c^(m)||
κ̂_c ≈ R̄_c (d − R̄_c²) / (1 − R̄_c²)

a_c^T = (κ₀ z̄_c + κ̂_c m_c) / ||κ₀ z̄_c + κ̂_c m_c||
```

> 论文只写这 3 行公式；vMF / Bessel 等理论放附录。

#### §3.7 Inference & Optional Residual (C5)

最终预测：

$$\mathbf{l}_{\text{final}} = \zeta_1 \cdot \mathbf{h} \cdot (\mathbf{A}^T)^\top + \zeta_2 \cdot [\rho \mathbf{l}_g + (1-\rho) \mathbf{l}_\ell] - \zeta_3 \cdot \Psi(\mathbf{h}, \mathcal{M}^{\text{bnd}})$$

其中 $\Psi$ 是 boundary memory 的负向贡献（沿用 Point-Cache 实现）。

**MCP-3D++（可选）**：当算力允许时启用残差 $\Delta_c^V, \Delta_c^T$，损失 $\mathcal{L} = \mathcal{L}_{\text{ent}} + 0.3 \mathcal{L}_{\text{align}} + 0.2 \mathcal{L}_{\text{rep}}$（注意 align 权重 0.3，远低于 MCP++ 的 1.0），且只更新 top-3 混淆类残差，每 50 个样本重置以防漂移。

### 2.5 实验方案（精简到 8 组核心消融）

| 编号 | 实验 | 目的 | 优先级 |
|------|------|------|--------|
| **Main-1** | ModelNet-C 7 损坏 × 3 严重 × 4 backbone | 主实验 | ★★★★★ |
| **Main-2** | ScanObjectNN-C + Sim2Real + Objaverse-LVIS | 域泛化 / 真实 / 开词汇 | ★★★★★ |
| **A1** | 6 个记忆格子的组合消融（核心 8 组），**必含三项**：(i) 仅 global 行 vs 仅 local 行 vs 二者共用；(ii) 去掉 compactness 列观察掉点幅度；(iii) 融合权重 $\rho \in \{0, 0.3, 0.5, 0.7, 1\}$ 的 global/local 权衡曲线 | 验证 2 × 3 矩阵边际贡献 + global/local 互补性 | ★★★★★ |
| **A2** | $\alpha \in \{0, 0.3, 0.5, 0.7, 1\}$ × 7 损坏 + LogSumExp vs z-score 对比 + ICP 成功率统计 + **熵阈值 $\tau_{\text{conf}}$ 敏感性**（分位数 20%/30%/40%，global 和 local 分别扫） | 验证 C1 核心创新 + 阈值稳健性 | ★★★★★ |
| **A3** | 紧致性-性能相关性测量（2D vs 3D，21 个分布） | 验证 C2 理论贡献 | ★★★★★ |
| **A4** | 文本锚点：单模板 → 多模板 → LLM 平均 → vMF MAP → 高斯 MAP | 验证 C4 + 与 BayesMM 对比 | ★★★ |
| **A5** | 残差精修开/关 × 4 backbone × 7 损坏；记录失败模式 | 验证 C5（含负面结果） | ★★★ |
| **A6** | 推理时间 + 显存 + ICP 占比 | 效率分析 | ★★★ |

> **删除的实验**（v1 中冗余）：A4 容量敏感性（放附录）、A5 融合权重热图（放附录）、A11 DeepSeek vs ChatGPT（与论文主线无关，放附录）、A13/A14/A15/A16（合并到附录单页表格）。

### 2.6 预期结果（保守估计）

| 方法 | ModelNet-C 平均 | rotate | dropout_global | jitter |
|------|----------------|--------|----------------|--------|
| Zero-shot Uni3D | 65.3 | 48.5 | 60.2 | 70.1 |
| Point-Cache (CVPR'25) | 77.3 | 62.0 | 73.8 | 81.2 |
| BayesMM (CVPR'26) | 78.3 | 64.2 | 74.5 | 81.8 |
| **MCP-3D (ours, base)** | **79.5** | **67.0** | 74.2 | **82.5** |
| **MCP-3D++ (ours)** | **80.0** | **68.0** | 74.5 | 82.7 |

**诚实的负面结果**：
- dropout 下相对 BayesMM 仅 +0.0~0.3%（ICP 失败率高）
- 在 long-tail Objaverse-LVIS 上 MCP-3D++ 残差精修略低于 base 版本（0.2%）

> 主张「在 rotate / scale 类位姿损坏上 +3~5%」+「整体 +1.5~2% 稳定增益」就足以支撑顶会，不要追求"全场最佳"。

---

## 第三部分：实施路线图与对比说明

### 3.1 实施路线图（6 个月）

| 阶段 | 时间 | 任务 | 交付物 |
|------|------|------|--------|
| **P0** | 第 1 周 | 跑通 Point-Cache 官方代码（4 backbone × ModelNet-C） | baseline 复现表 |
| **P1** | 第 2-3 周 | 实现 vMF 文本锚点；与 Point-Cache 单模板对比 | 文本锚点模块 + 初步增益验证 |
| **P2** | 第 4-6 周 | 实现 ICP + Chamfer 紧凑记忆库（最关键模块） | A2 消融数据 |
| **P3** | 第 7-8 周 | 集成 2 × 3 记忆矩阵 + 多源融合 | Main-1 主实验数据 |
| **P4** | 第 9-10 周 | 紧致性-性能相关性诊断（2D 复现 + 3D 测量） | C2 figure + A3 数据 |
| **P5** | 第 11-13 周 | 残差精修（可选）+ 全部消融 | A1/A4/A5/A6 数据 |
| **P6** | 第 14-16 周 | 跨数据集验证（ScanObjNN/Sim2Real/Objaverse） | Main-2 数据 |
| **P7** | 第 17-22 周 | 论文撰写 + rebuttal 预演 | 投稿稿 |
| **P8** | 第 23-24 周 | 代码整理 + 投稿 | 投稿包 |

**关键里程碑**：P2 结束时如果 ICP-CD 紧凑记忆库的增益不显著（< 0.5%），**立即调整方案**——退化为"特征+CD 不带 ICP"的更简单设计，把 C1 重新定位为"几何信号作为辅助"。

### 3.2 风险对策清单

| 风险 | 触发判据 | 对策 |
|------|---------|------|
| ICP-CD 紧凑库增益 < 0.5% | A2 中所有 ω ∈ (0,1) 都不优于 ω = 1 | 移除 ICP 步骤，仅保留**尺度归一后的 CD**作为辅助度量；论文重新定位 C1 为"3D 几何信号融合"而非"位姿-形状解耦" |
| vMF 与高斯差异 < 0.1% | A4 显示两者在所有损坏下统计不显著 | 把 vMF 从 contribution 中删除，作为附录"实现细节" |
| MCP-3D++ 残差精修退化 | A5 中 4 个 backbone 至少 2 个出现负增益 | 论文中只把 ++ 作为"在某些场景下可启用的选项"，主体方法保持 base 版本 |
| Point-Cache 复现误差 > 0.5% | P0 阶段无法复现官方数字 | 直接 fork Point-Cache 源码做最小改动；论文报告"在我们的 setup 下基线数字" |
| 整体增益 < +1% | 主实验数字达不到投稿阈值 | 加 PointDA-10/40 跨域数据集；如仍不足，转投 WACV / 3DV / TPAMI |

### 3.3 与 v1（1538 行）/ v2（264 行）的差异说明

| 方面 | v1 | v2 | **本文档（v3）** |
|------|----|----|------------------|
| 创新点数量 | 6 | 6 | **3 主 + 2 辅** |
| 核心论证策略 | 全面铺开 | 修正 v1 的技术错误 | **聚焦 C1+C2，舍弃营销** |
| 文本建模 | 高斯 MAP | vMF MAP（修正） | **vMF MAP（降级为辅助）** |
| 度量融合 | 线性加权 | LogSumExp | **z-score + 线性（更稳健）** |
| 实验数量 | 16 组消融 | （未展开） | **8 组核心消融** |
| 残差精修地位 | 默认开启 | 默认开启 | **可选模块，主方法不含** |
| 失败模式讨论 | 无 | 无 | **专门一节诚实报告** |
| 文档篇幅 | 1538 行 | 264 行 | **约 350 行（聚焦版）** |

### 3.4 立即可执行的下一步（本周内）

1. **代码层面**：在 `Point-Cache/runners/` 下新建 `model_with_mcp3d.py`，从 `model_with_hierarchical_caches.py` fork 起步
2. **数据层面**：用 DeepSeek 一次性生成 ModelNet-40 / ScanObjectNN-15 / Objaverse-1156 的 paraphrase JSON（成本估计 < 50 元）
3. **实验层面**：先用 Uni3D + ModelNet-C rotate_2 这一个 setting 跑通 ICP-CD 紧凑库，**这是判断整个方案是否值得继续的关键决策点**
4. **论文层面**：先写 §3.3 紧致性-性能相关性的 2 页分析（这是即使主方法失败、仍能独立成稿的"保底贡献"）

---

## 附录：关键判断速查

> 写完之后回头读这一节，判断你的 paper 现在处于哪个状态。

### 性能目标三档声明（修补 G4，2026-05-10）

基于同类工作历史增益 anchor + W4 计划的 oracle ceiling 估计，设立三档验证目标。三档**都有对应 paper framing**，不是单档赌注：

| 档次 | 35-mean 增益 | scale 列单独 | 论文 framing | 对应下方三盏灯 |
|------|-------------|-----------|------------|--------------|
| **Floor 基础线** | ≥ +0.5pp | ≥ 0pp (至少不退化) | "首个针对 3D TTA 全局形变盲区的探索性工作"，定位为 position paper | Red light 临界 → Yellow 下沿 |
| **Target 目标线** | +1.0 ~ +2.5pp | +1 ~ +3pp | "新 SOTA + 失败案例修复"，AAAI 标准投稿 | Yellow → Green 过渡 |
| **Stretch 卓越线** | ≥ +3pp | ≥ +5pp | "范式级改进"，AAAI Oral / 期刊延伸可能 | Green light 稳定达成 |

#### 依据（外部 anchor + 内部 ceiling）

**外部 anchor**（同类 TTA 工作的历史增益）：
- TPT (NeurIPS'22) on ImageNet-C: **+1.2pp**
- TDA (CVPR'24) zero-shot TTA: **+0.8pp**
- Point-Cache (paper) on ModelNet-C 35-mean: **+3.1pp vs ZS**

**内部 ceiling**（W4 要做的 oracle 实验）：
- 把 cache 检索改成"已知 GT label，从 cache 挑同类样本"
- 这是任何 cache TTA 的天花板，会给出实际可达上界
- 若 oracle = 81% (vs ZS 78%)，目标线 +1-2.5pp 就有可解释的 plausibility
- 若 oracle 仅 79%，必须接受 +0.5-1pp 的现实，framing 转向"失败案例修复"

#### 三档决策规则

- 基础线 + Yellow light → **投 3DV / WACV**，不强投 AAAI
- 目标线 + Yellow-to-Green → **投 AAAI**，按原方案走
- 卓越线 + Green → **投 AAAI 并目标 Oral**，同时准备期刊扩展

#### 历史 Green/Yellow/Red 判据（保留，用作 project 生死决策）

**Green light**（值得全力推进）：
- ☑ ICP-CD 紧凑记忆库在 rotate 损坏上有 ≥ 2% 增益
- ☑ 紧致性-性能相关性 2D vs 3D 差异显著（差异 ≥ 0.1）
- ☑ 主实验整体 ≥ Point-Cache + 1.5%

**Yellow light**（继续推进但调整定位）：
- ⚠ ICP-CD 增益 0.5~2%：弱化 C1 表述，强化 C2 故事
- ⚠ 整体 +0.8~1.5%：转投 WACV / 3DV
- ⚠ vMF / 残差精修无显著增益：移到附录

**Red light**（重新设计）：
- ✗ ICP-CD 在所有损坏下增益 < 0.3%
- ✗ 紧致性-性能相关性 3D 与 2D 几乎一致（C2 失去新颖性）
- ✗ 整体增益 < +0.5%

→ **本周必须做的事**：把 Green/Yellow/Red 三个判据中的 ICP-CD 验证实验（在 Uni3D + ModelNet-C rotate_2 一个 setting 上）跑出来。这一组数据决定整个论文的走向。
