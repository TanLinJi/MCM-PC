# 下一步任务（MCP-3D 项目）

> 更新时间：2026-05-10 21:10
> 当前阶段：W1 ✅ → W2 ✅（bar1+bar2+bar3 全部 commit + tag w2-tta-baseline）→ **W2.5 待开始**（探针 P3/P5/P1/P2）+ **§2 论文草稿启动**

---

## 0. SOP（每次执行前必读）

每完成一个 W*.* 任务后，按 `MCM-PC/docs/project/MILESTONE_SOP.md` v1.0 走 7 步流程：

1. 实验数据归档 → `key_findings.md` 加 F#
2. 论文段落写作 → `docs/paper/0X_*.md` 按顶会格式
3. 代码 commit + tag → push 远程
4. 决策 + 漏洞维护 → `decisions.md` 加 D# / `doc_gaps.md` 加/移 G#
5. 会话归档 → `chat_summary.md` 加阶段 N + 重写本文件
6. 审稿人攻击模拟 → 找 3-5 个 reviewer comment
7. 下一阶段准备 → GPU / 数据 / API / 依赖

**不照做的后果**：实验数据丢、论文写作积累、漏洞被遗忘——博士最后半年崩盘的典型路径。

详见 `MCM-PC/docs/project/MILESTONE_SOP.md`（v1.0, 2026-05-10）。

---

## A 当前模式：D19 P4-fast-track ICP-CD scale-only oracle（进行中，2026-05-11）

按 D19 锁定，跳过 P3/P5/P1 顺序，直接验证 ICP-CD 在 ModelNet-C scale_2 上能否拉回 ≥0pp。

### 当前状态（2026-05-11 12:05，C 计划 smoke 跑中）
- ✅ v0.1：alpha=2, no ZM → smoke 50/scale_2: 72 vs 72 (持平)
- ✅ v0.1.1：alpha=6, ZM=True → smoke 50/scale_2: 70 vs 72 (-2pp)
- ⏳ v0.1.2：+ entropy gating (threshold=0.5) → C 计划 smoke 进行中
- ⏸ STAGE=full 5 severity x 2 行（90 min）等 smoke 出结果决定

### 暂停的并行任务（D19 lock 期间不做）
- 论文复习流程（"故事 → 鸟瞰 → 概念" 三段，已暂停）
- W2.5 P3/P5/P1（按 D11 顺序，D19 完成后回到）

### Pending decisions（OPEN）
- **D21 超参收敛**：D19 已引入 4 个核心超参 (α_g, β_g, zero_mean, τ_g)，等 C 计划 smoke
  出结果后处理：
  - acc ≥ baseline → 调默认值复用原 α/β/τ + 写 sensitivity table（方案 A+B）
  - acc < baseline → pivot 到 Fix D oracle 验证 ICP-CD 信号本身，超参延后

---

## B W2.5 探针实验（约 2-3 天，4 个，复习结束后启动）

按 D11 锁定执行顺序：影响越大的越先做。

### P3 — T4 显存可行性（0.5 天，最先做）

- **做什么**：开 OpenShape vitg14-rgb backbone + CLIP bigG-14 + V=32 多视角 + 长 cache，监控 GPU memory
- **判定**：≤ 14 GB 通过；OOM 则切 vitb16 / ULIP-2 base / feature 预编码到磁盘
- **影响**：架构级——失败则 PHASE A 整体路径要换 backbone
- **产物**：`Point-Cache/probes/p3_memory.py` + 在 `key_findings.md` 追加 F6 + `docs/project/progress.txt` NOTE

### P5 — 跨方法 scale 退化验证（0.5-1 天，与 P3 串行复用 backbone）

> D13 新增。修补 G8 漏洞。

- **做什么**：在 ModelNet-C scale 5 个 severity 上跑 global cache TTA，对比 ZS / global / hierarchical
- **判定**：
  - global 也退化 → F1 升级为 "特征余弦类共性"，§2.1 末段强写
  - global 不退化 → F1 仅限 hierarchical，C1 narrative 调整为 "针对层级缓存的退化模式"
- **影响**：决定 §2.1 Related Work 末段写法
- **产物**：`Point-Cache/probes/p5_scale_cross_method.sh` + `key_findings.md` 追加 F7
- **可选扩展**：跑 ULIP-2 backbone 同样 scale（+1 天）

### P1 — feature distance failure probe（1 天）

- **做什么**：100 样本 × 多种可控 corruption（rotate / scale / dropout，可先从 5 个旋转角度开始）过 OpenShape 编码器，测 clean-corrupted pair 的特征余弦相似度与最近邻 rank
- **判定**：
  - clean-corrupted cosine 始终 > 0.95 且 rank 稳定 → encoder 对该 corruption 已足够不变，C1 narrative 必须改 "corruption-specificity"
  - cosine 明显下降或 clean counterpart 跌出 top-k → feature distance 确有局部 failure，C1 motivation 可保留
- **影响**：决定 C1 narrative + §3.4 motivation 段（修补 G1 一部分），为 diagnosis-driven TTA 卖点提供前置证据
- **产物**：`Point-Cache/probes/p1_feature_distance_failure.py` + `key_findings.md` 追加 F8

### P2 — 紧致度-精度相关性（1-2 天）

- **做什么**：ModelNet-C 7 corruption × severity=2，测每类紧致度 vs 每类精度的 Spearman 相关 r_3D
- **判定**：
  - |r_3D − r_2D| < 0.05 → C2 卖 2D vs 3D 一致性
  - |r_3D − r_2D| ≥ 0.05 → C2 卖 3D 损坏特异性
- **影响**：决定 C2 narrative + §3.3 写作 + 紧致度作为独立 contribution C5 的归属（修补 G5）
- **产物**：`Point-Cache/probes/p2_compactness.py` + `key_findings.md` 追加 F9

### P4 — ICP 残差分布

延后到 W4，等 ICP 代码写好顺手做。

新增要求（2026-05-11）：
- **feature-vs-geometry ROC/AUC**：比较 feature distance 和 ICP-CD 在 same-class / different-class pair 上的区分能力
- **相似异类 class-pair 分析**：专门检查 chair/sofa/stool/table 等形状相近类别是否 CD 也很小
- **CD margin gating**：记录 top-1 CD 与 top-2 CD 的差距，用于决定是否降低 geometry 权重
- **判定**：
  - ICP-CD AUC > 0.85 且 hard pairs margin 可分 → C1 可作为主预测证据
  - ICP-CD AUC < 0.7 或 hard pairs 不可分 → C1 降级为辅助/诊断信号，预测时 fallback 到 feature/text

---

## C 写作并行任务（持续，参 D12）

### 已落地（2026-05-10）

- [x] 建 `MCM-PC/docs/paper/` 目录
- [x] `00_outline.md`：章节大纲 + 触发-写作映射表
- [x] `02_related_work.md`：v0.1 草稿（用 F1-F5 + 现有 baseline 知识）

### 用户反馈待办

- [ ] 评 §2 草稿三段的论证逻辑
- [ ] 评 "局限 → 贡献" 对应表的 anchor 是否清楚
- [ ] 决定 [需补 ref] 的 citation 是现在补还是 W13 收尾时补

### 写作触发表（参 `docs/paper/00_outline.md`）

```
W2.5 P1 完成 → §1 motivation 段填 F3 + P1 旋转鲁棒性
W2.5 P2 完成 → §1 motivation 段填 F1 + P2 紧致度 r
W2.5 P5 完成 → §2.1 末段填跨方法 scale 证据
W3 完成     → §3.3 vMF 锚点
W4 完成     → §3.4 ICP-CD
W5 完成     → §3.5 2×3 矩阵
W6-7        → §4.1 主结果
W8          → §4.2 真实场景
W9-10       → §4.3 消融 + §5.1 紧致度诊断
W11-12      → §5.2-5.3 + §6
W13-16      → 收尾 + 投稿
```

预算：每周 3-5 小时纯写作，每段写完 commit。

---

## D 漏洞修补任务（持续，参 `docs/context/windsurf/doc_gaps.md`）

当前 8 条 G1-G8。每个实验里程碑结束时回看是否能修补，修补后从待修复区移到已修复区。

| 漏洞 | 哪个实验/写作里程碑修补 |
|---|---|
| G1 (ICP-CD 假设无证据) | W2.5 P5 + W4 主实验 |
| G2 (vMF vs 平均区分度) | W3 数值仿真 |
| G3 (6 格子先验) | W5 leave-one-out |
| G4 (+1~+3pp 拆解) | 现在写三档目标到 docs |
| G5 (紧致度定位) | W9 实验后 + §5 写作 |
| G6 (低置信度作用) | 现在加 §3.5 注释 |
| G7 (vMF 不在点云侧) | 现在加 §3.3 末段 |
| G8 (scale 跨方法) | W2.5 P5 |

→ G4/G6/G7 是 "立刻可修" 的写作类漏洞，复习期间可以处理。

---

## E W3-W5 实现任务（W2.5 完成后）

### W3 — 实现 C2 vMF 文本锚点

```
步骤：
1. 用 DeepSeek API 跑 generate_paraphrase.py 生成 ModelNet40 paraphrase
2. 在 model_with_mcp3d.py Section 3 实现 compute_vmf_anchor
3. 对比 vMF / 简单平均 / 单点 三种锚点的 ZS 精度
4. 同时做附录 A 数值仿真（修补 G2）
5. 写 §3.3

风险闸 R2：
- 差 < 0.1pp → vMF 降附录
- 差 ≥ 0.1pp → vMF 进主 method
```

### W4 — 实现 C1 ICP-CD 几何距离

```
步骤：
1. 实现 PCA 主轴 + ICP 精调 + Chamfer Distance（pytorch3d）
2. 跑 P4：1000 对样本 ICP 残差分布定阈值
3. ICP-CD oracle 实验：scale_2 上单 ICP-CD 是否 ≥ ZS（修补 G1 sanity check）
4. 写 §3.4

风险闸 R3：
- ICP-CD 在 scale_2 上 < ZS → C1 motivation 失败，pivot
- ≥ ZS → C1 通过
```

### W5 — 实现 C3 2×3 矩阵

```
步骤：
1. 实现 6-cell 结构 + z-score 跨格融合
2. 6-cell leave-one-out 消融（修补 G3）
3. 写 §3.5（含 G6 boundary memory 澄清）
```

---

## F 长期里程碑（W6-W16）

- W6-W7：ModelNet-C 全 35 setting 主实验 + §4.1 写作
- W8：ScanObjectNN-C 真实场景 + §4.2 写作
- W9：紧致度诊断 + §5.1 写作（修补 G5）
- W10：BayesMM 跨域对照 + §5.2 写作
- W11-12：失败案例分析 + §5.3 + §6
- W13-14：abstract / intro / 整体收尾
- W15-16：投稿润色 → AAAI 投稿

详见 `docs/project/progress.txt` 和 `chat_summary.md`。

---

## G PHASE B 扩展（W17-W28，暂搁置）

- ULIP-2 cross-backbone validation
- Sim2Real (ScanObjectNN OBJ_BG)
- Objaverse-LVIS 大规模零样本
- CVPR rebuttal 强化
