# Session 2026-05-10 晚（19:30 - 22:00）：论文整体复习启动 + SOP 锁定

> **承接**：`session_2026-05-10.md` 停在傍晚 14:50 的 W2 commit 计划。本文是同日晚上接续。
>
> **会话主题**：用户启动"论文整体复习"模式（暂停跑实验），按 "故事 → 鸟瞰 → 概念" 三段流程。本会话讲完前两段。

---

## 1. 模式切换：从"跑实验"到"复习论文"

用户明确意图："**整体研读 docs，对比上传的 Point-Cache 论文，巩固论文整体思路 + 缩写认识 + 找逻辑漏洞**"。理由：W2 复现已完成，W2.5 之前需要先把整体思路在脑子里跑一遍，避免到后期实验跑完才发现 motivation 出了问题。

约束：
- 用户会问很多具体问题（Cascade 不主动讲解，等问）
- 不熟悉英文缩写（每次都要中文+括号注明）
- 边复习边记漏洞（落到 `doc_gaps.md`）

落地动作：建 `doc_gaps.md` 漏洞日志（6 类: L/E/X/W/S/C × 3 级严重度）。

---

## 2. 第一阶段：故事 arc（讲完）

按 motivation → method → results 三段式给的"大白话版"：

**第一幕：要做什么**
- 一句话：做 3D 点云的测试时适配 (TTA) 方法，让 OpenShape 类零样本分类器在遇到现实扰动时自动恢复精度，且不重训

**第二幕：为什么值得做（两个痛点）**
- 痛点 1：现有 TTA 只用"特征余弦"，特征空间在 scale 上盲（F1: -0.40pp 实证）
- 痛点 2：单一文本锚点表达不了类内多样性

**第三幕：怎么做（三贡献）**
- C1 ICP-CD 几何距离（针对痛点 1）
- C2 vMF 文本锚点（针对痛点 2）
- C3 2×3 记忆矩阵（C1+C2 的集成框架）

**第四幕：期望什么结果**
- 35-mean +1~+3pp、scale 列拉回正增益、ScanObjectNN-C +1~+3pp、加诚实失败案例 (dropout sev=4)

**讲故事结尾给了 5 个追问钩子（G1-G5 漏洞）**：
- G1: C1 假设无证据 (E, 🔴)
- G2: vMF vs 平均区分度 (C, 🟡)
- G3: 6 格子无消融 (X, 🟡)
- G4: +1~+3pp 拍脑袋 (S, 🟡)
- G5: 紧致度定位暧昧 (W, 🟢)

---

## 3. 用户深度追问 5 题（全部答完 + 落盘）

### Q1: 第二幕痛点像 Related Work，能不能现在就写？另外 F1 现象是不是只在 hierarchical 出现？

**A1**:
- 写到 `MCM-PC/docs/paper/02_related_work.md` v0.1（三 subsection × "现有工作 → 局限 → 我们" 三步式）
- 跨条件验证立项为 **P5 实验**（D13）：跑 global cache TTA + (可选) ULIP-2 backbone，1-2 小时成本，决定 §2.1 末段写法
- 修补 G8 漏洞

### Q2: 应不应该一边写论文一边跑实验？

**A2**: 应该。已锁定 **D12: 写作-实验并行原则**。落地：
- 建 `MCM-PC/docs/paper/` 目录
- 写 `00_outline.md`（章节大纲 + 触发-写作映射表，每个 W*.* 完成立刻写对应 §X.Y）
- 写 `02_related_work.md` v0.1
- 每周 3-5 小时纯写作预算

### Q3: 2×3 矩阵第二行（低置信度样本）是负样本吗，能不能用损失函数排斥？

**A3**: **不是负样本**，是 **boundary memory（模糊样本）**：
- TTA 是 training-free，没有损失函数让你优化
- 低置信度记忆存的不只是特征，还存样本的 top-k 软标签 (e.g. `[chair: 0.4, sofa: 0.35, stool: 0.25]`)
- 用法：新样本和 boundary 余弦高 → 说明在模糊区 → 用 boundary 的软标签做加权融合，降低主预测置信
- 作用本质：**校准信号**，不是排斥
- 修补 G6 漏洞（W: docs 没说清）

### Q4: 为什么 vMF 用在文本不用在点云特征？

**A4**: 三个理由：
- 样本量不对称：文本侧每类 ~30 paraphrase 需要参数化拟合 vMF；点云侧 cache 里 ~100-500 个同类样本天然构成经验分布
- 算力成本：vMF 每个 batch 重新拟合 N 个类太贵；KNN 经验聚合更便宜
- 维度差异：文本 d=512 适合 vMF；点云 d=1280 vMF 优势衰减
- 修补 G7 漏洞（W: docs 没解释）

### Q5: G4 的 +1~+3pp 怎么改成更靠谱的目标？

**A5**: 改三档目标：
- 基础线 (Floor): ≥+0.5pp 35-mean / scale 列 ≥ 0pp / 论文最低成立点
- 目标线 (Target): +1~+3pp / scale +1~+3pp / 主卖点
- 卓越线 (Stretch): >+3pp / scale +5pp / AAAI Oral
- 配 W4 oracle 实验定 ceiling
- 配同类工作历史 anchor (TPT +1.2 / TDA +0.8 / Point-Cache +3.1)
- 修补 G4 漏洞

---

## 4. 第二阶段：鸟瞰（讲完）

16 周 PHASE A → AAAI 路径（用 ASCII art 给的）：

```
W1-2     ████ 复现 ✅（关键发现 F1-F5）
W2.5     ▒▒▒  探针 P3/P5/P1/P2 ⏳ NOW
W3       ▓▓   C2 vMF + §3.3   🚪 R2
W4       ▓▓   C1 ICP-CD + §3.4 🚪 R3
W5       ▓▓   C3 2×3 + §3.5
W6-7     ████ ModelNet-C 主实验 + §4.1 🚪 R4
W8       ██   ScanObjectNN-C + §4.2
W9       ██   紧致度诊断 + §5.1
W10      ██   BayesMM + §5.2
W11-12   ██   失败案例 + §5.3 + §6
W13-14   ███  收尾 polish
W15-16   ██   AAAI submit 🎯
```

**鸟瞰强调 3 件事**：
- W2.5 是 narrative 锁定阶段，不是实现阶段（先跑探针再决定 C1/C2 narrative）
- 4 个 RISK_GATE (R1-R4) 是提前撤退点，每个都有数值阈值
- 写作不是最后 2 周，是 14 周持续滚动（D12）

---

## 5. 用户驱动的关键流程：SOP 锁定（D14）

用户主动提出："以后每完成一个阶段的任务，需要写论文草稿、commit、保存对话内容、修复扩充实验计划等等，应该还有需要补充的，请你帮我想一下"。

落地：`MCM-PC/docs/project/MILESTONE_SOP.md` v1.0（213 行）

**核心 7 步 checklist（每次 W*.* 完成后约 60-90 min）**：
1. 实验数据归档 (10 min) → key_findings.md F#
2. 论文段落写作 (30-60 min) → docs/paper/0X_*.md，按顶会格式
3. 代码 commit + tag (10 min) → push 远程
4. 决策 + 漏洞维护 (10 min) → decisions.md D# / doc_gaps.md G# 增删
5. 会话 + 计划归档 (10 min) → chat_summary.md 阶段 N + 重写 next_steps.md
6. 审稿人攻击模拟 (10 min) → 找 3-5 个 reviewer comment
7. 下一阶段准备 (5 min) → GPU/数据/API/依赖

**附录 A-E**：
- A. 顶会论文段落格式（Topic+Evidence+Bridge / 主动语态 / 数字带单位）
- B. commit/tag 命名约定
- C. 长期项目健康（每周/每月 review + 算力预算 500 GPU·h）
- D. 投稿前 1 周自检
- E. SOP 自我演化机制

---

## 6. Cascade 帮想的 6 个补充项（用户没明说但应该补）

1. 🔴 **数据备份策略** — autodl 容器临时性，关键实验数据必须 git push + 异地备份
2. 🔴 **算力预算追踪** — 双 T4 16 周 ≈ 500 GPU·h，分配到各阶段（W6-7 主实验 120h，W13-16 写作+缓冲 90h），每周硬限 30h
3. 🟡 **审稿人攻击模拟** — 已加 SOP 步骤 6
4. 🟡 **投稿日历倒推** — AAAI 8月、CVPR 11月、ICCV 3月，反推 16 周节奏
5. 🟡 **导师/同行预 review** — W6+ 起每月找师兄过一遍 method
6. 🟢 **数据集 license 合规** — ModelNet40/ScanObjectNN/Objaverse license 必须显式声明

---

## 7. 用户最后的指令（22:00）

1. **文档长度规则**："如果文档太长了，可以拆成多个文档，比如确保每个文档只存多少内容，然后依次对这些文档编号即可"
   - → 加入 SOP 附录 E（待办）
   - 单文件 300 行上限，超过按主题拆分编号
2. **保存所有历史对话**：用户即将新开窗口
   - → 落地：本 session 文件 + 更新 INDEX 恢复指引

---

## 8. 本次会话产出文件清单

| 文件 | 路径 | 行数 |
|---|---|---|
| 新建 paper outline | `MCM-PC/docs/paper/00_outline.md` | ~75 |
| 新建 §2 草稿 v0.1 | `MCM-PC/docs/paper/02_related_work.md` | ~95 |
| 新建里程碑 SOP v1.0 | `MCM-PC/docs/project/MILESTONE_SOP.md` | 213 |
| 新建漏洞日志 G1-G8 | `docs/context/windsurf/doc_gaps.md` | 101 |
| 新建本 session 笔记 | `docs/context/windsurf/session_2026-05-10_evening.md` | （本文） |
| 更新决策 D10-D14 | `docs/context/windsurf/decisions.md` | 234 |
| 更新阶段 8 + 当前位置 | `docs/context/windsurf/chat_summary.md` | 192 |
| 重写 W2.5 探针计划 | `docs/context/windsurf/next_steps.md` | 198 |
| 更新 INDEX 索引 | `docs/context/windsurf/INDEX.md` | 待更新 |

---

## 9. 下次会话恢复路径

按 `INDEX.md` 标准恢复指令读：
1. `INDEX.md`（含本会话路径）
2. `MCM-PC/docs/project/MILESTONE_SOP.md`（**新增必读，每次会话先看 SOP**）
3. `user_preferences.md`
4. `chat_summary.md`（看到阶段 8）
5. `next_steps.md`（当前 W2.5 暂停 + §2 草稿 v0.1）
6. `decisions.md`（D1-D14）
7. `doc_gaps.md`（G1-G8 待修）
8. `key_findings.md`（F1-F5 实证）
9. `session_2026-05-10_evening.md`（**本文，今晚详细笔记**）

**当前位置**：用户处于"论文复习暂停"状态，已讲完故事 + 鸟瞰，**等用户主动说"进概念"才启动第三阶段**：从 `MCM-PC/docs/concepts/00_overview.md` 开始逐个深入 vMF / ICP-CD / 紧致度 / 2×3 矩阵。

**绝不要做**：
- 主动讲第三阶段概念（用户驱动节奏）
- 重新讲故事或鸟瞰（除非用户明确要求）
- 启动 W2.5 P3-P5 实验（用户已 pause，复习完才启动）
- 破坏已锁的 D1-D14 决策
