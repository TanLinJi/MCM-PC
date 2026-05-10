# MCP-3D 项目里程碑完成 SOP

> **标准操作流程** (Standard Operating Procedure)
>
> **用途**：每完成一个 W*.* 实验里程碑、写作里程碑或重要决策后，按本 checklist 逐项执行。
>
> **维护**：本 SOP 由用户和 Cascade 共同维护。新经验追加到附录 E (changelog)。
>
> **版本**：v1.0 (2026-05-10)

---

## 1. 适用场景

✅ **实验类**：W2.5 任意探针 (P1-P5) / W3-W12 任意主实验完成
✅ **写作类**：任意 §X.Y 章节 v0.1+ 草稿写完
✅ **决策类**：用户锁定一条新决策 D# / 新方法 narrative pivot
❌ **不适用**：日常 debug、单次代码微调

---

## 2. 核心 Checklist（约 60-90 min 完整流程）

### 步骤 1 — 实验数据归档（约 10 min）

- [ ] 关键数字记入 `windsurf对话/key_findings.md`（追加 F# 编号）
    - 必含：出处（哪个实验跑出来）+ 数字 + 解读 + 对哪个 contribution 影响
- [ ] 日志/结果文件留在 `Point-Cache/results/` 或 `Point-Cache/logs/`
- [ ] `Point-Cache/progress.txt` 追加一行 NOTE
- [ ] 若有 figure/plot：保存到 `MCM-PC/figures/wN_*.svg`（矢量优先）

### 步骤 2 — 论文段落写作（约 30-60 min）

> 严格按 AAAI / CVPR 论文格式写。详见**附录 A**。

- [ ] 查 `docs/paper/00_outline.md` **触发-写作映射表**，确认要写哪个 §X.Y
- [ ] 写第一稿到 `docs/paper/0X_*.md`（版本号 v0.1 → v0.2 递增）
- [ ] 用附录 A 自检每段格式
- [ ] `00_outline.md` 状态栏更新（⏳ → 🟡 v0.X → ✅）

### 步骤 3 — 代码 commit + tag（约 10 min）

- [ ] `git status` 检查工作树
- [ ] 按 feat / fix / docs / test 拆分 commit
- [ ] commit message 用**附录 B** 模板
- [ ] 大里程碑结束打 git tag（`w3-vmf-anchor` / `paper-§3.4-draft`）
- [ ] `git push origin master --tags`（autodl 容器是临时的，必须及时 push）

### 步骤 4 — 决策 + 漏洞维护（约 10 min）

- [ ] 新决策追加 `windsurf对话/decisions.md`（D# 编号）
- [ ] 新漏洞追加 `windsurf对话/doc_gaps.md`（G# 编号）
- [ ] **重新评估全部待修复 G#**：本阶段是否修补了某条？
    - 修补的 → 移到已修复区，写明"由 W*.* 哪步骤修补"
- [ ] **风险闸 (RISK_GATE) 自检**：本阶段触发了哪个 R#？记录决策（继续 / pivot / debug）

### 步骤 5 — 会话与计划归档（约 10 min）

- [ ] `windsurf对话/chat_summary.md` 加"阶段 N"（按时间顺续编号 + 关键洞察）
- [ ] `windsurf对话/next_steps.md` 重写：删已完成、加新待办、更新顶部"当前阶段"
- [ ] 必要时建 `windsurf对话/session_YYYY-MM-DD.md` 单次详细笔记

### 步骤 6 — 审稿人攻击模拟（约 10 min）

> 假装你是 AAAI / CVPR reviewer，找 3-5 个最可能的攻击点。

- [ ] 列出 3-5 个 reviewer comment（按"严重度 × 概率"排序）
- [ ] 每条关联或新建 G# 漏洞（已存在则关联，不存在则新建）
- [ ] 评估当前进度能否 rebut；不能 → 反馈到下阶段实验/写作计划

### 步骤 7 — 下一阶段准备（约 5 min）

- [ ] 下阶段 prerequisite 是否就绪？（GPU 时间 / 数据下载 / API key / 依赖版本）
- [ ] 提前预约 / 下载 / 申请
- [ ] 关键依赖 freeze 到 `requirements_w*.txt`（每阶段独立一份，方便回滚）

---

## 附录 A：顶会论文段落写作格式规范

### A.1 段落结构（IMRaD-friendly: Topic + Evidence + Bridge）

每段约 80-150 词：

1. **首句 (Topic Sentence)**：本段主旨，单句陈述，不留悬念
2. **中段 (Evidence)**：至少一处具体证据
    - 实验数字（必带单位：76.5%, 3.2 ms, +1.2pp）
    - 文献 citation（暂用 `[需补 ref]` 占位符）
    - 表/图引用（Tab. 4, Fig. 3）
3. **末句 (Bridge)**：衔接下段 OR 回到论文 thesis

### A.2 语态与术语

- ✅ **主动语态**：`We propose ...` / `We observe ...` / `Our method ...`
- ✅ **缩写第一次出现展开**：`Test-Time Adaptation (TTA)`、`Iterative Closest Point (ICP)`
- ✅ **数字必带单位**：`76.5%`, `3.2 ms`, `+1.2pp`, `~1024 points`
- ❌ **禁用空泛词**：`very` / `really` / `extremely` / `obviously`
- ❌ **禁用第二人称**：避免 `you can see ...`，改 `One observes ...` 或 `Tab. X shows ...`
- ❌ **禁用过度形容**：`a novel method` 改 `a method that addresses ...`

### A.3 引用格式（按目标会议）

- **AAAI**：行内 `(Author 2024)`；参考文献 ACL 风格
- **CVPR / ICCV / ECCV**：行内 `[1]`；参考文献 IEEE 风格

### A.4 数学符号约定

- 集合用 `\mathcal{X}`、向量用 `\mathbf{x}`、矩阵用 `\mathbf{X}`、标量用 `x`
- 损失函数 `\mathcal{L}`、概率分布 `p(\cdot)`、期望 `\mathbb{E}[\cdot]`
- 第一次出现的符号必须立即定义

### A.5 段落示例（写得对的）

> The key limitation of feature-cosine matching emerges under global-shape corruptions. In our reproduction (Sec. 4.1), Point-Cache's hierarchical Test-Time Adaptation (TTA) degrades by **-0.40 percentage points (pp)** on the *scale* corruption, with 4 of 5 severity levels showing negative gains. The mechanism is intuitive: scale shifts the entire feature manifold uniformly, causing nearest-neighbor retrieval to surface co-corrupted samples that reinforce the wrong prediction rather than correct it. **This motivates our Iterative Closest Point with Chamfer Distance (ICP-CD) signal**, which operates orthogonally to the feature space and is invariant to global scale on normalized point clouds.

---

## 附录 B：Commit / Tag 命名约定

### B.1 Commit Type Prefix

- `feat`: 新代码功能
- `fix`: bug 修复 / 漏洞修补
- `docs`: 文档（论文段落、README、SOP、笔记）
- `test`: 测试代码
- `chore`: 环境 / 依赖 / 杂项
- `refactor`: 代码重构

### B.2 Commit Message 模板

```
实验：feat(wN): implement X for §Y
       (e.g.) feat(w3): implement vMF anchor with MAP estimation

写作：docs(paper): write §X.Y first draft (vN.X)
       (e.g.) docs(paper): write §3.4 ICP-CD first draft (v0.1)

修补：fix(gN): address gap GN by Y experiment/section
       (e.g.) fix(g6): clarify boundary memory role in §3.5

修订：docs(paper): revise §X.Y vN.X with W{n} data
       (e.g.) docs(paper): revise §2.1 v0.2 with P5 cross-method evidence
```

### B.3 Git Tag 模板

- 实验里程碑：`wN-experiment-name`（`w3-vmf-anchor`, `w4-icp-cd`, `w5-2x3-matrix`）
- 写作里程碑：`paper-§X.Y-draft`（`paper-§3.4-draft`）
- 大阶段完成：`phase-A-wN-complete`（`phase-A-w8-complete`）
- 投稿快照：`submit-aaai-2027`

### B.4 Push 频率

- ✅ **实验完成立刻 push**（autodl 容器临时，防数据丢失）
- ✅ **写作每段写完即 push**
- ✅ **重要决策立即 push**

---

## 附录 C：长期项目健康（每周 / 每月 review）

### C.1 每周 review（建议周一花 30 min）

- [ ] **进度对照**：实验进度 vs 16 周时间线（参 `chat_summary.md`）
- [ ] **算力预算**：GPU 已用小时数 vs 总预算（建议每周记录到 `progress.txt`）
- [ ] **数据备份**：本周关键实验结果是否已 git push + 异地备份（云盘 / 本地）
- [ ] **漏洞推进**：`doc_gaps.md` 里的 G# 是否在持续被推进、修补

### C.2 每月 review（建议月初花 1 h）

- [ ] **投稿日历**：AAAI / CVPR / ICCV / ECCV / NeurIPS deadline 倒推，决定下月节奏
- [ ] **依赖锁定**：`requirements.txt` / `environment.yml` 是否需要更新
- [ ] **图表风格**：所有 figure 是否用统一 matplotlib rcparams（字体 / 配色 / DPI）
- [ ] **导师 update**：每月给老师/导师发一次 1-page progress report
- [ ] **同行预 review**：W6+ 起每月找一个同学/师兄过一遍 method 描述

### C.3 算力预算建议（双 T4 16GB × 16 周）

| 阶段 | GPU 小时（双卡总和） | 累计 |
|---|---|---|
| W1-W2 复现 | 30 h | 30 |
| W2.5 探针 | 20 h | 50 |
| W3-W5 实现 + 单 backbone 验证 | 60 h | 110 |
| W6-W7 ModelNet-C 全实验 | 120 h | 230 |
| W8 ScanObjectNN-C | 60 h | 290 |
| W9-W10 消融 + 诊断 | 80 h | 370 |
| W11-W12 失败案例 | 40 h | 410 |
| **W13-W16 写作 + 补实验缓冲** | **90 h** | **500** |

→ 预算约 500 GPU·h（双卡总和）= 250 真实小时 / 双卡 = 约 10.4 天连续。**每周硬限 30 GPU·h**，超支立即报警。

---

## 附录 D：投稿前 1 周自检

- [ ] **Blind submission**：去除所有作者信息（脚本 `scripts/blind.sh`，待 W14 写）
- [ ] **数据集 license**：ModelNet40 / ScanObjectNN / Objaverse 的 license 合规检查
- [ ] **Supplementary material**：决定哪些放正文（≤ 8 页）哪些放附录
- [ ] **Plan B 投稿**：投稿失败的 fallback 期刊/会议（`PROJECT_HEALTH.md` 维护）
- [ ] **Reproducibility statement**：明确开源时间表 + 公开数据集说明
- [ ] **Ethics / Limitations 章节**：AAAI / CVPR 强制要求

---

## 附录 E：SOP 自我演化

- 用户或 Cascade 在使用过程中发现新流程经验，追加到本 SOP 对应附录
- 每次重大更新升版本号（v1.0 → v1.1 → v2.0），写 changelog
- 在 `decisions.md` 同步一条对应 D#

### 文档长度管理规则（v1.1 加入）

- 单 markdown 文件**软上限 300 行**
- 超过 300 行时按主题拆分，命名约定：
  - 按子主题：`<原名>_part1.md` / `<原名>_part2.md`
  - 按时间段：`<原名>_phase1-5.md` / `<原名>_phase6+.md`
- 拆分后必须在 `windsurf对话/INDEX.md` 或 SOP 自身的引导段加导航
- 拆分时**保留旧文件**（重命名为 `_archive` 后缀），不要直接覆盖丢历史

### Changelog

- **v1.0 (2026-05-10 21:30)**：初版。基于 W2 复现完成 + 用户主动提出"流程化"需求，将 D9-D14 的隐性实践经验固化为显性 SOP。
- **v1.1 (2026-05-10 22:00)**：用户提出"文档太长可拆分编号"，加入"文档长度管理规则"段，软上限 300 行。
