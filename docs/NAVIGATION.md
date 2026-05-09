# MCP-3D 项目导航（你的"地图"）

> 这份文档帮你快速搞清楚：**所有文件在哪里、每个文件干什么、推荐什么阅读顺序、当前进度到哪了**。
> 如果你只想看一份文档，看这一份就够了。

---

## 1. 项目目录结构总览

```
/root/autodl-tmp/MCP-Point-Cache/                  ← 项目根目录
│
├── 📄 NAVIGATION.md                    ← 你正在看的这份导航文件
│
├── 📚 思路演化文档（4 份，按时间从老到新）
│   ├── MCP3D_framework.md              ← 早期初步框架（最早的草稿）
│   ├── MCP3D_full_proposal.md          ← v1 完整提案（1538 行，太长，作为素材库）
│   ├── MCP3D_full_proposal_v2.md       ← v2 数学修正版（264 行，修正 v1 的公式问题）
│   └── MCP3D_feasibility_and_proposal.md  ← v3 可行性诊断 + 修订版提案【⭐ 最新理论文件】
│
├── 📋 执行文件（W1 我刚创建的 4 个）
│   ├── progress.txt                    ← 进度追踪（每周勾选 / Risk Gate）
│   ├── setup_env.sh                    ← 环境配置脚本（运行一次即可）
│   ├── download_data.sh                ← 数据 + 模型权重下载（运行一次即可）
│   └── generate_paraphrase.py          ← DeepSeek API 生成文本 paraphrase
│
├── 📖 docs/concepts/                   ← ⭐ 概念讲解（建立思想认知，30-35 分钟读完）
│   ├── README.md                       ← 文件夹索引 + 推荐顺序
│   ├── 00_overview.md                  ← 大背景与项目全貌（5 分钟）
│   ├── 01_vmf_anchor.md                ← 概念 1：vMF 文本锚点（7 分钟）
│   ├── 02_icp_cd_distance.md           ← 概念 2：ICP-CD 几何距离（8 分钟）
│   ├── 03_compactness_diagnosis.md     ← 概念 3：紧致性诊断（6 分钟）
│   └── 04_2x3_memory_matrix.md         ← 整合：2×3 记忆矩阵（5 分钟）
│
├── Point-Cache/                        ← 原始 Point-Cache 代码库（CVPR'25 论文）
│   ├── runners/
│   │   ├── model_with_hierarchical_caches.py    ← 原 Point-Cache 核心代码（参考）
│   │   ├── model_with_mcp3d.py         ← 我新建的 MCP-3D 骨架【⭐ 你的主战场】
│   │   ├── model_with_global_cache.py  ← 原代码其他变体（不动）
│   │   └── ...
│   ├── datasets/                       ← 各数据集加载器（不动）
│   ├── llm/                            ← 文本 paraphrase JSON（一会儿会生成新的）
│   ├── utils/                          ← 工具函数（不动）
│   ├── models/                         ← 3D-VLM 模型（OpenShape/Uni3D/ULIP，不动）
│   └── scripts/                        ← shell 脚本（一会儿会加新的）
│
└── MCP/                                ← 原 MCP 2D 代码库（ICCV'25，参考）
    └── ...

/root/.windsurf/plans/                            ← 计划文档（在工作区外）
├── mcp3d-publication-plan-f3c00d.md              ← 第 1 版投稿计划（26 周）
└── mcp3d-logic-audit-and-revised-plan-f3c00d.md  ← 第 2 版审计 + 修订计划（28 周）【⭐ 最新】
```

---

## 2. 文档详细说明（每份文件干什么）

### A. 思路演化（4 份 .md，从老到新）

| 文件 | 行数 | 角色 | 是否需要看 |
|------|------|------|-----------|
| `MCP3D_framework.md` | ~ | 最早的草稿，思路雏形 | ❌ 已过时 |
| `MCP3D_full_proposal.md` | 1538 | v1 完整提案，最详细但有数学错误 | 📚 当素材库 |
| `MCP3D_full_proposal_v2.md` | 264 | v2 数学修正版（vMF / SLERP 等公式） | 📚 附录参考 |
| `MCP3D_feasibility_and_proposal.md` | 330 | **v3 可行性诊断 + 精简提案** | ⭐ **必读** |

**v3 文件 (` .md`) 是你理解"为什么做这个研究"的关键文档**，分三部分：
- 第一部分：可行性诊断（创新性、技术难点、风险）
- 第二部分：修订版完整 Proposal（精简方法 + 实验 + 时间线）
- 第三部分：实施路线图与风险对策

### B. 计划文档（2 份，在 `/root/.windsurf/plans/`）

| 文件 | 行数 | 角色 | 是否需要看 |
|------|------|------|-----------|
| `mcp3d-publication-plan-f3c00d.md` | 251 | v1 计划：26 周双投 AAAI/CVPR 时间表 | ❌ 已被 v2 取代 |
| `mcp3d-logic-audit-and-revised-plan-f3c00d.md` | 354 | **v2 计划：26 个 gap 审计 + 28 周修订时间表** | ⭐ **必读** |

**v2 计划是当前的执行依据**，分五部分：
- 第一部分：26 个逻辑漏洞清单（按 🔴 阻断 / 🟡 重要 / 🟢 细化 分级）
- 第二部分：补充实验/分析/说明清单（24 项）
- 第三部分：修订时间线（28 周，AAAI 16 周 + CVPR 12 周）
- 第四部分：诊断结论 + 已确认决策（D1-D5）
- 第五部分：与 v1 plan 的差异说明

### C. 执行文件（4 份，W1 已交付）

| 文件 | 行数 | 角色 | 何时使用 |
|------|------|------|----------|
| `progress.txt` | 182 | 进度追踪文件，按 W1-W28 列出所有 deliverable + Risk Gate | 每周打开看一次，勾选完成项 |
| `setup_env.sh` | 158 | 一键环境配置（PyTorch + pytorch3d + chamferdist + dassl） | W1 跑一次 |
| `download_data.sh` | 215 | 数据 + 权重下载（ModelNet-C / ScanObjectNN-C / OpenShape） | W1 跑一次 |
| `generate_paraphrase.py` | 331 | DeepSeek API 生成文本 paraphrase | W3 跑一次 |

### D. 代码骨架（1 份，W1 已交付）

| 文件 | 行数 | 角色 |
|------|------|------|
| `Point-Cache/runners/model_with_mcp3d.py` | 728 | **MCP-3D 主代码骨架**，包含 11 个 Section |

**这个骨架文件的 11 个 Section**：
1. Imports（依赖导入）
2. 默认超参（kappa0, omega, alpha 等）
3. **vMF 文本锚点**（已实现完整闭式解）
4. **ICP-CD 几何距离**（接口齐全，body 是 TODO，W4-W5 实现）
5. **流式 z-score 统计**（已实现完整 EMA）
6. **MemoryCell 单格子**（已实现完整）
7. **MCP3DMemoryMatrix 2x3 矩阵**（已实现完整）
8. **per-cell logits 计算**（接口齐全，body 是 TODO，W6-W7 实现）
9. **多源融合**（已实现完整）
10. **主 TTA 循环**（占位符，W6-W7 填充）
11. **入口 main()**（已实现完整调度）

> **TODO 标注的部分** = 我留给后续实现的"埋坑"，是有意为之，因为这些是核心算法，需要在 W4-W7 阶段经过多次实验调参才能定型。**当前 W1 阶段不需要碰它们**。

---

## 3. 推荐阅读顺序（针对完全零基础）

### 路径 A：理解项目（不写代码，3-4 小时）

1. 第一步（30 分钟）：看 v3 可行性诊断
   - 文件：`MCP3D_feasibility_and_proposal.md`
   - 目标：知道**这个研究在解决什么问题、为什么是 novelty、有什么风险**
2. 第二步（30 分钟）：看 v2 计划的"第一部分 26 个 gap"
   - 文件：`/root/.windsurf/plans/mcp3d-logic-audit-and-revised-plan-f3c00d.md`
   - 目标：知道**论文有哪些潜在漏洞、补什么实验**
3. 第三步（30 分钟）：看 v2 计划的"第三部分 28 周时间线"
   - 同上文件第三部分
   - 目标：知道**未来 6 个月每周做什么**
4. 第四步（30 分钟）：看 progress.txt
   - 文件：`progress.txt`
   - 目标：建立"每周勾选清单"的工作模式
5. 第五步（1 小时）：看 model_with_mcp3d.py 的 Section 1-2 + 11 个 Section 的注释（不看实现）
   - 文件：`Point-Cache/runners/model_with_mcp3d.py`
   - 目标：知道**代码骨架长什么样，每个模块对应论文的哪一部分**

### 路径 B：开始动手（路径 A 之后）

6. 第六步（30 分钟）：跑 setup_env.sh，看输出报错是否需要修
7. 第七步（1-2 小时）：跑 download_data.sh，等下载完成
8. 第八步（1 小时）：跑 Point-Cache 官方 baseline（W2 任务，确认环境跑得动）
9. 之后进入 W3：vMF + paraphrase

---

## 4. 你最容易迷茫的 3 个问题

### Q1：为什么 v1/v2/v3 提案文件都保留？

因为它们记录了"思路是怎么演化过来的"——v1 是详细但有错误的版本，v2 修了数学错误，v3 砍掉了不创新的部分并做了风险评估。**v3 是最新的方法学依据**，但 v1/v2 偶尔会被引用为"曾经考虑过"。

### Q2：v1 计划和 v2 计划的关系？

- **v1 计划** (`mcp3d-publication-plan-f3c00d.md`)：26 周时间线，乐观版
- **v2 计划** (`mcp3d-logic-audit-and-revised-plan-f3c00d.md`)：28 周时间线，**审计 v1 后发现的漏洞 + 修订版**

**当前执行依据是 v2 计划**。v1 留作历史档案。

### Q3：model_with_mcp3d.py 里的 TODO 是什么意思？

`raise NotImplementedError("...: implement W4-W5 per plan §3.4")` 的意思是：
- **当前 W1 阶段不需要管它**（W1 只是搭骨架）
- **W4-W5 阶段我们一起来填这部分**（届时按 v2 计划 §3.4 实现）
- 这种"留 TODO + 标明何时实现"的做法是大型项目的常规做法，避免一次性把所有东西做完

---

## 5. 当前进度状态

```
✅ 思路演化（v1 → v2 → v3）           [4 份 .md 文档，已完成]
✅ 投稿计划（v1 → v2 + 审计）          [2 份 .md 文档，已完成]
✅ 5 个决策点（D1-D5）已锁定            [写入 progress.txt 决策日志]
✅ W1 启动任务（5 项交付物）            [progress + setup + download + paraphrase + skeleton]
⏳ W1 任务剩余：实际跑 setup_env.sh    [需要你决定何时执行]
⏳ W2 Point-Cache 复现                 [W1 完成后启动]
⏳ W2.5 探针实验（关键 gate）          [W2 之后]
⏳ W3-W16 后续 13 个阶段                [按 v2 计划推进]
```

---

## 6. 我能帮你做什么（请你点菜）

我接下来可以：

| 选项 | 内容 | 时间 |
|------|------|------|
| **A. 概念讲解** | 把 v3 提案里"vMF 锚点 / ICP-CD / 紧致性诊断"用更通俗的方式讲一遍 | 15-30 分钟对话 |
| **B. 文件走读** | 跟你一起逐行读 model_with_mcp3d.py，每一段解释为什么这么写 | 30-60 分钟对话 |
| **C. 代码示意图** | 画一张架构图（draw.io 或 markdown 流程图）让你直观理解数据流 | 单次产出 |
| **D. 快速上手指南** | 写一份"5 分钟 / 30 分钟 / 1 小时"三档的快速入门 README | 单次产出 |
| **E. 直接动手** | 现在就跑 setup_env.sh + download_data.sh，遇到错误一起处理 | 实时 |

请告诉我你想从哪个开始。我建议从 **A 概念讲解**或者 **B 文件走读**起步，让你先建立认知，再动手不迟。
