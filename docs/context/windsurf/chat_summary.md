# 对话阶段摘要（MCP-3D 项目）

> 最近更新：2026-05-10 14:50
> 项目根目录：`/root/autodl-tmp/MCM-PC/`  （旧名 MCM-PC 已改）

---

## 阶段 1：审计与规划（第 1 次会话）

- 阅读 Point-Cache 代码结构（`utils/`, `runners/`, `llm/`）
- 发现 Point-Cache 三个"软肋"：
  1. 文本锚点用算术平均，忽略球面几何
  2. 缓存检索用特征相似度，不含几何信息
  3. 记忆格子是单层固定结构，无动态门控
- 撰写审计计划文档：`/root/.windsurf/plans/mcp3d-logic-audit-and-revised-plan-f3c00d.md`
- 确定 MCP-3D 三个 contribution：C1=ICP-CD 距离、C2=vMF 锚点、C3=2×3 记忆矩阵

## 阶段 2：W1 实施（第 1 次会话续）

创建了以下文件（W1 任务全部完成）：

| 文件 | 作用 |
|------|------|
| `docs/project/progress.txt` | W1-W28 任务 + Risk Gate 追踪 |
| `setup_env.sh` | Conda 环境 + 依赖安装脚本 |
| `download_data.sh` | 数据集 + 权重下载脚本 |
| `generate_paraphrase.py` | DeepSeek API 生成 paraphrase（1427类×40句）|
| `Point-Cache/runners/model_with_mcp3d.py` | MCP-3D 代码骨架（11 Section，TODO 占位）|
| `NAVIGATION.md` | 项目导航文档 |
| `docs/concepts/` | 概念讲解目录（5 个 md 文件）|

## 阶段 3：用户学习概念（第 2 次会话）

**起因**：用户发现"只有文件，但不知道这些文件在讲什么"，要求先理解概念。

**用户具体提问**：
- 软肋1 是什么意思？
- 球面是什么？
- vMF 是什么？
- MAP 是什么？
- 这些缩写是论文里的吗？

**处理**：完整重写 `docs/concepts/01_vmf_anchor.md`，新版从零开始解释：
- 第0节：全缩写表（来源说明：均为 ML 领域通用词，非本论文创造）
- 第1节：什么是文本锚点（从 CLIP = "翻译机" 开始）
- 第2节：软肋1 = 算术平均 + 球面问题
- 第3节：球面 = 归一化后向量端点落在超球面上
- 第4节：vMF = 球面上的高斯分布（mu=中心方向，kappa=集中度）
- 第5节：MAP = 先验 + 观测数据加权（医生诊断类比）
- 第6节：四个概念串联成完整答案

## 阶段 4：W1 收尾 + Smoke Test（2026-05-09 晚）

**目标转变**：从"学概念"切换到"跑通代码验证环境"。

- 验证 `setup_env.sh` 安装的 `mcmpc` conda env
- 补齐遗漏的 Point-Cache 依赖 `torch_redstone==0.0.6`（OpenShape PPTA 需要）
- 修复 `Point-Cache/runners/zs_infer.py:40` 的 `wandb.log` 未被 `args.wandb` 守护的 bug
- 跑通 zero-shot smoke test：
  ```
  dataset = modelnet_c, corruption = add_global_2, npoints = 1024
  backbone = OpenShape PointBERT-ViT-g/14 (模型总参 2.57B)
  final accuracy = 71.47%
  ```
- 确定 wandb 使用策略（方案 B）：`WANDB_MODE=offline` 为项目默认
  - 无需注册 wandb 账号、无需联网
  - 日志保存到 `Point-Cache/wandb/offline-run-<ts>-<id>/`
  - 想上云时 `wandb sync <run-dir>` 批量上传

## 阶段 5：Point-Cache 代码 ↔ 论文映射讲解（2026-05-09 夜）

**起因**：用户说"跑之前先理解 Point-Cache 结构、和论文对应起来"。

**处理**：给出 4 层地图式讲解：
1. 论文一句话定位：两层（global+local）× 两类（pos+neg）= 4 缓存
2. 目录 ↔ 论文模块对应表（runners/ ↔ TTA 主循环、models/openshape/ppta.py ↔ Hierarchical patch 聚类、等）
3. 数据流图（`pc → 3D encoder → pc_feats + patch_centers → cache 更新 → logits 三路融合`）
4. 关键公式 ↔ 代码行号：
   - `logits_zs = 100 * pc_feats @ clip_weights`（CLIP logit_scale）
   - `cache_logits = α * exp(-β(1-A)) @ one_hot`（Tip-Adapter 核）
   - `patch_centers.mean(0, keepdim=True) @ local_cache_keys`（局部 cache query 用 5 patch 的均值）
   - Negative cache 的 value 是 prob_map 经阈值掩码，且是 `final -= ...` 不是 +=

**用户补问的细节**：
- wandb 是什么 / 为什么登录 / offline vs online vs disabled（已讲透）
- wandb 目录里的 3 个 symlink（`latest-run`、`debug.log`、`debug-internal.log`）指向哪里——都是指向当前最新 run 的相对路径 symlink，wandb 自动维护

## 阶段 6：Figure 1(a) bar 1 + bar 2 复现完成（2026-05-10 中午）

**目标**：复现论文 Figure 1(a) OpenShape 列的前两根柱子。

**协议确认**：ModelNet-C benchmark = 7 corruption × 5 severity = **35 setting 算术平均**。

**bar 1**（clean ModelNet40 zero-shot）：
- 单跑命令，~5 min
- 结果 **83.27%** vs 论文 84.56%，差 -1.29pp（容忍内）

**bar 2**（ModelNet-C 35-mean zero-shot）：
- 双 T4 并行，~90 min（实际跑了 ~2 h，因 scale_1.h5 损坏多了一次单跑）
- 结果 **~72.51%** vs 论文 73.49%，差 -0.98pp（容忍内）
- 中途遇到 `scale_1.h5` 是 0 字节空文件，从 hf-mirror 重下 30 MB → 单独跑 scale_1 → 数字落到 78.81 ✓

**关键发现**：
- jitter 退化最剧烈（79.29 → 32.66），是 OpenShape 软肋 → 印证 MCP-3D 的 C1（ICP-CD）方向
- rotate 最稳健（84.12 → 72.33）：OpenShape 自带 rotation-awareness
- 整体退化曲线随 severity 单调递减 ✓ 数据 pipeline 正确

**新落盘的两个脚本**（未 commit）：
- `Point-Cache/scripts/repro_fig1a_bar2_zs_corruption.sh`（双 T4 批处理 35 setting）
- `Point-Cache/scripts/repro_fig1a_summarize.py`（log-dir → per-run + mean，支持嵌套子目录）

**锁定决策**：
- D9（新）：接受 ~1pp 偏差不深 debug，列入"已知噪声"
- D8（之前）：继续按"先 hierarchical TTA 再补别的"顺序往下

## 阶段 7：bar 3 复现完成 + W2 closure（2026-05-10 晚）

**bar 3 数字**：35-mean = **75.27**（论文 76.59，Δ -1.32pp ✓）

**TTA 增益**（bar 3 − bar 2）= +2.76pp（论文 +3.10pp，方向 + 量级一致）

**关键洞察**（已永久存档到 `docs/context/windsurf/key_findings.md`）：

| finding | 数字 | MCP-3D 启示 |
|---|---|---|
| F1 ⭐ scale 上 TTA **退化** -0.40pp | 5 sev 中 4 sev 为负 | C1 (ICP-CD) 主 attack point |
| F2 jitter 增益最大 +6.98pp | sev=4 单 cell +12.40pp | C1 stress test |
| F3 rotate 几乎无增益 +0.58pp | 已内置 rotation-aware | C1 narrative 改 "corruption-specificity" |
| F4 dropout sev=4 cliff | TTA 也救不回 | 失败案例素材 |
| F5 系统性 -1pp | 三柱同向 | D9 接受 |

**新锁决策**：
- D10：scale = MCP-3D 主 attack point（基于 F1）
- D11：W2.5 三探针执行顺序 P3 → P1 → P2（P4 延后到 W4）

**已 commit + tag**：
- `1a4badf` feat(repro): bar3 driver + Fig1a summary
- `e966ca2` docs(progress): close W2 milestone with bar3 = 75.27
- tag `w2-tta-baseline`（已 push 远程）

## 阶段 8：论文复习启动 + SOP 锁定（2026-05-10 晚）

**用户启动"论文整体复习"模式**，按 "故事 → 鸟瞰 → 概念" 三段流程，**暂停跑实验**先把整体思路过一遍。

**已讲完**：
- 故事 arc（动机 → 三贡献 C1/C2/C3 → 期望结果），含 5 个追问钩子 G1-G5
- 鸟瞰（16 周 PHASE A 时间线 + 4 个 RISK_GATE + PHASE B 概览）

**用户追问 5 题，全部答完 + 落盘**：
- Q1: Related Work 草稿 → `docs/paper/02_related_work.md` v0.1 + 新加 P5 实验（修补 G8）
- Q2: 写作-实验并行 → D12 锁定 + 建 `docs/paper/` 目录
- Q3: 低置信度记忆 → 不是负样本，是 boundary memory（G6 漏洞）
- Q4: vMF 不在点云侧 → 三理由（G7 漏洞）
- Q5: +1~+3pp 拆解 → 三档目标（基础/目标/卓越）

**用户主动驱动的新流程**：
- D14: 锁定**里程碑完成 SOP**，固化到 `MCM-PC/docs/project/MILESTONE_SOP.md` (v1.0)
  - 7 步 checklist + 5 个附录（论文格式 / commit 约定 / 长期健康 / 投稿自检 / SOP 演化）
  - 强制：每次 W*.* 完成后必走，Cascade 新会话必先读

**新挖漏洞**：G6 / G7 / G8（共 8 条 G1-G8）。详见 `doc_gaps.md`。

**新锁决策**：D10 / D11 / D12 / D13 / D14 五条。详见 `decisions.md`。

**新建文件**：
- `MCM-PC/docs/paper/00_outline.md`：论文章节大纲 + 触发-写作映射表
- `MCM-PC/docs/paper/02_related_work.md`：§2 v0.1 草稿
- `MCM-PC/docs/project/MILESTONE_SOP.md`：里程碑完成 7 步 SOP（v1.0）⭐
- `docs/context/windsurf/key_findings.md`：F1-F5
- `docs/context/windsurf/doc_gaps.md`：G1-G8

## 当前位置（2026-05-10 21:45）

**项目阶段**：W1 ✅ → W2 ✅ → **W2.5 ⏸️ 暂停**（用户复习中，复习完再启动 P3-P5）

**Figure 1(a) OpenShape 列**：
- bar 1 ✅ 83.27 / bar 2 ✅ 72.51 / bar 3 ✅ 75.27

**工作树**：未 commit 的 paper drafts + SOP（建议本次会话末批量 commit）

**下一步**（按用户复习节奏）：
1. 用户消化 §2 Related Work v0.1 草稿 + SOP v1.0 + 阶段 8 答的 5 题
2. 进入第三阶段"概念"：从 `concepts/00_overview.md` 开始
3. 概念复习完成后启动 W2.5 P3 → P5 → P1 → P2

**核心引用**：
- 进度全景：本文件（chat_summary.md）
- 阶段流程：`MCM-PC/docs/project/MILESTONE_SOP.md` ⭐
- 实证证据：`key_findings.md` (F1-F5)
- 待修漏洞：`doc_gaps.md` (G1-G8)
- 锁定决策：`decisions.md` (D1-D14)
- 详细单日：`session_2026-05-10.md`
