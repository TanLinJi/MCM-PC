# Windsurf 对话归档索引

> 项目：MCP-3D（Point-Cache 基线 → MCP-3D 方法）
> 项目代码：`/root/autodl-tmp/MCM-PC/`
> 最近更新：2026-05-11 10:17

---

## 1 句话当前状态

W1 ✅ → W2 ✅（bar1+2+3 已 commit + tag `w2-tta-baseline`）→ **W2.5 ⏸️ 暂停**（用户处于"论文复习"状态，已讲完故事 + 鸟瞰，正在进入第三阶段：概念）

---

## 必读文件（按顺序，新会话 5-10 min 完整恢复）

| # | 文件 | 作用 |
|---|---|---|
| 1 | 本文 INDEX.md | 入口 + 1 句话状态 |
| 2 | [`user_preferences.md`](./user_preferences.md) | 用户偏好 + 沟通规则（缩写展开 / gap mining / 不主动讲概念） |
| 3 | **`MCM-PC/docs/project/MILESTONE_SOP.md`** ⭐ | **流程 SOP v1.0**（每次 W*.* 完成必走 7 步） |
| 4 | [`chat_summary.md`](./chat_summary.md) | 阶段 1-8 顺续 + "当前位置"段（最后一段最关键） |
| 5 | [`next_steps.md`](./next_steps.md) | 当前 W2.5 待办 + 写作并行 + 漏洞修补任务 |
| 6 | [`decisions.md`](./decisions.md) | D1-D14 已锁决策（不要破坏） |
| 7 | [`doc_gaps.md`](./doc_gaps.md) | G1-G8 待修漏洞 |
| 8 | [`key_findings.md`](./key_findings.md) | F1-F5 实证证据 |
| 9 | 最新 session 详细笔记（见下表） | 单日完整对话脉络 |

---

## 单次会话详细笔记（append-only）

| 文件 | 时间 | 主题 |
|---|---|---|
| [`session_2026-05-10.md`](./session_2026-05-10.md) | 05-09 晚 → 05-10 早 | W1 收尾 + smoke test 71.47% + wandb offline + Point-Cache 代码讲解 |
| [`session_2026-05-10_evening.md`](./session_2026-05-10_evening.md) ⭐ | 05-10 19:30-22:00 | **论文复习启动 + 故事/鸟瞰 + 5 题答案 + SOP v1.0 锁定** |
| [`session_2026-05-12.md`](./session_2026-05-12.md) ⭐ | 05-12 | **D22 anchor pollution pivot + conditional anchor switching 解释与架构图** |
| [`session_2026-05-13.md`](./session_2026-05-13.md) ⭐ | 05-13 | **逐条回答规则 + HTML 暗黑主题 + anchor pollution 证据位置纠正** |

（2026-05-09 之前会话摘要已并入 `chat_summary.md` 阶段 1-3）

---

## 项目代码侧关键文件（不在 docs/context/windsurf/，但必读）

| 路径 | 作用 |
|---|---|
| `MCM-PC/docs/project/MILESTONE_SOP.md` ⭐ | 7 步 SOP + 5 附录 |
| `MCM-PC/docs/paper/00_outline.md` | 论文章节大纲 + 触发-写作映射表 |
| `MCM-PC/docs/paper/01_introduction.md` | §1 v0.1 骨架草稿（diagnosis-driven framing 落地） |
| `MCM-PC/docs/paper/02_related_work.md` | §2 v0.1 草稿 |
| `MCM-PC/docs/concepts/00_overview.md` | 5 个核心概念入口（用户复习第三阶段起点） |
| `MCM-PC/docs/reports/2026-05-10_concepts_overview.html` | 概念阶段 00 overview 的 HTML 复习页 |
| `MCM-PC/docs/reports/2026-05-11_concepts_detailed.html` | 概念阶段完整 HTML 讲义（00-04：overview / vMF / ICP-CD / compactness / 2×3 matrix） |
| `MCM-PC/docs/proposals/MCP3D_full_proposal_v2.md` | 项目方案 v2（最新） |
| `MCM-PC/docs/experiments/fig1a_summary.md` | Figure 1(a) 复现数字 |
| `MCM-PC/docs/project/progress.txt` | 项目进度日志 |

---

## 其他文件（与 MCP-3D 无关）

| 文件 | 作用 |
|---|---|
| `apply_patch.py` | 别的项目遗留 |
| `切号器修复方案.md` | 别的项目遗留 |

---

## 压缩上下文卡片

| 文件 | 时间 | 作用 |
|---|---|---|
| [`context_compact_2026-05-10_2301.md`](./context_compact_2026-05-10_2301.md) | 2026-05-10 23:01 | 新会话快速恢复用的短版上下文：当前阶段、用户偏好、已修漏洞、下一步主线 |

---

## 新会话恢复上下文的标准指令模板

**复制给新会话的 Cascade**：

```
请按以下顺序读取文件恢复 MCP-3D 项目背景：

# 必读
1. /root/autodl-tmp/MCM-PC/docs/context/windsurf/INDEX.md
2. /root/autodl-tmp/MCM-PC/docs/context/windsurf/user_preferences.md
3. /root/autodl-tmp/MCM-PC/docs/project/MILESTONE_SOP.md   ← 流程 SOP，每次会话必读
4. /root/autodl-tmp/MCM-PC/docs/context/windsurf/chat_summary.md
5. /root/autodl-tmp/MCM-PC/docs/context/windsurf/next_steps.md
6. /root/autodl-tmp/MCM-PC/docs/context/windsurf/decisions.md
7. /root/autodl-tmp/MCM-PC/docs/context/windsurf/doc_gaps.md
8. /root/autodl-tmp/MCM-PC/docs/context/windsurf/key_findings.md

# 最新会话脉络
9. /root/autodl-tmp/MCM-PC/docs/context/windsurf/session_2026-05-10_evening.md   ← 最新

读完后简要告诉我：
- 当前项目阶段（W几）+ 是否处于复习暂停状态
- 上次会话结尾的未完成动作
- 我现在最应该做的一件事
- D1-D14 中是否有我应该立刻知道的硬性决策
```

---

## Cascade 更新此归档的触发时机

按 `user_preferences.md` 第 5 条 + SOP 步骤 5：

1. 会话即将结束
2. 用户做出重要决策（commit 粒度、方法选择、验证方案等）
3. 跑通关键实验（数字落盘的那一刻）
4. 连续概念讲解之后（如 wandb / Point-Cache 结构）
5. 修改项目代码/文档/环境（和 `docs/project/progress.txt` 同步）

**不要每条用户消息都保存**，会打断对话节奏。

---

## 文档长度维护规则（用户 2026-05-10 22:00 提出）

- 单 markdown 文件**软上限 300 行**，超过按主题拆分
- 拆分命名约定：`<原名>_part1.md` / `<原名>_part2.md`，或按时间段 `<原名>_phase1-5.md` / `<原名>_phase6+.md`
- 拆分后必须在本 INDEX.md 加引导（哪个 part 装什么）
- 当前各文件状态（行数）：
  - `session_2026-05-10.md` 316 行（**已超**，建议下次更新前拆为 morning + evening）
  - `decisions.md` 234 行
  - `next_steps.md` 198 行
  - `chat_summary.md` 192 行
  - `key_findings.md` 147 行
  - `MILESTONE_SOP.md` 213 行
  - 其他 < 100 行
