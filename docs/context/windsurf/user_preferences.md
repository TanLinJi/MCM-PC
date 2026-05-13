# 用户偏好与项目规则

> 最近更新：2026-05-13

---

## 沟通风格偏好

- **语言**：中文
- **解释深度**：从零开始，不假设有 ML 背景
- **格式**：遇到缩写必须先定义，用类比/比喻代替纯数学，代码块格式展示公式
- **节奏**：每次专注一个概念，不同时推进多个话题
- **反馈方式**：用户会直接说"我不明白 XX 是什么"，收到后立刻暂停当前任务，先解释概念

## 用户曾明确要求的规则

1. **不要只帮我运行，我要知道在做什么**
   > 原话："刚刚你总结的哪些文档在哪里，现在我发现都是你在帮我运行，我依然什么都不知道"
   → 每次做任何操作前，先简要说明"我准备做什么，为什么"

2. **先讲概念，再做实现**
   > 原话："先从第一部分概念讲解吧"
   → 用户确认理解当前概念后，再推进下一步实现

3. **上下文存到服务器**
   > 原话（2026-05-09）：把对话存到服务器 `/root/autodl-tmp/MCM-PC/docs/context/windsurf/`
   → 每次会话结束时，更新 `chat_summary.md` 和 `next_steps.md`

4. **LaTeX 公式改用代码块**
   → 不使用 `$...$` 或 `$$...$$`，改用 \```代码块\``` 展示数学内容

5. **经常保存进度到 `/root/autodl-tmp/MCM-PC/docs/context/windsurf/`**
   > 原话（2026-05-10）："以后我们的进度和对话，已经我的疑问，你要经常保存，以便你能经常查阅"
   → 触发时机：会话收尾、关键决策、跑通关键实验、连续概念讲解之后
   → 文件：chat_summary.md / next_steps.md / decisions.md / user_preferences.md / session_YYYY-MM-DD.md
   → 不要每条消息都保存（会打断对话），但在阶段性转折点必须 flush

6. **实验命令由 Cascade 给，用户自己跑**
   > 原话（2026-05-09）："你不要帮我跑，你需要给我讲命令是什么，我会问你许多问题"
   → 除了极短的一次性验证/smoke test，不要自己 run 长实验
   → 给完整可复制命令，说清每个参数的意义，等用户问

7. **commit message 简短，一句话**
   > 原话（2026-05-10）："以后 commit 的注释请简短一点，一句话就可以了"
   → 不再写多段 body（之前 W2 commit 的多段格式作废）
   → 格式：`type(scope): one short imperative sentence`
   → 例：`feat(repro): add bar3 hierarchical TTA driver`
   → 不写 body，不写 paper-anchor 数字（数字只进 docs/project/progress.txt）

8. **输出格式偏好 HTML > Markdown（2026-05-10 晚）**
   > 原话："以后都尽量使用 HTML 的方式"（参考 Thariq @trq212 "Using Claude Code: The Unreasonable Effectiveness of HTML"）
   → **分层规则**（按读者决定格式）：

   | 产出类型 | 格式 | 理由 |
   |---|---|---|
   | `docs/context/windsurf/*.md`（Cascade 自读恢复上下文） | **Markdown** | token 效率、diff 清晰 |
   | 短 checklist / 决策列表 / 漏洞表 | **Markdown** | 结构简单 |
   | **概念解释讲解（给用户读）** | **HTML** ⭐ | SVG 图 + 公式 + 交互 |
   | **进度 / review 报告**（如今晚 summary） | **HTML** ⭐ | 多 tab / gantt / 可视化 |
   | **W*.* 实验结果报告** | **HTML** ⭐ | table + plot + 可筛选 |
   | **代码解释 / PR review** | **HTML** ⭐ | 内联标注 + diff |
   | 论文正稿 `docs/paper/*.md` → LaTeX | Markdown → LaTeX | 投稿要求 |
   | paper outline / 章节映射 | Markdown | 结构简单 |

   → HTML 产出放在 `/root/autodl-tmp/MCM-PC/docs/reports/` 下，命名 `YYYY-MM-DD_主题.html`
   → HTML 必须 self-contained（内联 CSS + JS），不用外部 CDN
   → HTML 默认采用暗黑主题，必须 responsive；复杂内容需要 tab navigation 或清晰的分区导航
   → 不能改的格式：git commit message / `docs/project/progress.txt` / 项目根 `README.md`

   **HTML 输出的触发关键词**（用户说以下词时，默认输出 HTML）：
   - "做个 report" / "总结一下" / "给我一份 review"
   - "展示 / 可视化" / "画一张图"
   - "解释 XX 概念"（若超过 200 行内容）
   - "写个 prototype / editor / explorer"

9. **新思路一旦验证就立刻进论文草稿（含图表）**
   > 原话（2026-05-11）："以后我们提到的新的思路，如果是经过理论验证或者实验验证，你需要分析，合理的内容，要及时写入论文草稿，包括图表。"
   → **触发条件**（任一即触发）：
     - 理论验证：推导成立、数学论证 self-consistent、与已有 contribution 不冲突
     - 实验验证：探针 P# / oracle 单测 / leave-one-out / 复现实验跑出关键数字
     - 用户口头确认"这个思路成立"或"采用这个写法"
   → **Cascade 必须做的分析**（落论文前自检）：
     - (a) 这个 idea 是否与现有 D# 决策、G# 漏洞、F# 发现一致？
     - (b) 是否在 D16 / D17 / G1-G2 等写作 guardrail 范围内？
     - (c) 该写到哪个章节、哪个段落、需不需要新建图/表？
     - (d) 现有证据强度：preliminary observation / hypothesis / claim / contribution？
   → **论文落地的最小动作**（写完即视为完成）：
     - 在 `MCM-PC/docs/paper/0X_*.md` 对应章节加段或新文件
     - 若需图表：先写 `[Tab. N: 描述]` / `[Fig. N: 描述]` 占位 + 图表来源说明，等数据齐时回填
     - 在该章节顶部 ⚠️ 区写入证据强度（v0.X / preliminary / claim）
     - 在 `docs/context/windsurf/decisions.md` / `key_findings.md` / `doc_gaps.md` 任一对应文件追加引用
     - `00_outline.md` 状态栏更新
   → **不该立刻写论文的情况**：
     - 仅是用户的提问、思考片段、未确认的猜想
     - 与现有决策冲突且未解决冲突
     - 引用了尚未跑通的实验数字
   → **如果只是 HTML 讲义 / 概念笔记里出现**，那不算"已写入论文"——必须真正落到 `docs/paper/`

10. **实验脚本默认双卡并行（2026-05-11 加入）**
   > 原话（2026-05-11）："以后都尽量使用双卡，这样可以节省时间"
   → 所有新建的 eval / ablation / smoke 驱动脚本，默认以**双卡并行**为基线设计：
     - 2 个 job：GPU 0 ∥ GPU 1 同时跑
     - N 个 job（N>2）：按 batch 循环，每 batch 填满两卡
     - 1 个 job（无法并行）：只用 GPU 0，写清楚原因
   → **例外**（允许单卡）：
     - 单 job 本身需要 >14GB 显存（即两卡显存都不够合并使用）
     - 实验明确要对齐某篇论文的单卡 baseline
     - 脚本仅 smoke / 调试性质且已知 <30s
   → **实现模板**：
     ```bash
     run_one 0 "<cor_A>" "<extra_A>" "<tag_A>" &
     PID0=$!
     run_one 1 "<cor_B>" "<extra_B>" "<tag_B>" &
     PID1=$!
     wait $PID0 $PID1
     ```
   → **N>2 的 batch 模板**：参见 `Point-Cache/scripts/repro_fig1a_bar3_tta.sh` 的 `for ((i=0; i<N; i+=2))` 结构。
   → **检查点**：每写完一个驱动脚本，自检是否用到了两卡；若没用到，注明理由。
   → **与 SOP 的关系**：会同步写进 MILESTONE_SOP.md 附录 C（算力预算）。

11. **定期回顾并保存对话记录（2026-05-12 加入）**
   > 原话："你需要时不时的回顾我们的对话，并把对话记录保存下来，以便以后超出了你的最大上下文查询，你可以去从保存的对话记录中去查看"
   → 工作中遇到阶段切换、重要解释、方法 pivot、实验结论、用户新偏好时，更新 `docs/context/windsurf/`。
   → 优先更新 `chat_summary.md`、`next_steps.md`、`decisions.md`、`key_findings.md`、`user_preferences.md`。
   → 长解释或关键问答可追加到 `session_YYYY-MM-DD.md`。
   → 不需要每条消息都保存，但每个阶段性结论必须落盘，便于上下文超限后恢复。

12. **术语解释要详细，英文缩写第一次出现必须定义（2026-05-12 加入）**
   > 原话："不要用尽量少的术语逐条解释，而是用尽量详细的术语逐条解释"
   → 回答技术问题时保留必要术语，但第一次出现要解释中文含义、英文全称、它在本项目里的作用。
   → 对用户没学过的机器学习概念，不只给直译，还要说明它为什么影响当前论文路线。
   → 后续如果使用 `D20`、`P1`、`TTA`、`anchor`、`feature`、`regime`、`pivot` 等项目内术语，先展开含义再继续推理。

13. **用户偏好变化要及时写入本文件（2026-05-12 加入）**
   > 原话："如果以后你发现是用户偏好的内容，你需要及时写进用户偏好中去，你要记住这些，如果用户偏好发生了变化，你还是需要及时在里边更新"
   → 只要用户表达长期沟通方式、文档格式、工作节奏、命令执行方式、保存方式、输出结构等偏好，就更新本文件。
   → 若新偏好覆盖旧偏好，要在旧条目附近写清"已更新/已替代"，不要让两个互相冲突的规则并存。
   → 更新后在回复中明确告诉用户已写入哪个文件。

14. **每次回答末尾给出下一步选项（2026-05-12 加入）**
   > 原话："每次你的回答后边，应该列出几个选项，询问我下一步继续做什么工作"
   → 除非用户明确要求只给最终结果，否则回答末尾给出 2-4 个可选下一步。
   → 选项要具体，例如"继续讲概念"、"检查 HTML 报告"、"进入 runner 设计"、"先 commit 当前文档"。
   → 不用泛泛地说"如果你想..."，而是直接列出可执行选项请用户选择。

15. **用户问到的每个问题都必须逐条详细回答（2026-05-13 加入）**
   > 原话："我问的每个问题你都必须详细的解释"
   → 用户列出编号问题时，必须按编号逐条回答，不能漏答、合并到含糊总结里、或只回答其中一部分。
   → 如果某个问题需要先查文件或实验结果，先查证再回答，并明确给出证据路径、数据范围和结论强度。
   → 如果之前漏答，下一轮必须先补漏，再继续新任务。

16. **HTML 报告默认暗黑主题且要有更好的视觉质量（2026-05-13 加入）**
   > 原话："你做的HTML页面不好看，而且请采用暗黑色主题"
   → 给用户阅读的 HTML 默认使用暗黑主题，而不是浅色主题。
   → 要有清晰视觉层级：深色背景、内容面板、表格对比、强调色、间距和移动端适配。
   → 不要只做能打开的 HTML；需要像正式项目报告或论文辅助图一样可读、清楚、专业。

---

## 项目目标（用户视角）

- 理解 MCP-3D 的完整原理
- 实现 MCP-3D 并在 Point-Cloud 数据集上跑通实验
- 最终目标：投稿 AAAI 或 CVPR

## 学习路径偏好

```
理解顺序（已确认）：
  概念文档（docs/concepts/） → 代码骨架（model_with_mcp3d.py） → 实验运行
  
不要跳步：不能在用户还没读完概念文档时，就开始讲代码实现细节
```

## 恢复上下文的指令模板

下次新会话时，对 Cascade 说：

```
请先读取以下文件恢复项目背景：
- /root/autodl-tmp/MCM-PC/docs/context/windsurf/chat_summary.md
- /root/autodl-tmp/MCM-PC/docs/context/windsurf/next_steps.md
- /root/autodl-tmp/MCM-PC/docs/context/windsurf/decisions.md
- /root/autodl-tmp/MCM-PC/docs/context/windsurf/user_preferences.md
然后告诉我当前项目状态和下一步应该做什么。
```
