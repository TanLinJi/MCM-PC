# 技术决策记录（MCP-3D 项目）

> 最近更新：2026-05-10

---

## D1：MCP-3D 三个核心 Contribution（已锁定）

```
C1 = ICP-CD 距离（最核心，最有创新性）
     用点云几何对齐（ICP）+ 形状距离（Chamfer Distance）替代特征相似度来检索缓存

C2 = vMF 文本锚点（次核心，数学严谨化）
     用 vMF MAP 估计替代简单算术平均，得到更准确的类别文本锚点

C3 = 2×3 记忆矩阵（整合架构）
     6个记忆格子：置信度行×(全局/局部)，紧致度行×(全局/局部)，边界行×(全局/局部)
```

## D2：DeepSeek Paraphrase 生成参数（已锁定）

```
每类生成 40 句 paraphrase
支持 4 个数据集：ModelNet40, ScanObjectNN, ShapeNet55, Objaverse-LVIS
共约 1427 类
断点续传：已生成的类跳过
输出格式：JSON（{class_name: [paraphrase_list]}）
环境变量：DEEPSEEK_API_KEY
```

## D3：vMF 先验强度（待 W3 验证）

```
暂定 kappa_0 = 5.0
W3 阶段扫描：{1.0, 3.0, 5.0, 10.0}，取最优
```

## D4：文档风格（已锁定）

```
面向读者：ML 概念不熟悉的用户
要求：
  - 每个缩写首次出现必须解释
  - 用比喻/类比代替纯数学描述
  - 不使用 LaTeX 公式，改用代码块格式
  - 先讲"是什么"再讲"为什么"再讲"怎么用"
```

## D5：代码骨架策略（已锁定）

```
文件：Point-Cache/runners/model_with_mcp3d.py
策略：
  - 先写接口（函数签名 + docstring）
  - 核心算法用 TODO 占位
  - W2-W4 阶段逐步填充
目标：不破坏 Point-Cache 原有代码，以扩展方式叠加
```

## D6：Risk Gate 触发条件（已锁定）

```
R1（W2）：ICP 对一个 batch（8个点云）耗时 > 2秒 → 降为 CD-only
R2（W3）：vMF vs 简单平均精度差 < 0.1pp → vMF 降为附录细节
R3（W4）：紧致性与性能 Spearman 相关 < 0.5 → 去掉紧致性诊断
```

## D7：wandb 使用策略（已锁定，2026-05-10）

```
策略：离线优先
原因：autodl 网络受限；不想让用户为学术实验注册第三方账号

具体：
- 项目默认 export WANDB_MODE=offline
- setup_env.sh 末尾提示用户设此环境变量
- README 加 "Experiment Logging (Weights & Biases)" 章节说明完整工作流
- 所有正式 runner 命令都带 --wandb-log（让日志本地落盘）
- 上云选择：未来真要分享时再 wandb login + wandb sync <run-dir>

副作用：
- Point-Cache 的 runner（zs_infer / model_with_global_cache / model_with_hierarchical_caches）
  line 40/173/227 附近的 wandb.log 没被 args.wandb 守护
- 只修了 zs_infer.py，另外两个 runner 只要始终传 --wandb-log 就没问题
- 如果将来要改 runner 行为（比如不传 --wandb-log），需要补上 args.wandb 守护
```

## D8：W2 复现优先级（已锁定，2026-05-10）

```
先跑 hierarchical cache（论文主方法）7 corruption × L2，
不先补齐全部 zero-shot baseline 列。

理由：
- zero-shot 已有 add_global_2=71.47% 一个点，足够判断 encoder 与数据正常
- hierarchical cache 数字才是触发 R1 RISK_GATE（误差 >2pp 要 debug）的关键
- 如果误差在阈内，再补 zero-shot 6 个点填表；如果超阈，先 debug 主方法
```

**修订**（2026-05-10 14:00）：用户优先复现 Figure 1(a) 而非 Table 4，
所以 D8 实际执行变成"先跑 zero-shot 35 setting → 再跑 hierarchical 35 setting"
（图 1a bar 2 → bar 3）。Table 4 的 7×L2 协议合并到 bar 3 里（severity=2 那一列）。

## D9：接受 ~1pp 复现偏差（已锁定，2026-05-10）

```
观测到的偏差：
- bar 1 (clean ModelNet40):  你 83.27 vs 论文 84.56，delta = -1.29pp
- bar 2 (ModelNet-C 35-mean): 你 ~72.51 vs 论文 73.49，delta = -0.98pp

判定：在 ±2pp 容忍内，列入"已知噪声"，不深 debug。

噪声来源（不深查）：
1. cudnn benchmark 默认 True，浮点累加顺序影响 cosine 后第 2 位小数
2. 80 个 CLIP text templates 的循环顺序（看 templates.py 是否被改过）
3. fp16 GEMM 精度（CLIP-ViT-bigG-14 默认 fp16）
4. OpenShape ckpt 是 final 还是 EMA 版本（论文未明说）

升级触发：
- 如果 bar 3 (hierarchical TTA) 偏差仍 ~1pp 同向 → 系统性偏差，仍接受
- 如果 bar 3 偏差 >2pp 或反向（你的数 > 论文）→ 触发 R1，开始 debug
```

**触发器结果**（2026-05-10 19:10）：bar 3 = 75.27 vs 论文 76.59，偏差 -1.32pp 同向。
**判定**：系统性偏差确认，**继续接受不 debug**。RISK_GATE R1 PASSED，进 W2.5。

## D10：ModelNet-C scale corruption 是 MCP-3D 主 attack point（已锁定，2026-05-10；2026-05-11 修正写作边界）

```
基于 bar 3 复现的关键观察（详见 docs/context/windsurf/key_findings.md F1）：

  ModelNet-C scale corruption family 上 Point-Cache hierarchical TTA 退化 -0.40pp（5 sev 中 4 sev 为负）

意义：
- 这是 Point-Cache 在可控 benchmark stress test 下的失败案例，Point-Cache paper 里没单独讨论
- 不应直接写成现实世界分布偏移的自然类别；scale / rotate / jitter / dropout 都是人工 corruption families，是现实分布偏移的代理和诊断工具
- 是 MCP-3D 的 C1 (ICP-CD) 最有 promise 的攻击点
- 在该 benchmark protocol 下，scale 是全局几何 corruption；特征余弦可能失效，而 ICP+CD 在归一化点云上对 scale 更稳

对方法/论文的影响：
- W2.5 P2 探针：先在 scale_2/3/4 上做 ICP-CD 残差分布统计
- W4 主实现：C1 的优先 unit test 必须用 scale 数据
- 论文 §4 motivation：引用 -0.40pp 数字时必须写成"benchmark corruption stress test 下的证据"，不能直接等价为现实世界 scale drift
- 论文 §6 Table 4：scale 列上预期 MCP-3D 拉回正增益（target +1~+3pp），是最强卖点之一
```

## D11：W2.5 三探针执行顺序（已锁定，2026-05-10）

```
按"决策影响范围"排序，先做影响最大的：

P3（显存可行性测试）：0.5 天
  - 必须先做：如果 T4 16GB 装不下完整 pipeline，整个 PHASE A 都要重新规划 backbone
  - 不通过则切 PointBERT-vitb16 或 ULIP-2 base 或 feature 预编码

P1（旋转鲁棒性）：1 天
  - 决定 C1 narrative：保留"pose-shape 解耦"还是改为"corruption-specificity"
  - 已基于 F3 (rotate +0.58pp) 强烈倾向后者；P1 是定量收尾

P2（紧致度-精度相关性 r）：1-2 天
  - 决定 C2 (vMF 锚点) narrative：保留 2D 对比还是转向"3D 损坏特异性"
  - 影响 §3.3 写作 + W3 实现优先级

P4（ICP 残差分布）：延后到 W4
  - 不影响 W3 实现路径，等 ICP 代码写好顺手做即可
```

## D12：写作-实验并行原则（已锁定，2026-05-10）

```
原则：每完成一个实验里程碑，立即写对应论文段落，不延后。

理由：
- 避免实验 → 写作 时间差导致细节遗忘 (用户主动提出的核心驱动)
- 写作过程中暴露的逻辑漏洞可即时反馈给下一个实验
- 段落 commit 可作为里程碑物证

实施：
- 已建 docs/paper/ 目录
  - 00_outline.md：章节大纲 + 触发-写作映射表
  - 02_related_work.md：v0.1 草稿 (基于 F1-F5 + 现有 baseline 知识)
- 各章节触发条件见 outline 表
- 写作时间预算：每周 3-5 小时纯写作，与实验并行
- 每段写完 commit：docs(paper): write §X.Y first draft
- 每章写完打 tag：paper-§X-draft

例外：
- 在主实验跑完前，§4 Experiments 留空（数据未稳定）
- 在 W2.5 全部完成前，§1 Intro motivation 段留 placeholder
```

## D13：W2.5 加 P5 跨条件验证（已锁定，2026-05-10）

```
新增 P5 实验：在 OpenShape ModelNet-C scale 5 个 severity 上跑 global cache TTA，
对比 ZS / global / hierarchical 三组，验证 F1 (scale 退化) 是否仅限 hierarchical。

可选扩展：在 ULIP-2 backbone 上跑同样 scale 实验（10 run），验证 backbone 共性。

成本：方案 A (仅 global cache) ~ 1 小时；方案 B (含 ULIP-2) ~ 1 天。
判定：
  - 若 global 也退化 → F1 升级为"特征余弦类方法共性" → §2.1 末段强写
  - 若 global 不退化 → F1 仅限 hierarchical → C1 narrative 调整为"针对层级化缓存的退化模式"
  
执行顺序：可与 P3 (显存测试) 并行。P3 跑完即可立即跑 P5 (复用同样 backbone 加载状态)。
```

## D14：里程碑完成 SOP 锁定（已锁定，2026-05-10）

```
由用户主动提出，将"每次阶段完成应做的事"固化为流程文档。

落点：MCM-PC/docs/project/MILESTONE_SOP.md (v1.0)

核心 7 步（每次 W*.* 完成后约 60-90 min 全流程）：
  1. 实验数据归档     → 写 key_findings.md F#
  2. 论文段落写作     → docs/paper/0X_*.md，按顶会格式 (附录 A)
  3. 代码 commit+tag  → push 远程，模板见附录 B
  4. 决策+漏洞维护    → decisions.md D# / doc_gaps.md G# 增删评估
  5. 会话+计划归档    → chat_summary.md 加阶段 N + 重写 next_steps.md
  6. 审稿人攻击模拟   → 找 3-5 个 reviewer comment 关联 G#
  7. 下一阶段准备     → GPU / 数据 / API / 依赖

附录：
  A. 顶会论文段落格式 (Topic+Evidence+Bridge, 主动语态, 数字带单位)
  B. commit/tag 命名约定
  C. 长期项目健康 (周/月 review + 算力预算 500 GPU·h)
  D. 投稿前 1 周自检
  E. SOP 自我演化机制

强制：
- 每个 W*.* 实验完成后必须按 SOP 走完
- Cascade 在新会话被引用时必须先读 SOP 再继续工作
- 用户发现缺失/问题立即 update + 升版本号
```

## D15：HTML 作为长/复杂输出默认格式（已锁定，2026-05-10 晚）

```
由用户主动提出，参考 Thariq @trq212 "Using Claude Code: The Unreasonable
Effectiveness of HTML"（PDF: /root/autodl-tmp/使用说明.pdf）。

核心观点：
- Markdown 对简单短文档好；但 100+ 行 / 需要可视化 / 需要交互 → HTML 更高效
- HTML 信息密度 (表格/SVG/CSS/JS) + 分享便利 + 双向交互 > Markdown
- Markdown 的唯一优势：version control diff 清晰

分层规则（Cascade 产出格式）：
  HTML  → 概念解释、进度报告、W*.* 结果报告、代码解释、design prototype
  Markdown → Cascade 自读文件 (docs/context/windsurf/)、SOP 流程、短列表、commit message

产出位置：/root/autodl-tmp/MCM-PC/docs/reports/YYYY-MM-DD_主题.html
规格：
- self-contained single file (内联 CSS + JS，无 CDN 依赖)
- dark mode + responsive + tab navigation (若内容复杂)
- SVG 为矢量 (gantt / flowchart / spatial layouts)
- 保留 details/summary accordion 处理长 Q&A

首个 demo: /root/autodl-tmp/MCM-PC/docs/reports/2026-05-10_review_session.html

规则落点：user_preferences.md 第 8 条 + MILESTONE_SOP.md 附录 A 头部
```

## D16：corruption benchmark 只能作为 distribution shift 的代理（已锁定，2026-05-11）

```
用户指出并锁定的论文写作边界：

- 真实研究问题：3D 点云模型在现实部署中的 distribution shift。
- 评测工具：ModelNet-C / ScanObjectNN-C 的 jitter、dropout、rotate、scale 等是人工设计的 corruption families。
- 写作原则：不能把这些 corruption 直接写成现实世界分布漂移本身；只能写成可控 proxy / stress test / diagnostic benchmark。
- F1 scale 负增益的正确表述：Point-Cache 在 ModelNet-C scale corruption family 下出现 -0.40pp benchmark failure，不等价于现实世界所有尺度漂移都会失败。
- 论文中推荐句式：
  "Real-world distribution shifts motivate robustness evaluation; controlled corruptions such as scale, rotation, jitter, and dropout serve as diagnostic stress tests."

影响：
- §1 motivation：先讲 distribution shift，再讲 benchmark corruptions 是代理。
- §2 related work：所有 scale 负增益表述都必须带 benchmark / ModelNet-C 限定。
- §3.4 ICP-CD：C1 motivation 是修补 benchmark stress-test 暴露的 feature-space weakness，而不是宣称解决全部现实漂移。
- §4 experiments：scale 列是诊断切片，不是现实世界类别。
```

## D22：P1 anchor pollution pivot（已锁定，2026-05-12）

```
触发：
P1 feature drift probe 与 anchor pollution simulation 跑完后，发现 D19 的 raw ICP-CD
路线基于错误的 H1 假设。

术语：
- anchor pollution：测试流 cache 中的 anchor 被错误预测污染，后续样本参考这些 anchor 时会放大错误。
- stable anchor source：不会被低置信测试流连续覆盖的稳定锚点来源。strict source-free TTA 主方法应来自 text / vMF anchors 或高置信测试时证据；clean test anchors 和 labeled source prototypes 只能作为诊断或上界。
- conditional anchor switching：根据当前样本可靠性，在 stable anchor source、stream anchor、abstention 之间切换。

关键证据：
- scale_2: cos(f_clean, f_scale)=0.9306，class-consistent=95.5%，说明 feature 没有严重失效。
- scale_2 anchor simulation: corrupt anchor 84.44%，clean anchor 95.46%，Δ=+11.02pp。
- jitter_3 anchor simulation: corrupt anchor 81.93%，clean anchor 30.75%，Δ=-51.18pp。

决策：
- 终止 raw ICP-CD 作为主预测修复路径；D19 保留为失败复盘和审稿防线材料。
- C1 从 "geometry-as-feature-backup" 改写为 "corruption-aware anchor source selection"。
- 明确协议边界：clean anchor 是 oracle；labeled source prototype 是 source-available ablation；strict source-free TTA 主线不能依赖干净测试样本或带标签源训练样本。
- 下一步实现最小版 conditional anchor runner，先验证 scale_2 不低于 hierarchical baseline，jitter_3 不明显倒退。

落点：
- 复盘：docs/decisions/D22_p1_anchor_pollution_pivot.md
- 论文：docs/paper/01_introduction.md §1.2.2 + §1.3
- 计划：docs/context/windsurf/next_steps.md
- 架构图：docs/reports/2026-05-12_conditional_anchor_switching_flow.html
```

## D21：D19 几何项超参数收敛（已作废，2026-05-12）

```
触发：
用户在 STAGE=smoke C 计划运行期间提问 "gating 阈值是不是超参，我们是不是引入了
许多超参"。

状态更新（2026-05-12）：
D22 已终止 raw ICP-CD 主路径，因此本条不再作为近期任务推进。若未来在 appendix
保留 ICP-CD 诊断实验，可复用本条作为 sensitivity analysis 的备忘。

现状盘点 (D19 v0.1.2 / C plan)：
  D19 引入的 flags 共 7 个，其中 4 个是核心可调超参：
    - --geom_alpha           default 6.0   核心
    - --geom_beta            default 5.0   核心
    - --geom_zero_mean       default True  核心（设计选择）
    - --geom_entropy_threshold default 0.5 核心
    - --geom_estimate_scale  default True  按 corruption 类型固定
    - --geom_max_iter        default 20    性能参数，固定
    - --enable_geom_cache    bool          ablation flag，不算超参

  Point-Cache 原方法本身已有约 10 个超参（α=4, β=3, shot_capacity=3 等）。
  → 累加后 14 个 hyperparams，TTA 文献里属于"中等偏多"。

风险（审稿层面）：
  - "Why 4 new hyperparameters for 0-2pp gain?"
  - "Why not reuse existing α/β?"
  - "Provide sensitivity analysis."

未来处理方向（暂未锁定，等 C 计划结果出来后选）：
  方案 A (推荐)：
    - geom_alpha default 改为 = Point-Cache 的 alpha (4.0)
    - geom_beta  default 改为 = Point-Cache 的 beta  (3.0)
    - geom_entropy_threshold default 改为 = neg_cache.entropy_upper (0.5)
    - geom_zero_mean 固定 True，写论文时讲"理论动机"
    → 对外只暴露 1 个新超参（甚至 0 个）

  方案 B：保留独立 flag 作调研工具，论文里写 sensitivity table 证明鲁棒性

  方案 C：组合 grid (α_g × τ_g)，3×3 = 9 个点扫一次，~18 min 双卡

触发时机：等 STAGE=smoke (C plan) 结果出来：
  - 若 acc ≥ baseline：处理超参收敛 + 写 sensitivity（方案 A + B）
  - 若 acc < baseline：先 pivot 到 Fix D oracle 验证 ICP-CD 信号本身，
    超参收敛延后

落点：
  - 本条 D21（OPEN 状态，待 C plan smoke 结果决定具体方案）
  - 写论文时在 §3.4 / §4.X 的 hyperparameter design rationale 段落
```

## D20：所有驱动脚本默认双卡并行（已锁定，2026-05-11）

```
用户原话（2026-05-11 11:21）：
"现在是单卡吗？可以改成双卡吗？以后都尽量使用双卡，这样可以节省时间"

规则：
- 新建 eval / ablation / smoke 脚本 → 默认双卡并行
- 2 job → GPU 0 ∥ GPU 1；N > 2 job → for ((i=0; i<N; i+=2)) batch
- 1 job / 无法并行 → 默认 GPU 0 + 写清理由
- 允许单卡的例外：显存 >14GB / 对齐他人 single-card baseline / 调试 <30s

执行：
- eval_p4_scale_icpcd.sh smoke 阶段已从顺序改为双卡并行（GPU0=baseline, GPU1=geom）
- full 阶段原本就是双卡，无需改动
- 以后所有新脚本按此模板写

落点：
- docs/context/windsurf/user_preferences.md No.10
- MCM-PC/docs/project/MILESTONE_SOP.md v1.3 附录 C.4
- 本条 D20
```

## D19：W2.5 P4-fast-track — scale-only ICP-CD oracle on hierarchical（已锁定，2026-05-11）

```
触发：用户希望先看 ICP-CD 能否把 ModelNet-C scale 上 -0.40pp 拉回 ≥0pp，再决定要不要展开完整 W3-W5。

实验定位：
- 名称：P4-fast-track (scale-only)
- 与 D11 关系：跳过 P3 (T4 显存) / P5 (跨方法) / P1 (旋转探针) 的原顺序，
  但因为复用现有已 commit 的 hierarchical baseline (75.27% / scale -0.40pp)，
  P3/P5/P1 不构成本实验的硬前提。后续仍按 D11 顺序补做。
- 与 G1 关系：本实验直接为 G1 (ICP-CD 假设无证据) 提供首个数据点。
- 与 D17 关系：本实验是 diagnosis-driven TTA 的首条 evidence。

实验设计（用户已确认 4 项）：
1. 范围：只跑 ModelNet-C scale 5 个 severity（约 90 min 单卡），不跑全 35 setting。
2. 安全门控：v0.1 不加 ROC/AUC + CD margin gating，先看 raw ICP-CD 加权效果。
3. 不影响 D11：P3/P5/P1/P2 后续仍要做。
4. 代码隔离：新建 runners/model_with_hierarchical_icpcd.py，不破坏已 commit 的 hierarchical baseline。

成功判据（D17 三档）：
- Floor:   Δ vs ZS ≥ 0pp   → 可写"benchmark stress test 下拉回非负"
- Target:  Δ vs ZS ≥ +1pp  → 可写主卖点
- Stretch: Δ vs ZS ≥ +3pp  → AAAI Oral 候选

失败判据：
- Δ vs ZS < 0pp → C1 oracle 失败，必须 pivot：
  (a) 改用 oracle Option A (无 TTA loop，仅看 NN top-1 是否更准)
  (b) 改用 z-score + omega scan 替代 raw 加法
  (c) 最坏情况 pivot 到 corruption-specificity narrative

落地动作（按 D18 规则同步）：
- 代码：runners/model_with_hierarchical_icpcd.py + scripts/eval_p4_scale_icpcd.sh
- 论文：docs/paper/03_method.md (§3.4 ICP-CD v0.1 占位) + docs/paper/04_experiments.md (§4.1 scale 表占位)
- 决策：本条 D19
- 漏洞：跑出数字后，按结果移动 G1 (修补 / 升级风险 / pivot)
- 发现：按结果追加 F# 到 key_findings.md
```

## D18：经过理论或实验验证的新思路必须立刻写入论文草稿（含图表）（已锁定，2026-05-11）

```
用户原话（2026-05-11 10:51）：
"以后我们提到的新的思路，如果是经过理论验证或者实验验证，
 你需要分析，合理的内容，要及时写入论文草稿，包括图表。"

触发条件（任一即触发）：
- 理论验证：推导自洽、与现有 D# / G# / F# 不冲突
- 实验验证：P# 探针 / oracle / 复现 / ablation 跑出关键数字
- 用户明确确认采用某写法或思路

Cascade 落论文前必做的分析：
- (a) 与现有决策 / 漏洞 / 发现是否一致
- (b) 是否在 D16 / D17 / G1-G2 等 guardrail 范围内
- (c) 该写到哪个章节、是否需要新建图/表
- (d) 当前证据强度（preliminary observation / hypothesis / claim / contribution）

论文落地最小动作：
- 修改 MCM-PC/docs/paper/0X_*.md 对应章节
- 图表先用 [Tab. N: 描述] / [Fig. N: 描述] 占位 + 来源说明
- 章节顶部 ⚠️ 区写入证据强度
- 同步更新 decisions.md / key_findings.md / doc_gaps.md / 00_outline.md
- HTML 讲义 / 概念笔记不算"已写入论文"，必须真正落到 docs/paper/

不立刻写论文的情况：
- 仅是用户提问、思考片段、未确认猜想
- 与现有决策冲突且未解决
- 引用尚未跑通的实验数字

落点：user_preferences.md 第 9 条 + MILESTONE_SOP.md 步骤 2 + 附录 A。
```

## D17：MCP-3D 采用 diagnosis-driven TTA 作为候选主卖点（已锁定，2026-05-11）

```
用户提出的新论文定位：

现有许多 TTA / cache 方法主要从"提出新模块提升精度"的角度出发；
MCP-3D 可以强调：我们从根因上分析当前 SOTA 在某些 benchmark corruption 下为什么会出现负增益，
再根据诊断结果设计几何感知多记忆修复。

推荐卖点句：
"We provide a diagnosis-driven analysis of negative adaptation in cache-based 3D TTA, and use the diagnosis to motivate a geometry-aware multi-memory design."

必须满足的证据前提：
- P5：确认 scale 负增益是 hierarchical 特性还是 cache-family 共性。
- P1：证明 feature distance 在可控 corruption 下存在局部 failure。
- Feature-vs-geometry ROC/AUC：证明 ICP-CD 在某些区域提供互补证据。
- P2：用 compactness diagnosis 解释 memory drift / negative adaptation。

写作边界：
- 在证据完成前，只能写 preliminary observation / hypothesis。
- 证据完成后，可以作为 Introduction 的核心 selling point 和 Discussion 的机制贡献。
```
