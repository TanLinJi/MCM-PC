# MCP-3D 压缩上下文卡片（2026-05-10 23:01）

## 1. 当前一句话状态

W1、W2 已完成并打 tag `w2-tta-baseline`；当前处于 W2.5 前的论文复习暂停状态，已完成“故事”和“鸟瞰”，现在回到论文主线的第三阶段：“概念”。

## 2. 用户当前偏好

- 复杂内容优先输出为自包含 HTML，而不是长 Markdown。
- 讲解顺序采用“故事 → 鸟瞰 → 概念”。
- 写作和实验并行推进，每个实验阶段都要同步形成论文材料。
- 每次里程碑完成后执行 `MCM-PC/docs/project/MILESTONE_SOP.md` 的 7 步 SOP。
- 重要上下文要及时保存，方便新会话恢复。

## 3. 已完成论文复习内容

- 故事线：动机 → Point-Cache 软肋 → MCP-3D 三个核心贡献 → 预期结果。
- 鸟瞰线：16 周阶段规划、W2.5 探针实验、W3-W5 主实验、风险门控。
- Related Work v0.1：`MCM-PC/docs/paper/02_related_work.md`。
- SOP v1.0：`MCM-PC/docs/project/MILESTONE_SOP.md`。
- HTML 复习报告：`MCM-PC/docs/reports/2026-05-10_review_session.html`。

## 4. 当前应继续的主线

从 `MCM-PC/docs/concepts/00_overview.md` 启动概念阶段，先解释：

1. TTA 问题定义。
2. 3D TTA 为什么比 2D 难。
3. Point-Cache 的机制。
4. MCP-3D 要修的 3 个软肋。
5. C1-C4 贡献和风险等级。
6. 2×3 memory matrix 的整体作用。

## 5. 已修补漏洞

- G4：性能目标拆成 Floor / Target / Stretch 三档。
- G6：boundary memory 不是负样本，而是校准信号。
- G7：vMF 只用于文本侧，不用于点云侧的理由已补充。

## 6. 待修漏洞

仍需后续通过实验或写作修补 G1、G2、G3、G5、G8。

## 7. 下一步动作

- 生成 `concepts/00_overview.md` 对应的 HTML 概念复习页。
- 保持风格：少公式、多结构图、突出“为什么做 → 怎么做 → 结果期待”。
- 概念复习完成后再恢复 W2.5：P3 → P5 → P1 → P2。
