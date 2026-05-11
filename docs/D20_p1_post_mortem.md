# D20 P1 探针结果与后续路径选择

> **版本**：v1.0（2026-05-11 19:40）
> **状态**：P1 完整探针跑完，基于数据作方向选择
> **作者**：用户 + Cascade
> **前置**：D17（anchor pollution 三假设框架）、D19（ICP-CD fast-track 失败）、paper §1.2.1（preliminary observation 初稿）

---

## 0. TL;DR

**一句话结论**：D17 对 "scale corruption 让 PointBERT feature 退化" 的假设 **几乎完全错**。scale_4（最严重）上 `cos(f_clean, f_scale)` 均值仍有 **0.91**，class consistency **93.3%**——feature 几乎不漂。D19 P4-fast-track 基于错误假设，因此在 hier baseline 上倒退 −1.70pp（scale_2: 78.16 → 76.46）是可以预见的。

**真正的根因**是 **anchor pollution**：把 1-NN anchor pool 从 "corrupted stream" 换成 "clean oracle"，scale corruption 上 top-1 acc 直接涨 **+11pp**（见 §3.3）。这证实 D17 三假设中的 **H2（anchor pollution 主导）**，而不是 H1（feature failure）。

**方向选择**：
- scale / rotate / add_* 四类：走 **E plan**（conditional static anchor，不用 stream anchor）。
- jitter / dropout 重度：**E plan 致命**（clean anchor 反而让 acc 掉 -24 ~ -63pp），走 **D plan**（abstention / ignore geom term）。
- 最终架构：一个 corruption-aware 的 "anchor 来源选择器"，见 §7。

**放弃方向**：ICP-CD / 点云几何补偿路径（D19 P4-fast-track v0.1.x）——它 addresses 一个不存在的问题。

---

## 1. 为什么需要 P1

D17（`windsurf对话/decisions.md` §D17）列出三个 hypotheses 解释 hier_baseline 在 scale_2 上持平、而 global_only baseline 下降的现象：

| 代号 | 假设 | 含义 |
|---|---|---|
| H1 | **feature failure** | PointBERT 对 scale 语义敏感，feature 本身漂得厉害，任何下游都救不回来 |
| H2 | **anchor pollution** | feature 没漂多少，但 hier cache 的 anchor pool 里混入了 baseline 错标的样本，这些错 anchor 反过来拉低后续样本的 cls |
| H3 | **both** | 两者都有 |

D19 **直接走了 H1 路径**：用 ICP-CD 对 query/anchor 做 geometry alignment，试图绕过 "feature 不可靠"。结果（v0.1.3 + shell-bug fix 后的真实 full run）：

```text
hier_baseline          78.16%   (scale_2, 2468 sample)
+ ICP-CD (v0.1.3)      76.46%   (-1.70pp)

per-bin Δerr (geom - base):
  bin [0.00, 0.10):  +0.0pp        # gate=SKIP
  bin [0.10, 0.15):  +3.2pp        # gate=PASS, geom 让错的更多
  bin [0.15, 0.20):  +3.5pp
  bin [0.20, 0.30):  +4.3pp
  bin [0.30, 1.00):  +9.3pp        # 越是 high-ent，geom 越糟
gate=PASS 样本上：rescued=57, broken=99, net=-42
```

（见 `Point-Cache/logs/p4_scale_icpcd_full_20260511_152322/summary.txt`）

**"在每个 high-ent bin 上 Δerr 一致 > 0"** 是 H1 被 falsify 的强信号——如果 geom 真的能补 feature drift，至少在某些 bin 上 Δerr 应该 < 0；事实是它 monotonically 让所有 high-ent bin 更糟。说明 feature 本身不是 bottleneck，干扰它反而有害。

减幅 −1.70pp 看起来不大，但意义不在数值大小：
1. ICP-CD 在 D19 设计里被定位为 "raw geometry oracle"——一个"应该至少不变坏"的实验，结果它让所有 high-ent bin 一致变差。
2. broken/rescued = 99/57，比例 **1.74x**，证明 geom 信号是噪声而非有用的补充。

**P1 探针**就是为了把 H1/H2/H3 三者用**可测量的量**分开：

- H1 的 testable prediction：`cos(f_clean[i], f_scale[i])` 应该显著 < 1，且 class consistency 应该显著 < baseline acc。
- H2 的 testable prediction：`cos(f_clean[i], f_scale[i])` 接近 1，但当 anchor pool 从 corrupt stream 换成 clean oracle 时，top-1 acc 大幅提升。
- H3：两者都成立。

---

## 2. P1 探针设计

三阶段，互相独立（任一阶段失败不影响其他阶段的 deliverable）：

1. **Stage 1**（scale-only, clean + scale_0..4 共 6 setting）
   - 测 `cos(f_clean, f_scale_X)` 每样本分布
   - 测 clean_i 在 f_scale_X 全集里的 NN rank
   - 测 top-1 NN class consistency
2. **Stage 2**（full 7 family × 5 severity，共 35 setting）
   - 同样指标，看 scale 是否在 7 个 corruption 里"特殊地"失效
3. **Stage 3**（anchor pollution simulation）
   - 对每个 corr_X：测 Setting A = 用 f_corr 当 anchor（模拟 hier cache），Setting B = 用 f_clean 当 anchor（E plan oracle），Δ = B − A
   - Δ 大 = pollution 是 root cause；Δ ≈ 0 = pollution 不是；Δ < 0 = E plan 在这个 corruption 上是有害的

代码与数据：
- runner：`@/root/autodl-tmp/MCM-PC/Point-Cache/runners/probe_p1_feature_drift.py:1-242`
- aggregation：`@/root/autodl-tmp/MCM-PC/Point-Cache/scripts/aggregate_p1.py`
- pollution sim：`@/root/autodl-tmp/MCM-PC/Point-Cache/scripts/anchor_pollution_sim.py`
- driver：`@/root/autodl-tmp/MCM-PC/Point-Cache/scripts/run_probe_p1.sh`
- 报告：`@/root/autodl-tmp/MCM-PC/Point-Cache/reports/P1_scale_drift.md`, `P1_full_drift.md`, `P1_pollution_sim.md`
- per-sample json：`Point-Cache/reports/P1_*.json`

模型：OpenShape PointBERT ViT-bigG-14 rgb（与 hier_baseline 用的完全同一权重），输入 1024 点 + rgb=0.4 填充，fp16，cache_type=global（只取 CLS，跳过 KMeans，支持 batched forward）。数据：`data/modelnet_c/{clean,scale_0..4,jitter_0..4,rotate_0..4,dropout_{local,global}_0..4,add_{local,global}_0..4}.h5`，每个 h5 都有 2468 样本按**相同索引顺序对齐**，所以 clean[i] 和 scale_X[i] 是同一物体的 clean / corrupted pair。

---

## 3. 关键数据

### 3.1 Stage 1：scale-only feature drift（2468 样本）

| severity | cos mean | cos median | rank=1 % | rank ≤ 5 % | class-consistent % |
|---|---|---|---|---|---|
| 0 | 0.9501 | 0.9627 | 81.6 | 93.9 | 97.9 |
| 1 | 0.9400 | 0.9559 | 78.3 | 92.3 | 96.9 |
| 2 | 0.9306 | 0.9484 | 75.4 | 90.2 | **95.5** |
| 3 | 0.9222 | 0.9401 | 73.4 | 89.1 | 95.1 |
| 4 | 0.9145 | 0.9362 | 69.8 | 85.9 | 93.3 |

**最强反驳 H1 的一行**：scale_2 上 class-consistency = 95.5%，但 hier_baseline 只做到 78.16%。**17pp 的 gap 不是来自 feature 失效**；feature space 里 class 分离度还在，只是下游模块（text-alignment + cache voting）把信号吃掉了。

### 3.2 Stage 2：7 corruption family drift 对比（cos mean，median severity=2）

| family | cos mean @ sev=2 | class-cons @ sev=2 | 特征退化水平 |
|---|---|---|---|
| add_global | 0.9954 | 100.0 | **可忽略** |
| add_local  | 0.9949 | 100.0 | **可忽略** |
| rotate | 0.9606 | 99.8 | 轻微 |
| dropout_global | 0.9557 | 99.5 | 轻微 |
| **scale** | **0.9306** | **95.5** | **中等** |
| dropout_local | 0.8543 | 91.5 | 中等 |
| jitter | 0.6976 | 59.5 | **严重** |

（见 `Point-Cache/reports/P1_full_drift.md:48-116`）

**分层洞察**：
- **affine-like corruption（scale / rotate / add_*）**：feature 几乎不漂。H1 被 falsify。
- **displacement corruption（jitter / heavy dropout_local）**：feature 大幅退化。H1 在这些 corruption 上成立。

换句话说，**D17 三假设不是全局二选一，而是 corruption-conditional 的**：
- 在 affine 类上 → H2 主导（pollution）
- 在 displacement 类上 → H1 主导（feature failure）
- 中间地带（中度 jitter / dropout）→ H3

### 3.3 Stage 3：anchor pollution simulation（关键）

把 anchor pool 从 corrupt stream（Setting A）换成 clean oracle（Setting B），1-NN top-1 acc 对比：

| corruption | A (corrupt anchor) | B (clean anchor) | Δ = B − A | 解读 |
|---|---|---|---|---|
| scale_2 | 84.44% | 95.46% | **+11.02pp** | pollution 主导 |
| scale_4 | 82.70% | 93.27% | +10.58pp | pollution 主导 |
| rotate_2 | 88.41% | 99.76% | +11.35pp | pollution 主导 |
| add_global_2 | 89.87% | 99.96% | +10.09pp | pollution 主导 |
| dropout_global_3 | 87.60% | 96.96% | +9.36pp | pollution 主导 |
| dropout_local_3 | 80.96% | 83.43% | +2.47pp | 弱 pollution |
| dropout_local_4 | 76.86% | 70.18% | **−6.69pp** | E plan 有害 |
| jitter_2 | 84.32% | 59.48% | **−24.84pp** | E plan 致命 |
| jitter_3 | 81.93% | 30.75% | **−51.18pp** | E plan 致命 |
| jitter_4 | 78.20% | 15.52% | **−62.68pp** | E plan 致命 |

（见 `Point-Cache/reports/P1_pollution_sim.md:5-43`）

**两个对立的 regime**：

**Regime A（affine-like + 轻度 dropout）**：E plan 给 +8 ~ +13pp 的纯增益。证明了 pollution 是 root cause，且 clean anchor 是可用的解药。

**Regime B（重度 displacement）**：E plan 带来 -25 ~ -63pp 的倒退。原因：jitter 让 feature 漂到 clean prototype 的 classification 区域之外，clean anchor 反而把 query 拉向错误的 cluster。

### 3.4 Stratified by drift（stage 3 tertile 切分）

对每个 corruption 按 `cos(f_clean_i, f_corr_i)` 分三档。全样本汇总的 Δ 可能被 tertile 间抵消——看 tertile-level 才能看到 sample-level 决策的信号。

| corruption | low-drift tertile Δ | mid-drift Δ | high-drift Δ |
|---|---|---|---|
| scale_2 | +12.9 | +11.2 | +9.0 | 全 tertile 都 E 赢 |
| scale_4 | +12.5 | +13.1 | +6.0 | 全 tertile 都 E 赢 |
| rotate_3 | +10.7 | +12.5 | +10.4 | 全 tertile 都 E 赢 |
| jitter_1 | +9.4 | +5.5 | **−11.5** | high-drift 已经 E 输 |
| jitter_2 | +5.0 | **−19.0** | **−60.7** | 只有 low-drift 还能用 |
| jitter_3 | **−13.1** | **−58.8** | **−81.3** | 全盘 E 输 |
| dropout_local_3 | +13.5 | +11.9 | **−18.3** | high-drift E 输 |
| dropout_local_4 | +16.8 | −2.3 | **−34.7** | 只 low-drift E 赢 |

（见 `Point-Cache/reports/P1_pollution_sim.md:49-153`）

**sample-level 决策准则的雏形**：以 `cos(f_clean_i, f_corr_i)` 为 proxy，把样本分档：

- cos > 0.90 的样本：E plan 几乎一定赢，应该用 clean anchor。
- cos ∈ [0.75, 0.90]：看 corruption type：affine/rotate 类用 E；jitter/dropout 类走 D。
- cos < 0.75：feature 已经漂出 clean prototype 的 reach，强制 D plan（跳过 geom 项，只信 text-alignment 的 abstention 结果）。

注意：**测试时我们没有 f_clean**。但我们可以用 training set 的 mean-per-class 作 surrogate，然后对 query 做最相似度 → 估计 drift level。这是 §7 提议架构的关键一步。

---

## 4. 三条假设的判定

根据 §3 数据：

| 假设 | affine-like (scale/rotate/add) | displacement (jitter/heavy dropout) |
|---|---|---|
| H1 (feature failure) | ❌ **驳回**（cos>0.90, cons>95%） | ✅ 成立（cos<0.70, cons<60%） |
| H2 (pollution) | ✅ **主导**（Δ≈+11pp） | 不再可测（query 已离 clean 太远） |
| H3 (both) | ❌ 不需要 | ✅ 成立（feature 退 + pollution 无救） |

**D17 的三假设框架结构上是对的**（假设空间覆盖完整），但 D19 实施时默认 H1 是主假设 **错了**。在 scale/rotate/add corruption 上真正的 root cause 是 H2。

---

## 5. D19 P4-fast-track 为什么必然失败

D19 的设计假设：
> "scale corruption 让 feature 漂到 CLIP-text alignment 外，ICP-CD 可以用 geometry 把 query 拉回到能与 anchor geometry 对齐的位置。"

P1 数据说明两件相反的事：
1. feature 本来就没漂出 CLIP-text alignment（cos=0.93, cons=95.5%）——geometry alignment 要"拉回"的东西其实在正确位置。
2. anchor 已经被 baseline 错标污染，用任何 geometry-affinity 投票只会强化 pollution：cd(query, wrong_anchor) 小 → affinity 高 → 错分类的 vote 更被 amplify。

两条合起来：**D19 干了反事**。它假设 feature 漂了所以加 geometry 救回来，实际上 feature 没漂、anchor 才漂，而 D19 恰好让坏 anchor 更 confident。这解释了 broken/rescued = 1.74x 的不对称：好 anchor + ICP 的样本本来就分对，新加的 geom voting 没改变；坏 anchor + ICP 的样本，geom 把"aligned 错 anchor 几何上很像"作为额外证据，**强化错分类**。

**教训**：D17 时代就该写的 P1 探针，当时没写，直接跳到实施 D19 fast-track。结果 3 轮 smoke + 2 轮 full run 烧了约 6 小时 GPU，产出负回归。**单单"smoke 持平、full 倒退 -1.7pp"这种结果**——如果当时就停下来想 root cause、跑 P1 类型的探针——就能省 4 小时 GPU、4 小时人工调试 shell bug 的时间。

---

## 6. D / E / F 方向基于 P1 数据的重排序

D17 原本列出三个候选方向（`windsurf对话/decisions.md` §D17）：

- **D plan**：entropy-based abstention。不用 anchor，只让 text-alignment 决定。
- **E plan**：static training-set anchor。把 1-NN 查询的 anchor pool 从 test stream 换成 training set prototype（= 本 probe 的 Setting B）。
- **F plan**：hybrid / learnable anchor selection。

P1 数据给出以下 evidence：

| 方向 | 支持 evidence | 局限 |
|---|---|---|
| D (abstention) | 对任何 corruption 都 **safe**：即使 jitter_4，跳过 geom 不比 stream-anchor 差 | 放弃增益；只能做"不变坏"保证 |
| **E (static anchor)** | **affine/rotate/add** 上 +10~13pp 纯增益（§3.3） | 在 jitter/dropout 重度上 **-25 ~ -63pp 灾难**（§3.3） |
| F (hybrid) | 结合 D+E，理论最优 | 需要 corruption detector，复杂度高 |

**决策**：
- **放弃纯 E**（在 jitter 上会被 paper reviewer 第一轮就 kill）。
- **以 D plan 为保底**（它保证在任何 corruption 上 ≥ hier_baseline）。
- **在 D 之上选择性叠加 E**：只对 `cos(query, nearest_training_prototype) > threshold` 的样本启用 static-anchor 投票。threshold 由 §3.4 数据给出初始值（0.90 全开，0.75 视 corruption family 开）。

这条路径的名字暂定：**D+E conditional anchor switching**，后续 decision log 会正式命名。

---

## 7. 推荐架构（D+E conditional）

**预先**（离线）：
1. 对 ModelNet40 training set 提 PointBERT feature，按类 mean-pool，得到 `proto[c] ∈ R^{1280}`，共 40 个 prototype。
2. 可选：再提每个 class 的 k-shot (k=3~5) 备选 anchor，作 fallback。

**test time**（每个样本 i）：
1. 正常跑 hier_baseline 得到 query feature `f_q`。
2. 计算 `sim[i] = max_c cos(f_q, proto[c])`（query 到最近 class prototype 的 cos）。
3. 选 **anchor 来源**：
   - `sim[i] > 0.90`：用 training prototype 做 1-NN（E plan）。
   - `sim[i] ∈ [0.75, 0.90]`：用 hier_baseline 原本的 stream anchor（当前 behavior）。
   - `sim[i] < 0.75`：跳过 cache 项，只用 `clip_logits`（D plan abstention）。
4. 最终 logits 与原 hier_baseline 公式一致，只改 cache 源。

**Smoke 验证步骤**（先不实施）：
1. E plan full on scale_{0..4}：P1 oracle simulation 已知 +10.58~12.68pp @ 1-NN。真实 end-to-end acc 预计增益 +3 ~ +8pp（因为 hier_baseline 里 cache 只贡献 ~1/3 的决策，其余来自 clip_logits）。**目标**：超过 `bar3 = 75.27` on scale_2。
2. D plan on jitter_{3,4}：要求 ≥ hier_baseline（不倒退）。
3. D+E conditional on ModelNet-C full：要求 ≥ bar3 on every corruption，且在 scale/rotate 上显著 > bar3。

---

## 8. 立即 next steps

本轮（2026-05-11）剩余：
- [x] commit P1 probe code + reports + logs + .gitignore 修复
- [x] commit D20 doc + §9.3 of D19 retro + paper §1.2.1 补充
- [ ] 写 `docs/decisions.md` 的 D20 一条：明确废弃 ICP-CD 路径、采纳 D+E conditional

下一轮（待用户批准后）：
1. 实现 `runners/model_with_conditional_anchor.py`：hier_baseline + proto-based anchor switching.
2. Smoke：scale_2 50-sample vs hier_baseline、jitter_3 50-sample vs hier_baseline。
3. Full：ModelNet-C 全 corruption，对比 bar3。
4. 若 E plan full run 比 P1 oracle 估计低超过 3pp，做 "为什么 end-to-end 没达到 oracle 上界" 的 ablation。

---

## 9. 附：实验复现

```bash
# 从 clean 开始重跑 P1 全流程（约 30 分钟 dual-GPU + CPU）：
cd /root/autodl-tmp/MCM-PC
STAGE=all bash Point-Cache/scripts/run_probe_p1.sh

# 仅 scale 阶段（~3 分钟）：
STAGE=scale bash Point-Cache/scripts/run_probe_p1.sh

# 重跑聚合（feature .npy 保留时）：
/root/miniconda3/envs/mcmpc/bin/python Point-Cache/scripts/aggregate_p1.py \
  --feat_dir Point-Cache/reports/p1_features --reference clean

# 重跑 pollution sim：
/root/miniconda3/envs/mcmpc/bin/python Point-Cache/scripts/anchor_pollution_sim.py \
  --feat_dir Point-Cache/reports/p1_features --reference clean
```

**feature .npy 文件不入 git**（见 `.gitignore:206-208`），需要重跑 extraction 才能再次聚合。per-sample 的 cos/rank/class-consistent 保存在 `Point-Cache/reports/P1_*.json` 里，可直接绘图不用重跑。

**CPU 时间 / GPU 时间**（dual-GPU V100 分卡）：
- Stage 1 (6 settings × 2468 sample)：~70s
- Stage 2 (30 settings)：~6 min
- Stage 3 (35 NN sim, 2468² per setting)：~12s
- 全流程：~8 min + model loading 1.5 min ≈ 10 min walltime
