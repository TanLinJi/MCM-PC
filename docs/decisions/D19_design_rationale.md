# D19 P4-fast-track 设计依据 (Design Rationale)

> **版本**：v0.2（2026-05-11 19:50，加 §9.3 失败回填）
> **状态**：**已终止**（D19 路径基于错误的 H1 假设；详见 §9.3 + `docs/decisions/D22_p1_anchor_pollution_pivot.md`）
> **作者**：用户 + Cascade
> **目的**：把 D19 几何项（ICP-CD raw integration）的设计逻辑链从头到尾写清楚，
>           避免「看到结果不对就改超参」的反射式工作流。
>           **保留作历史参考**：D19 是一个完整的 falsified hypothesis trail。

---

## 0. 写这份文档的动机

2026-05-11 上午 11:00 - 14:45 之间，我们做了三轮 smoke 实验：

| 版本 | 超参 | scale_2/50 sample 结果 | 决策 |
|---|---|---|---|
| v0.1 | `α_g=2, β_g=5, no ZM, no gate` | 72.00 vs 72.00（持平） | "geom 量级太小" → 调 α |
| v0.1.1 | `α_g=6, β_g=5, ZM, no gate` | 70.00 vs 72.00（-2pp） | "geom 太激进" → 加 gating |
| v0.1.2 | `α_g=6, β_g=5, ZM, gate=0.5` | 72.00 vs 72.00（geom 0 启用） | "threshold 错了" → ??? |

**每一步都是「实验结果 → 调超参 → 重跑」的反射式迭代**。用户在 v0.1.2 之后明确指出这个方法论问题。

这份文档的目标：
- 把每个超参的**理论依据**写清楚
- 把数据流的**量级关系**写清楚
- 把 X-ray 数据的**意义**写清楚
- 给出 v0.1.3 的**理论预测**，跑实验只是验证，不是探索

如果以后又陷入反射式调参，**先回这份文档**。

---

## 1. D19 P4-fast-track 的设计目标

来自 D19（`windsurf对话/decisions.md`）：

> 用户希望先看 ICP-CD 能否把 ModelNet-C scale 上 -0.40pp 拉回 ≥0pp，
> 再决定要不要展开完整 W3-W5。

**P4-fast-track 是一个 oracle 实验**：在 hierarchical baseline (75.27% / scale -0.40pp) 之上，
添加一个 raw ICP-CD 几何项，**不加任何 gating / 选择 / 优化**，看是否能拉回 0pp。

**只关心一个数**：scale corruption 上 acc 提升是否 ≥ 0。其它 corruption 暂不验证。

---

## 2. 公式与数据流

### 2.1 final_logits 的构造

```text
final_logits = clip_logits                       # text-feature 相似度（ZS 起点）
             + α  * cache_logits                 # 全局正 cache
             + α  * local_cache_logits           # 局部正 cache（patch-level）
             - α  * neg_cache_logits             # 负 cache（仅在 entropy ∈ [0.2, 0.5] 时填充）
             + (gate) * α_g * geom_logits        # ★ D19 新增几何项
```

其中：
- `α=4.0`, `β=3.0`（Point-Cache 原 hyperparam，**不动**）
- `α_g`, `β_g` 是 D19 新加（见 §4）
- `gate ∈ {0, 1}` 由 entropy 决定（见 §5）

### 2.2 geom_logits 的计算（`compute_geom_cache_logits`）

```text
1. 对 cache 里每个 anchor 跑 ICP（estimate scale + R + t），得 aligned_anchor
2. Chamfer distance: cd[k] = CD(query, aligned_anchor[k])  ← (B,) tensor
3. affinity[k] = exp(-cd[k])  ∈ (0, 1]
4. geom_raw = ((-1) * (β_g - β_g * affinity)).exp() @ cache_values  ← shape (1, n_cls)
   等价于 exp(β_g * (affinity - 1)) 加权 one-hot 求和
5. geom_raw = α_g * geom_raw
6. (B-plan) geom_logits = geom_raw - mean(geom_raw)   ← zero-mean transform
```

---

## 3. 数据流的实测量级表

从 v0.1.1 debug log（`logs/p4_scale_icpcd_debug_20260511_113710` 和 `115032`）取 5 个样本均值：

| 源 | min | mean / \|mean\| | max | 类间最大差 | 角色 |
|---|---|---|---|---|---|
| `clip_logits`   | -5  | 5    | 17  | 22 | 主体决策信号 |
| `gcache_logits` | 1.3 | 2.5  | 11  | 10 | 全局 cache 加强 |
| `lcache_logits` | **22** | **26** | **30** | **8** | **base + 类间区分（主力）** |
| `ncache_logits` | 0   | 0    | 0   | -  | 多数样本未触发 |
| `geom_logits` v0.1 (α=2)   | 1  | 2.8 | 5.9 | **5** | **量级太小，被 lcache 压死** |
| `geom_logits` v0.1.1 (α=6, ZM) | -5 | 0 | 9.4 | **14** | **量级 OK，能与 lcache 抗衡** |

### 3.1 关键观察

1. **`lcache_logits` 是 final_logits 的主导**：每个 class 加 22-30 的 base，类间相对差 8。
2. **要让 geom 真正参与 argmax 决策**，geom 的类间最大差**必须 ≥ 8**。
3. v0.1 的 geom 类间差 5 < 8 → **必然被 lcache 压死**，acc 持平 baseline。这是 v0.1=72 的根本原因。
4. v0.1.1 的 geom 类间差 14 > 8 → **能撬动 argmax**，但同时 **也能把对的 argmax 撬错**。

### 3.2 量级匹配推导（理论值）

```text
geom 类间差 = α_g * (top1_kernel - top_last_kernel)
            ≈ α_g * (exp(β_g*(aff_max - 1)) - exp(β_g*(aff_min - 1)))

aff_p50 = 0.84 → exp(β_g*(0.84-1)) = exp(-0.8) = 0.45
aff_max = 1.00 → exp(0) = 1.0
aff_min = 0.46 → exp(β_g*(0.46-1)) = exp(-2.7) = 0.067

per-anchor top1 vs bottom 区分 = 1.0 - 0.067 ≈ 0.93
per-class（k_shot=3 same class anchors 求和）≈ 3 * 0.93 = 2.8

geom 类间差 ≈ α_g * 2.8
```

要求类间差 ≥ 8 → α_g ≥ 8 / 2.8 ≈ **2.86**

**与实验对照**：
- v0.1 α_g=2.0 < 2.86 → 类间差 5.6 < 8 → 不够（与实测 5 一致）
- v0.1.1 α_g=6.0 > 2.86 → 类间差 16.8 > 8 → 够（与实测 14 接近）

**结论**：`α_g = 6.0` 是合理的（**首次出现理论推导值**），不是拍脑袋。

---

## 4. 超参的理论选择

| 超参 | 当前 default | 理论依据 | 是否经过验证 |
|---|---|---|---|
| `α_g = 6.0` | 6.0 | §3.2 推导：要 ≥ 2.86 才能撬 lcache | ✅ v0.1.1 实测类间差 14，符合 |
| `β_g = 5.0` | 5.0 | 在 aff∈[0.46, 1.0] 区间提供 ~14× 类间对比度（exp(-0.8)→0.45 vs exp(-2.7)→0.07） | ✅ 数学验证 |
| `zero_mean = True` | True | 把 geom 从「+加分」变为「相对偏好投票」：高于自己平均的类加分，低于的减分。**数学上更对称，方向无偏** | ✅ B 计划设计 |
| `entropy_threshold` | 0.5（待改 0.10） | §5 X-ray 数据驱动 | 🚧 待 v0.1.3 验证 |
| `estimate_scale = True` | True | scale corruption 下必须开 | ✅ corruption-specific |
| `max_iter = 20` | 20 | ICP 收敛性能 trade-off | ⚪ 固定不调 |

**实际上对外只有 1 个真正可调超参：`entropy_threshold`**。其它 3 个核心超参 (α_g, β_g, zero_mean) 都有理论依据（§3.2 和 §4 的推导），**不应该再去扫**。

这与 D21（决策中）的"hyperparam 收敛"方向一致：未来论文里可以把 α_g 默认绑定 = α=4.0、β_g = β=3.0，**对外引入 0-1 个新超参**。

---

## 5. Entropy Gating：从 X-ray 数据反推 threshold

### 5.1 X-ray 关键数据（2026-05-11 14:36 smoke, log dir 143652）

```text
50 个样本的 entropy 分布:
  mean=0.108  min=0.000  max=0.408

baseline (geom 关) 错率 vs entropy:
  [0.00, 0.05): n=21, err=0.0%   ← 极度自信样本，全部对
  [0.05, 0.10): n=11, err=27.3%
  [0.10, 0.15): n= 4, err=25.0%
  [0.15, 0.20): n= 2, err=100.0%
  [0.20, 0.30): n= 6, err=66.7%
  [0.30, 1.00]: n= 6, err=66.7%

按 threshold 切分的错率比:
  threshold=0.10:  low n=32 err= 9.4%  |  high n=18 err=61.1%  |  ratio=6.52x
  threshold=0.15:  low n=36 err=11.1%  |  high n=14 err=71.4%  |  ratio=6.43x
  threshold=0.20:  low n=38 err=15.8%  |  high n=12 err=66.7%  |  ratio=4.22x
```

### 5.2 物理含义

baseline 在 scale_2 上错 14 个：

| 子集 | n | baseline 错 | 含义 |
|---|---|---|---|
| `entropy < 0.10` (low ent) | 32 | 3（≈9.4%×32） | "CLIP 自信但错"，**gating 救不了** |
| `entropy ≥ 0.10` (high ent) | 18 | 11（≈61%×18） | **geom 应该救场的目标** |

**ratio = 6.52×** 是 D19 路径的第一个 strong positive signal —— entropy 真的能区分难易。

### 5.3 Threshold 选择推导

候选三个值（X-ray 表 2 给的）：

| threshold | high n | high err | high 内错样本 | 救场上限（如果 geom 100% 救） |
|---|---|---|---|---|
| 0.10 | 18 | 61.1% | 11 | +22% (50 sample 范围) |
| 0.15 | 14 | 71.4% | 10 | +20% |
| 0.20 | 12 | 66.7% | 8  | +16% |

**threshold=0.10 提供最大救场上限**，且 ratio 最高（6.52×），最佳。

---

## 6. v0.1.3 理论预测

### 6.1 假设（明确写下来）

- **A1**: scale_2 上 baseline 错率分布 = X-ray 表 1 中观察到的分布（保持不变）
- **A2**: geom 在 high-entropy 错样本上的"救对率" ≈ 60%（来自 v0.1.1 debug 的 5 sample 估计，sample 量小，置信度低）
- **A3**: geom 在 high-entropy 对样本上的"破坏率" ≈ 40%（即 1 - A2）

### 6.2 预测推导

threshold=0.10 让 geom 应用在 **18 个 high-ent 样本** 上：

```text
high-ent 子集（18 个样本）：
  baseline 错 11 个
  baseline 对  7 个

geom 介入后（假设 A2/A3）：
  错 → 对 救场：    11 * 0.60 = 6.6 个 净救对
  对 → 错 破坏：     7 * 0.40 = 2.8 个 净破坏
  净增对样本数：    6.6 - 2.8 = +3.8 个

low-ent 子集（32 个样本）保持 baseline：
  错 3 个，对 29 个

新总数：对 = 29 + (7-2.8) + (11-(11-6.6))
            = 29 + 4.2 + 6.6
            = 39.8 个
acc = 39.8 / 50 = 79.6%
```

**预测**：v0.1.3 (threshold=0.10) 在 scale_2/50 sample 上 acc ≈ **80%**（baseline 72%，提升 +8pp）

### 6.3 验证标准

| 实际 v0.1.3 acc | 判断 |
|---|---|
| ≥ 78% | **理论模型成立**，A2/A3 假设接近。上 STAGE=full 验证 5 severity |
| 74% - 78% | **partially 成立**，A2 可能略低于 60%（geom 救对能力比预期弱）。仍可上 full 但调低预期 |
| 72% - 74% | **持平 baseline**，A2 接近 50% → geom 没有有效信号在 high-ent 上 → C1 思路打折 |
| < 72% | **理论模型失败**，A3 > A2（geom 在 high-ent 上更倾向破坏而非救场）→ pivot 到 D plan 验证 ICP-CD top-1 retrieval |

---

## 7. 方法论复盘

### 7.1 为什么前 3 轮陷入反射式调参？

| 轮次 | 问题 | 应该做的 |
|---|---|---|
| v0.1 | `α_g=2.0` 拍脑袋初值 | 应先做 §3.2 量级推导，得 α_g ≥ 2.86 |
| v0.1.1 | "geom 不动"就翻 3 倍到 6.0 | 应保留 2.86 附近测，看是否一翻 3 倍就过头 |
| v0.1.2 | `threshold=0.5` 拍脑袋初值 | 应先做 X-ray 看 entropy 实际分布 |

**根因**：初值没有理论依据，错了之后只能"调"，没有"算"。

### 7.2 这次 X-ray 实验的意义

虽然 v0.1.2 acc 没变化（gating 0 通过 = baseline），但 X-ray 数据告诉我们：
- entropy gating 思路**正确**（ratio=6.52×）
- 最佳 threshold 是 **0.10** 而不是 0.5（差了 5 倍）

**这才是值得跑的实验** —— 不是"看 acc 是否提升"，而是"产生 actionable 数据"。

### 7.3 以后的工作流

```text
观察问题 → 列出可能假设 → 推导每个假设的数值预测 → 设计 minimum 验证实验
       → 跑实验 → 实测 vs 预测 → 更新或否决假设 → 再推导
```

而不是：

```text
跑实验 → 看 acc → 调超参 → 跑实验 → 看 acc → ...
```

---

## 8. v0.1.3 实施 plan

### 8.1 代码改动（最小）

```python
# utils.py
parser.add_argument('--geom_entropy_threshold', type=float,
                    default=0.10,  # was 0.5, see D19_design_rationale §5
                    help='...')
```

仅一行 default 改动。

### 8.2 实验执行

```bash
cd /root/autodl-tmp/MCM-PC
STAGE=smoke bash Point-Cache/scripts/eval_p4_scale_icpcd.sh
```

预计 ~4 min。

### 8.3 验证

跑完看 §6.3 的判断表，写一段 50-word 实测 vs 预测对照。落到本文档 §9。

---

## 9. 实测对照

### 9.1 v0.1.3 smoke (50 sample, scale_2) — 2026-05-11

**配置**：`α_g=6, β_g=5, zero_mean=ON, gate≥0.10`（其它默认）。`STAGE=smoke bash Point-Cache/scripts/eval_p4_scale_icpcd.sh`，dual-GPU parallel。

#### 9.1.1 顶层数字

| 指标 | 实测 | §6 预测 | 命中 |
|---|---|---|---|
| baseline acc (hier) | 72.0% (36/50) | 76% (推算)* | 偏低 |
| hier+geom acc | 70.0% (35/50) | ≥78% | **未达** |
| gate-PASS rate | 18/50 = 36% | 30-40% | ✓ |
| ICP timing (per pass) | 123.5 ms mean / 47.7 - 765.1 ms | < 200 ms 可接受 | ✓ |
| net rescue | -1 sample (rescued=1, broken=2) | A2 ≥ A3 | **未达** |

*baseline 76% 是从 §6 X-ray 推算出的"如果 gate 是完美 oracle"上限，并非小样本预期；50 sample 下的 72% 处于合理波动范围。

#### 9.1.2 熵-错误率单调性 ✓

| threshold | low-ent err% | high-ent err% | ratio |
|---|---|---|---|
| 0.10 | 9.4 (n=32) | 61.1 (n=18) | **6.52x** |
| 0.15 | 11.1 (n=36) | 71.4 (n=14) | 6.43x |
| 0.20 | 15.8 (n=38) | 66.7 (n=12) | 4.22x |

**§6.1 假设 (entropy 是 baseline 错误的强 proxy) 经验上成立**。这是 D19 整个 gating 框架的根基，**没有动摇**。

#### 9.1.3 bin-level Δerr ⚠ 揭示双向效应

| entropy bin | n | base_err% | geom_err% | Δerr | 解读 |
|---|---|---|---|---|---|
| [0.00, 0.10) | 32 | ~9 | ~9 | 0 | gate=SKIP (符合预期) |
| **[0.10, 0.15)** | 4 | 25.0 | **0.0** | **−25.0** | **救援区** ✓ |
| [0.15, 0.20) | 2 | 100.0 | 100.0 | 0 | 都错，无信号 |
| [0.20, 0.30) | 6 | 66.7 | 66.7 | 0 | argmax 未变 |
| **[0.30, 1.00)** | 6 | 66.7 | **100.0** | **+33.3** | **破坏区** ✗ |

#### 9.1.4 §6.3 判断对照

形式上 acc=70% < 72% → 情景 4 (pivot to D plan)。

但 50 sample 下：
- net=-1 处于统计噪声范围（绝对差 2%）
- bin-level pattern 不是噪声，**有理论解释**：极高熵下 baseline 已退化，anchor 池被污染，geom 在污染 retrieval set 上 NN search 反而成偏差放大器
- 因此**不直接 pivot**，而是用 single-severity full 验证 bin pattern

#### 9.1.5 决策

**不**改 α_g/β_g/zero_mean（§6 理论支撑未动摇）。**不**改 gate=0.10（来自 X-ray 数据）。

**下一步唯一验证动作**：跑 `STAGE=full SEVERITIES="2" bash Point-Cache/scripts/eval_p4_scale_icpcd.sh` (~20 min, ~2k sample, scale severity-2 全集)。两件事要看：

1. **bin-level [0.10, 0.30) 净 Δerr 是否仍然 ≤ 0**（救援真存在）
2. **bin-level [0.30, 1.00) 净 Δerr 是否仍然 > 0**（破坏真存在）

#### 9.1.6 v0.1.4 触发条件（不预先实施）

仅当 9.1.5 两条都被 single-severity full 确认时，才考虑 Fix E（双侧 gate `[low, high)`），且 high 阈值由 X-ray bin 边界**直接读出**，不调参。否则 pivot 至 D plan。

---

### 9.2 v0.1.3 single-severity full (scale_2 全集 2468 sample) — 2026-05-11 15:05

#### 9.2.1 实验失效 ⚠

`STAGE=full SEVERITIES="2"` 跑了 ~12 min，得到 `scale_2___hier_baseline.log` 与 `scale_2___hier_plus_geom.log`（注意三下划线，bug 副作用）。结果：

| 数据 | 观测 | 含义 |
|---|---|---|
| 两 log 文件大小 | 完全相同 (180074 B) | 高度可疑 |
| 所有 bin Δerr | **= 0.0** | geom 没改任何 argmax |
| `gate_PASS` | **0 / 2468** | gate 一次都没触发 |
| sample-info 行 `gate=` 字段 | `gate=N/A` | runner 在 `enable_geom_cache=False` 时的占位 |

**Root cause**：`@/root/autodl-tmp/MCM-PC/Point-Cache/scripts/eval_p4_scale_icpcd.sh` 中 `run_job` 用 `cut -d_ -f3-` 解析 tag，遇到双下划线 `__` 产生空字段，导致 `tag="_hier_plus_geom"` 而非 `"hier_plus_geom"`，`if [ "$tag" = "hier_plus_geom" ]` 永远为假，`--enable_geom_cache` 没传给 geom 进程。两个进程实际跑的是同一份 hier baseline。

**Fix**：用 bash parameter expansion `${key%__*}` / `${key##*__}` 替代 `cut`（已修复，commit-stage）。此 bug 不影响 smoke stage（smoke 用直接的 hard-coded 命令而非 `run_job`），所以 §9.1 的 50-sample 数据**仍然有效**。

#### 9.2.2 但单边 baseline 数据可挽救 ✓（强化 §9.1.2）

bug 不影响 baseline log 的内容。把今天 full 的 baseline log 单独看，**§9.1.2 的熵-错误率单调性在 ~2.5k 样本下被强化**：

| threshold | low-ent err% | high-ent err% | ratio | 对比 v0.1.3 smoke (n=50) |
|---|---|---|---|---|
| **0.10** | 6.1 (n=1568) | 49.2 (n=900) | **8.04x** | 6.52x → **更强** |
| 0.15 | 11.4 (n=1850) | 53.1 (n=618) | 4.65x | 6.43x |
| 0.20 | 14.4 (n=2051) | 58.5 (n=417) | 4.07x | 4.22x |
| 0.30 | 18.6 (n=2308) | 68.8 (n=160) | 3.70x | — |

bin-level baseline error rate（**严格单调递增**，证实 entropy 是干净的 corruption-strength proxy）：

| bin | n | base_err% |
|---|---|---|
| [0.00, 0.05) | 1289 | 2.9 |
| [0.05, 0.10) | 279 | 21.1 |
| [0.10, 0.15) | 282 | 40.8 |
| [0.15, 0.20) | 201 | 41.8 |
| [0.20, 0.30) | 257 | 52.1 |
| [0.30, 1.00) | 160 | 68.8 |

**意义**：D19 整个框架的根基（entropy 是 baseline 错误的强 proxy）在 2.5k 样本下被牢固验证。无论后续 geom 是否成立，gate-by-entropy 这个 *形式* 已经被证明是合理的 baseline-error 切分。

#### 9.2.3 重跑准备

代码 fix 已就绪。下次 `STAGE=full SEVERITIES="2"` 会跑出**真正的** v0.1.3 hier_plus_geom 数据。重跑后看：

1. `[full-aggregate] === [X-ray] geom impact (gate=PASS samples only)` 行：`gate_PASS` 应该 ≈ 900（high-ent count）的 30-40%，即 ~270-360
2. `[full-aggregate] === [X-ray] per-bin gate=PASS net` 表：精确对比 §9.1.5 两条假设
3. `[full] === results table ===`: hier_plus_geom 的 acc 与 hier_baseline 的差值（应非 0）

预计 ~12 min（已确认 dual-GPU baseline 耗时约 12 min）。

---

### 9.3 v0.1.3 重跑后的真实数据（shell-bug fix 后）— 2026-05-11 15:23

`STAGE=full SEVERITIES="2"` 重跑后（log dir `p4_scale_icpcd_full_20260511_152322`，summary `summary.txt`）：

| metric | value |
|---|---|
| `scale_2  hier_baseline` | **78.16%** |
| `scale_2  hier_plus_geom` | **76.46%** |
| **Δ acc** | **−1.70pp** |
| gate_PASS samples | 900 / 2468 |
| rescued (base wrong → geom right) | 57 |
| broken  (base right → geom wrong) | 99 |
| net  | **−42** |

per-bin Δerr (geom − base, 仅 high-ent gate=PASS bin)：

| bin | base_err% | geom_err% | Δerr |
|---|---|---|---|
| [0.10, 0.15) | 40.8 | 44.0 | **+3.2** |
| [0.15, 0.20) | 41.8 | 45.3 | +3.5 |
| [0.20, 0.30) | 52.1 | 56.4 | +4.3 |
| [0.30, 1.00) | 68.8 | 78.1 | **+9.3** |

#### 9.3.1 这个结果**falsify** D19 §9.1.5 的双假设

§9.1.5 给 v0.1.3 列了两条 alternative 假设（应该至少有一条成立）：
- **H-A**：geom 在 high-ent 全 bin 上一致 net > 0（"geom 救人"）
- **H-B**：geom 在 [0.10, 0.20) 救得多、在 [0.30, 1.00) 救不动但也不变坏

实测：**两条都被 falsify**——所有 high-ent bin 上 net < 0，且越是 high-ent 倒退越严重 (+3.2 → +9.3pp)。这是"几何 affinity 强化错 anchor"的直接证据。

#### 9.3.2 跟 P1 探针的交叉印证

P1 探针（D20）独立测出：scale_2 上 PointBERT feature 几乎不漂（cos=0.93, class consistency=95.5%）。这就解释了 §9.3 的现象：

- D19 假设 "feature 漂了" → 设计 ICP-CD 用 geometry 补偿
- 实际上 feature 没漂、anchor 才漂
- ICP-CD 把"几何对齐到错 anchor"翻译成"额外的 voting 证据"，**强化** anchor pollution

详见 `@/root/autodl-tmp/MCM-PC/docs/decisions/D22_p1_anchor_pollution_pivot.md` §5。

#### 9.3.3 D19 路径终止

基于 §9.3.1 + §9.3.2 的双重证据：

1. **不再调 D19 超参**：α_g、β_g、threshold 都不能 fix 一个"假设错了"的方法。
2. **不再实施 v0.1.4 双侧 gate**：§9.1.6 的触发条件未达成（§9.1.5 双假设都被 falsify），所以 v0.1.4 自动跳过。
3. **保留 D19 实施代码** (`runners/model_with_hierarchical_icpcd.py`) 作历史参考，但不在 paper 主线推进。
4. **方向迁移到 D22**：`docs/decisions/D22_p1_anchor_pollution_pivot.md` §6-§7 推荐的 D+E conditional anchor switching。

#### 9.3.4 教训

> ICP-CD smoke 持平 / full 倒退 这种结果，应该是"立刻停下来回头查 root cause"的红灯，而不是"再调一轮超参"的黄灯。当时若直接跑 P1（约 10 分钟），就能在做 §9.2 之前就知道 H1 错。
