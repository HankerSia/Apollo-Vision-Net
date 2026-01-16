# DiffusionDriveV2 代码级实现解析（更偏工程落地）：强化学习（RL, Reinforcement Learning）+ Truncated Diffusion 如何把“多模态”做成“可用的多模态”

> 面向读者：想把论文读成“能跑的工程”，并且希望知道每个关键公式/模块在代码里具体落到哪。
>
> 参考：
> - DiffusionDriveV2 论文（arXiv）：https://arxiv.org/html/2512.07745v1
> - 官方代码仓库：https://github.com/hustvl/DiffusionDriveV2
> - 训练/评估说明：https://github.com/hustvl/DiffusionDriveV2/blob/master/docs/train_eval.md
> - 对比论文（DIVER / arXiv:2507.04049）：https://arxiv.org/abs/2507.04049
>
> 备注：本文聚焦 **DiffusionDriveV2** 的代码实现路径（不是 DiffusionDrive V1）。文中涉及的“anchor / truncated diffusion / GRPO / PDM（PDMS）”等术语，尽量以论文与代码为准。

---

## 1. 先把结论讲清楚：DiffusionDriveV2 在解决什么核心问题？

在工程落地中，挑战并不在于“生成多条轨迹”，而在于：

1) **多模态不等于有效多样性**：如果训练目标仍是单 GT（Ground Truth，标注专家轨迹）的 imitation learning（IL, Imitation Learning，模仿学习），多模态输出往往向 GT 附近集中（mode collapse），或产生大量不可行样本（碰撞/越界/不舒适）。
2) **评价指标不对齐**：纯 open-loop 的 L2/L1，只鼓励贴近单条 GT，无法对“多意图覆盖”和“闭环可执行性”给出直接监督。

DiffusionDriveV2 的策略可以概括为一句话：

> **保留“anchor + truncated diffusion”的多模态生成框架，同时把 diffusion 采样过程当成 stochastic policy，用 GRPO（Group Relative Policy Optimization，一类组相对策略优化）风格的 RL 把质量下界抬起来，并通过探索噪声机制在采样空间中显式引入多样性。**

对应到代码层面对外表现就是：

- 推理仍然快（truncated diffusion，常见两步 DDIM：Denoising Diffusion Implicit Models，一类用于加速扩散采样的隐式去噪采样器）。
- 训练不再仅依赖 IL：会计算奖励（PDM/PDMS 等闭环 proxy；PDM = Planning Decision Metric，PDMS = PDM Score），并用 logprob（log-probability，对数概率）进行 policy gradient（策略梯度）更新。
- 为了避免跨意图的直接比较导致模式坍缩，采用 **intra-anchor** 的组内相对优势；同时用 **truncated / hard constraint** 在跨意图层面施加底线安全约束。

---

## 2. 读代码不迷路：论文主线 → 代码导航图

把论文的“输入 → 生成 → 评估 → 更新”流程，映射成 5 个你在仓库里能直接搜到的落点：

1) **感知/特征抽取（Transfuser backbone）**
- 位置：`navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py`
- 关键类：`V2TransfuserModel` + `TransfuserBackbone`

2) **多模态轨迹生成（anchor-conditioned truncated diffusion）**
- 位置：同文件里的 `TrajectoryHead`
- 关键资产：`plan_anchor` / `plan_anchor_path`
  - 常见 shape：`(20, 8, 2)`（20 个意图簇，每条 8 个 waypoint，xy 坐标）

3) **探索噪声 + logprob**
- 位置：自定义 `DDIMScheduler_with_logprob.step()`（在 `diffusiondrivev2_model_rl.py`/`diffusiondrivev2_model_sel.py` 中出现）
- 作用：把“采样一步”变成可计算 $\log \pi_\theta$ 的 stochastic transition，为 RL 链路提供梯度通道。

4) **RL：intra-anchor GRPO / truncated GRPO（优势函数与截断逻辑）**
- 位置：`TrajectoryHead.forward_train_rl()` / `TrajectoryHead.get_rlloss()`

5) **Mode selector（coarse-to-fine 两阶段选择器）**
- 位置：`navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py`
- 直观理解：生成器负责生成候选，selector 负责最终决策。生成器输出越稳定，selector 的决策负担就越小。

---

## 2.1 论文 Figure 2（Overall architecture）在代码里怎么对应？

Figure 2 本质上画的是一条“**感知 → 多模态生成 → 质量优化（RL）→ 决策选择（selector）**”的链路；如果你把它当作代码导读图，建议按下面 4 个模块理解：

1) **Condition encoder（条件编码 / 场景表征）**
   - 输入：多传感器与状态特征（camera/lidar/history/status 等）。
   - 输出：BEV（Bird’s-Eye View，俯视栅格）等中间表征，供后续规划头做 cross-attention。
   - 代码对应：`V2TransfuserModel` / `TransfuserBackbone`（`diffusiondrivev2_model_rl.py`）。

2) **Anchor-conditioned trajectory diffusion（基于 anchor 的截断扩散生成器）**
   - 输入：场景表征 + `plan_anchor`（意图先验）。
   - 输出：每个 anchor 下的一组轨迹候选（多模态 proposals）。
   - 关键点：Figure 2 的 “truncated diffusion” 对应到实现里通常是**非常少的采样步数**（例如两步 DDIM），用来把推理时延压到规划可用的量级。
   - 代码对应：`TrajectoryHead` + `DDIMScheduler_with_logprob.step()`。

3) **RL on diffusion as policy（把扩散采样当 stochastic policy 做 RL）**
   - Figure 2 里最容易忽略的一点：RL 优化的不是“轨迹点本身”，而是**扩散采样链路的概率模型**。
   - 实现上会把每一步采样产生的 $\log \pi_\theta$
     （对数概率）记录下来，然后用 GRPO（Group Relative Policy Optimization）风格优势函数做 policy gradient（策略梯度）。
   - 代码对应：`forward_train_rl()` / `get_rlloss()` + scheduler 里返回 `log_prob` 的改造。

4) **Mode selector（候选打分与最终决策）**
   - Figure 2 的右侧通常对应 “scoring / selection”：在多条候选中选出最终执行轨迹。
   - 实现上会把候选轨迹 token 当成 query，与 BEV/agent token 做 cross-attention，然后输出每条候选的 score。
   - 代码对应：`diffusiondrivev2_model_sel.py` 中的 scorer decoder（例如 `ScorerTransformerDecoder`），并结合 BCE（Binary Cross-Entropy，二元交叉熵）/ranking loss 做训练。

如果你用一句话去“翻译” Figure 2：**DiffusionDriveV2 用 anchor 把多模态拆成若干意图簇；用 truncated diffusion 低成本生成每簇候选；用 GRPO 式 RL 抬高每簇候选的质量下界；最后用 selector 在全局语境下做最终选择。**

## 2.2 论文 Figure 5（Qualitative comparison）在说明什么？

Figure 5 给了一个非常直观的“效果侧证”：在 NAVSIM navtest split 的转弯（turning）场景里，对比 **Vanilla Diffusion / DiffusionDrive / DiffusionDriveV2** 的轨迹可视化，可以把三者差异理解成从“能生成”到“能用”的三个阶段。

1) **Vanilla Diffusion：多样性有，但可执行性弱**
   - 在转弯场景里更容易出现轨迹几何形状不合理（例如转弯半径不连续、轨迹飘到不可行驶区域），本质原因是：它主要在做分布拟合，缺少面向驾驶可执行性的显式约束。

2) **DiffusionDrive（V1 思路）：更贴近驾驶先验，但仍容易“围着单 GT 收敛”**
   - 引入 anchor / intention prior 后，轨迹的宏观形态会更像“可驾驶”的一组模式；但如果训练信号仍以单 GT（Ground Truth，标注专家轨迹）的模仿为主，很容易看到候选在关键决策点附近趋同（多模态“看起来多”，但有效差异不足）。

3) **DiffusionDriveV2：候选轨迹更干净、转弯更稳定，且模式之间更分离**
   - Figure 5 里更常见的现象是：转弯轨迹的曲率变化更平滑、越界/碰撞类硬失败更少；同时不同候选之间不会全部挤在同一条路径附近。
   - 这和代码侧的两处改造对应得很紧：
     - **truncated diffusion + DDIM**：用很少的采样步数把推理时延压下去，保证在线规划可用；训练阶段再用更合适的优化信号补偿质量。
     - **RL（Reinforcement Learning，强化学习）+ GRPO（Group Relative Policy Optimization）+ selector**：用 reward/truncation 把“硬失败”压下去，用组内优势保持多模态结构，最后由 selector 在全局语境里挑出最优执行轨迹。

如果你在读代码时想把 Figure 5 变成“检查清单”，可以关注三类可视化特征：**（1）转弯曲率是否连续；（2）是否越界/碰撞；（3）候选之间是否存在可用的意图差异，而不是在同一路径附近做细微抖动。**

## 3. 工程启动路径：训练/评估入口在哪里？

DiffusionDriveV2 基于 NAVSIM（NVIDIA Autonomous Vehicle Simulation，自动驾驶仿真/评测工具链）devkit + Hydra（Python 配置管理框架）配置启动。你如果想复现，建议按下面的层次去理解：

### 3.1 缓存（强烈建议先做）

- `navsim/planning/script/run_dataset_caching.py`
- `navsim/planning/script/run_metric_caching.py`

缓存的意义：将特征/指标预计算，从而避免训练过程中每个 step 都进行高开销的 I/O（Input/Output，数据读写）与仿真计算。

### 3.2 两阶段训练入口

- RL 训练：`navsim/planning/script/run_training.py agent=diffusiondrivev2_rl_agent ...`
- selector 训练：`navsim/planning/script/run_training.py agent=diffusiondrivev2_sel_agent ...`

### 3.3 快速评估

- `navsim/planning/script/run_pdm_score_fast.py agent=diffusiondrivev2_sel_agent ...`

完整命令参考官方文档：`docs/train_eval.md`。

---

## 4. Agent 层到底在干啥：`Diffusiondrivev2_Rl_Agent` 的“工程职责”

文件：`navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_agent.py`

这层基本不承载算法创新，主要负责“将模型接入 NAVSIM 训练体系”的工程封装（glue code）：

- 初始化时构建 `TransfuserModel`（实际是 `V2TransfuserModel`）。
- RL 阶段通常 **冻结 backbone**，仅训练 `_trajectory_head`（提升优化稳定性，也更接近“仅微调规划头”的设定）。
- `forward()` 传入 `eta=1.0`：训练期更随机、更利于 exploration（也更符合“policy”语义）。
- `compute_loss()` 只负责把模型返回的 loss/reward/subreward 组织成日志。

一句话：**Agent 负责训练循环对接；核心逻辑都在 model / trajectory head。**

---

## 5. Truncated diffusion 生成器：`TrajectoryHead` 里的关键结构

文件：`navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py`

### 5.1 `V2TransfuserModel.forward()` 的骨架

典型结构是：

1) 读取 `features`（camera/lidar/status）。
2) backbone 得到 BEV（Bird’s-Eye View，俯视栅格）表征。
3) transformer decoder 做 cross-attention（query：轨迹/agent；kv：BEV/status）。
4) `self._trajectory_head(...)` 生成轨迹候选及（训练时）RL 相关统计量。

### 5.2 `plan_anchor`：多模态的“语义坐标系”

在 `TrajectoryHead.__init__()` 通常能看到：

- `plan_anchor = np.load(plan_anchor_path)`
- `self.plan_anchor = nn.Parameter(torch.tensor(plan_anchor), requires_grad=False)`

工程含义：

- 你可以把每个 anchor 理解成一个 coarse intention（左转/右转/直行/变道/让行…）。
- diffusion 并不是从“纯随机”出发，而是围绕 anchor 子空间做 denoise，这样更可控、更稳定。

---

## 6. 关键创新 ①：乘性探索噪声 + 可用的 logprob

如果只从工程角度看，`DDIMScheduler_with_logprob.step()` 做了两件决定 RL 能不能跑起来的关键事：

### 6.1 让采样步骤可微：返回 `log_prob`

原版 diffusers 的 `DDIMScheduler.step()` 更偏“推理工具”，不会把每步的概率项显式输出。

但 RL 需要 $\log \pi_\theta$，所以这里改成返回：

- `prev_sample`
- `prev_sample_mean`
- `log_prob`

### 6.2 探索噪声更接近“策略扰动”，而非“逐点抖动”

论文强调：轨迹 gaussian additive noise 会破坏几何结构。

代码里通常实现成“乘性为主 + 少量加性”的形式：

$$x_{t-1} = \mu_{t} \odot \epsilon_{mul} + \sigma_{t} \cdot \epsilon_{add}$$

直觉：

- 乘性噪声更接近整体尺度/偏置的变化，使生成轨迹更平滑，更符合驾驶风格的连续变化。
- 代码里一般还会给 std 下限（例如 `min=0.04` 一类的 clamp），避免探索熵塌缩。

---

## 7. 关键创新 ②：Intra-anchor GRPO（组内相对优势，避免跨意图坍缩）

DiffusionDriveV2 的一个关键工程策略是：

> **不同 anchor 表征不同意图，advantage normalization 更适合在同一意图簇内进行，避免跨意图直接混合归一化。**

实现上通常是把 reward reshape 成：

- `reward_group`: `(B, G, K)`
  - `K`：anchor 数（常见 20）
  - `G`：每个 anchor 内采样的变体数（组内多样性）

然后只在 `G` 维度做归一化，得到组内优势：

$$
A_{k,i} = \frac{r_{k,i} - \mathrm{mean}(r_{k,1..G})}{\mathrm{std}(r_{k,1..G}) + \epsilon}
$$

这相当于：在同一意图簇内进行相对比较与学习，而不同意图之间避免进行强行的全局排序。

代码层面常见还会叠一个 per-step discount（降低早期高噪声 step 的梯度权重）。

---

## 8. 关键创新 ③：Inter-anchor truncated GRPO（跨意图共享硬约束）

只做 intra-anchor 的风险是：某个意图组内的“最优”可能仍然很差。

DiffusionDriveV2 的折中是：

- 跨意图 **不做谁更优的 rank**（避免 mode collapse）。
- 但是跨意图 **共享绝对失败的约束**（例如 collision/越界）。

实现上通常会看到对优势做截断：

- 负优势置零（不惩罚“比均值差一点”的样本，减少坍缩压力）。
- 硬失败（例如碰撞）直接给强惩罚。

抽象成公式就是类似：

$$
A^{\text{trunc}} = \begin{cases}
-1, & \text{collision (hard fail)}\\
\max(0, A), & \text{otherwise}
\end{cases}
$$

---

## 9. RL loss：优势是怎么落到 diffusion 链路 logprob 上的？

这里可以理解为 PPO（Proximal Policy Optimization，近端策略优化）/GRPO 的简化实现：

1) rollout 阶段（`forward_train_rl()`）：采样 diffusion chain，记录每一步 logprob 与生成轨迹。
2) 更新阶段（`get_rlloss()`）：用当前参数重新计算 logprob，结合优势做 policy loss。

DiffusionDriveV2 通常不引入显式 value network（GRPO 的一个工程优势是减少对 critic 的依赖），并会混入一个 IL loss 作为稳定项：

$$
L = L_{RL} + \lambda L_{IL}
$$

当 batch 里正优势样本很少时，IL 权重会更大（避免训练崩）。

---

## 10. Mode selector：为什么它在实现上这么“重”？

在端到端（E2E, End-to-End）planner 中，selector 并非可选模块，而是承担最终决策的关键组件。

代码层面对应：

- `ScorerTransformerDecoder` / `ScorerTransformerDecoderLayer`：把轨迹 token 当 query，与 BEV/agent 语义做 cross-attention。
- coarse-to-fine：先粗排再精排。
- 常见：BCE（Binary Cross-Entropy，二元交叉熵）分类 + ranking / margin loss 组合。

工程层面的直观理解：

- 生成器越可靠，selector 越偏向“选择最优候选”，而非“过滤大量低质量样本”。
- 训练上分两阶段（先 RL 抬生成器下界，再训 selector）也是为了让学习目标更分离。

---

## 11. 阅读代码最容易迷路的 5 个点（以及更省时间的追法）

1) `G / num_groups / ego_fut_mode(K)`：
   - `K` 多数就是 anchor 数（常见 20）。
   - `G` 是每个 anchor 内采样变体数。
2) 为什么推理只做 2 步 diffusion：truncated diffusion，以推理速度换取采样开销（训练阶段通过 RL/selector 等机制补偿质量）。
3) logprob 从哪来：靠 scheduler 改造 + 训练期 stochastic（`eta=1`）。
4) PDM reward 是不是在线交互：不是，更多是 NAVSIM 的离线闭环 proxy/并行 scorer。
5) 为什么经常出现 “GT vs proposals” 的 pairwise 计算：更省算力，也更明确“候选相对 GT 的好坏”。

---

## 12. DiffusionDriveV2 vs DIVER（arXiv:2507.04049）：同样是“RL + Diffusion”，差异在哪里？

这部分我建议用“研究动机 → 机制设计 → 训练信号 → 评价指标”四个维度去对齐，因为它们其实在解决两个相邻但不完全相同的问题。

### 12.1 共同点：都在对抗 IL 的单 GT 瓶颈

DIVER 的论文开宗明义指出：多数端到端自动驾驶（E2E-AD, End-to-End Autonomous Driving）规划依赖单专家轨迹做监督，会导致保守、同质化、以及多模态的 mode collapse。

DiffusionDriveV2 也是在这个问题域里，但它更偏“让 DiffusionDrive 的多模态候选可用”。

### 12.2 核心机制对齐表

| 维度 | DiffusionDriveV2 | DIVER (2507.04049) |
|---|---|---|
| 多模态来源 | anchor / intention prior + truncated diffusion（强调推理两步、高吞吐） | Policy-Aware Diffusion Generator（PADG）：条件包含 map+agents，显式生成多条轨迹并引入 **Reference GTs** |
| “单 GT”怎么破 | 仍以 GT 为重要参考，但通过 RL + exploration 抬多模态质量下界 | 从单 GT 派生 **multiple reference trajectories**，并用 Hungarian matching 让每个 mode 对齐不同 reference，直接缓解“全围着 GT 收敛” |
| RL 用来优化什么 | 在 diffusion 采样链上做 policy gradient：用优势约束碰撞/可行驶区域/效率等；intra-anchor 归一化避免跨意图坍缩 | 把 diffusion 当政策，用 GRPO 优化 **diversity reward + safety reward**，显式把“多样性”写进 reward（论文式定义 pairwise distance） |
| 关键稳定性技巧 | intra-anchor advantage + inter-anchor truncated（只传播硬失败）+ IL 稳定项 | hybrid IL+RL：$L_{total}=\lambda_{match}L_{match}+\lambda_{RL}L_{RL}$；用 matching loss 强化模式分配 |
| 评价指标取向 | NAVSIM/PDM/PDMS 等更偏闭环质量；多样性更多通过多模态候选/selector体现 | 明确指出 L2 不适合多模态，提出 **Diversity Metric**（归一化 pairwise dispersion，范围 [0,1]） |

### 12.3 DIVER 更“理论化/指标化”的点：Reference GT + Diversity Metric

从 DIVER 的论文表述来看，它有两个非常“强论文信号”的设计：

1) **Reference GTs + Hungarian matching loss**：
   - 用多个 reference GT 引导不同 mode 对齐不同“驾驶风格/意图”。
   - 训练目标不再把所有 mode 都拉向同一个 GT。
2) **Diversity Metric**：
   - 显式反对 L2 open-loop。
   - 使用归一化的轨迹分散度作为衡量多样性的指标（并且设计成 [0,1] 可比）。

### 12.4 DiffusionDriveV2 更“工程化”的点：两步采样 + selector + 可落地的训练配方

DiffusionDriveV2 的优势更偏向工程实践总结：

- truncated diffusion：推理成本极低，适配在线规划延迟。
- 把 RL 作用点放在“采样链路的可控改造”（logprob + 噪声设计 + 优势截断），避免训练不稳定。
- selector 两阶段：把“生成”和“决策”解耦，利于工程迭代。

如果你要从工程实现角度选路线：

- 想要**快 + 工程可控**：DiffusionDriveV2 的“anchor + truncated diffusion + RL 抬下界 + selector”更直接。
- 想要**把多样性定义/评价做得更严格**：DIVER 的 reference GT + diversity metric 更体系化。

---

## 13. 结尾总结：你应该带走的 3 个代码级要点

1) 多模态的“坐标系”在 `plan_anchor`，truncated diffusion 决定了推理成本。
2) RL 能不能跑起来，关键是 `DDIMScheduler_with_logprob.step()` 这类工程改造：必须拿到 logprob，且探索噪声要“像驾驶”。
3) 为缓解模式坍缩，优势设计需要尊重意图结构：intra-anchor 归一化 + inter-anchor 的硬失败共享，是一种兼顾稳定性与多样性的折中方案。

---

（完）
