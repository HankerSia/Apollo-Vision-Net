# SparseDriveV2 技术解析：因子化词表如何把静态打分路线推到新上限

> 面向读者：希望把论文读成可落地的工程方案，并进一步理解 SparseDriveV2 的核心设计为什么有效、实现上应如何理解，以及它与静态词表和动态生成两类路线到底差在哪里。
>
> 参考：
> - 论文（arXiv）：https://arxiv.org/html/2603.29163v1
> - 官方代码仓库：https://github.com/swc-17/SparseDriveV2
> - 仓库说明里的训练与评估文档：https://github.com/swc-17/SparseDriveV2/blob/main/docs/train_eval.md
> - 论文中展示的结构图与结果图：仓库 `assets/` 目录中的示意图
>
> 备注：本文聚焦 SparseDriveV2 的方法逻辑与工程实现思路，尽量按“问题定义 → 方法设计 → 工程落点 → 实验结论”的顺序展开，而不是逐行复述公式。

---

## 0. 导读：为什么 SparseDriveV2 值得认真看

在最近一波端到端自动驾驶规划工作里，SparseDriveV2 最值得关注的地方，不在于它提出了一套更复杂的生成范式，而在于它沿着一条看似更朴素的 scoring-based 路线，做出了足够有说服力的领先结果。

如果先只看结果，它的重要性已经非常明确：

1. 在 **NAVSIM v1** 上，SparseDriveV2 以 **92.0 PDMS** 超过了同为 ResNet-34 backbone 的 DriveSuprim、DiffusionDriveV2 和 ipad。
2. 在 **NAVSIM v2** 上，它给出了 **90.1 EPDMS**，说明这套方法在更严格的评测环境下依然成立。
3. 在 **Bench2Drive** 上，它进一步拿到 **89.15 Driving Score** 和 **70.00 Success Rate**，把 open-loop 的提升延续到了闭环场景。

换句话说，它超越的是此前一批很有代表性的 SOTA 路线，包括 **DriveSuprim、Hydra-MDP++、DiffusionDriveV2、ipad**，以及同阶段的 **GoalFlow、ARTEMIS、FUMP、DIVER** 等方法。

更关键的是，它并不是靠更重的生成器、更多的采样步数，或者更复杂的训练技巧取得领先；它的核心优势恰恰来自一条很多人以为已经接近天花板的路线：**先把可选轨迹覆盖得更细，再把评分过程做得更轻。** 具体来说，就是让候选空间更密、但不把所有候选都逐个重算，而是先粗筛、再精排。

这也是 SparseDriveV2 最值得认真读的原因：它不是在既有框架上做小修小补，而是在重新定义 scoring-based planning 的上限。

如果把论文的贡献压缩成最关键的三点，其实就是：

1. 它先用一组 scaling study 证明，scoring-based planning 的瓶颈主要来自词表密度和计算预算，而不是范式本身。
2. 它提出了 path / velocity 的因子化词表，把候选空间从“有限轨迹集合”扩展成“可组合的超密轨迹空间”。
3. 它设计了 coarse-to-fine 的可扩展打分策略，让超密候选集在训练和推理时都仍然可算。

---

## 1. 核心判断：SparseDriveV2 到底在解决什么问题

SparseDriveV2 面对的是端到端自动驾驶里一个非常现实的问题：**多模态规划并不缺候选，真正稀缺的是既覆盖得足够密、又能评分得足够快、同时还保留足够高质量的候选集合。**

论文开头的 scaling study 其实已经把问题说透了：当静态 trajectory vocabulary 继续变密时，性能会持续上升，而且在计算瓶颈出现之前并没有明显饱和。这说明一个非常重要的判断：

> 以前很多 scoring-based planner 不是“静态词表不行”，而是“词表还不够密，且打分方式还不够省”。

基于这个判断，SparseDriveV2 给出的答案不是“把模型继续做复杂”，而是把下面两件事同时做到位：

1. **把轨迹词表做成因子化结构**：把 trajectory 分解为 geometric path 和 velocity profile，两者组合出超密候选集。
2. **把打分做成分层过滤**：先对 path 和 velocity 做粗粒度评分，再对少量组合候选做细粒度评分。

所以它的核心，并不是再发明一个更复杂的生成器，而是把 scoring-based planning 推到新的密度上限：**先用更轻的结构覆盖更大的动作空间，再用更高效的筛选把这些候选真正转化成分数优势。**

---

## 2. 方法总览：从“单一轨迹”到“路径 + 速度”的组合空间

SparseDriveV2 的标题已经把它的立场写得很明确：`Scoring is All You Need`。它并不是要否定动态生成路线，而是想证明另一件事：**只要静态词表足够密、打分足够高效，纯 scoring 路线依然可以做到很强。**

![SparseDriveV2 overall architecture](https://raw.githubusercontent.com/swc-17/SparseDriveV2/main/assets/overview.png)

> 图 1：SparseDriveV2 的整体框架。核心思路是先把轨迹拆成 path 和 velocity，再分别做粗粒度打分，最后只对少量组合候选做精排。

整篇论文的主线可以概括成四步：

1. **问题重述**：自动驾驶规划最终是从候选轨迹里选一个最优解。
2. **瓶颈识别**：单一 monolithic trajectory vocabulary 很快会遇到计算和内存上限。
3. **结构重写**：把轨迹拆成 path 和 velocity 两个因子。
4. **高效打分**：先粗筛，再精排，避免在全量组合上做重计算。

如果把一条轨迹记作

$$
\tau = \{(x_t, y_t)\}_{t=1}^{T},
$$

SparseDriveV2 的关键，不是直接对这个序列建一个巨大的词表，而是把它分解成：

$$
\tau \leftrightarrow (p, v),
$$

其中：

1. $p$ 表示 geometric path，描述“往哪里走”。
2. $v$ 表示 velocity profile，描述“怎么快慢地走”。

这个拆法的价值在于，它把一个难以直接枚举的 spatiotemporal 空间，转成了两个更容易离散化、也更容易组合的子空间。

从论文视角看，这里其实还隐含着一个更强的判断：**轨迹空间的稠密覆盖，不一定要靠更复杂的生成过程，也可以靠更合理的结构化表示来获得。** SparseDriveV2 的核心创新，本质上就是把这一判断做成了可训练、可评估、可部署的工程方案。

---

## 3. 仓库阅读路径：代码应该从哪里开始看

从官方仓库 README 的结构看，SparseDriveV2 的工程分层比较清晰，主要可以从这几块进入：

1. `navsim/`：核心规划实现所在，承载模型、训练逻辑、评估接口和任务封装。
2. `scripts/`：训练、数据缓存、评估、可视化等入口脚本。
3. `docs/train_eval.md`：实际跑实验最关键的说明文档。
4. `assets/`：论文里提到的结构图、实验图、可视化图。
5. `download/`：下载 checkpoint 或相关资源的地方。

如果你是第一次看这类仓库，建议优先按这个顺序理解：

1. 先看 README，知道模型目标和结果上限。
2. 再看 `docs/train_eval.md`，知道训练/评估流程。
3. 然后再下钻 `navsim/`，把“候选构造”和“打分流程”对应到代码职责。

如果用一句话概括它的代码阅读路径，那就是：**先看候选怎么构造，再看候选怎么被筛选。** 这和论文的组织方式是高度一致的。

---

## 4. 核心设计一：因子化词表到底解决了什么

传统 scoring-based planner 的做法通常是：先构建一个固定 trajectory vocabulary，再对所有候选逐个打分。问题在于，词表一旦继续变密，计算和显存压力都会迅速膨胀。

SparseDriveV2 的做法是把 trajectory factorize 成两个维度：

$$
p = \{(x_i, y_i)\}_{i=1}^{S}, \qquad v = \{v_t\}_{t=1}^{T}
$$

这里：

1. Path 只保留空间几何，不关心每个时刻到底多快。
2. Velocity 只保留时间上的速度演化，不关心轨迹几何长什么样。

这个设计的工程收益非常直接：

1. path vocabulary 可以更专注于空间覆盖，比如直行、左转、右转、弯道贴线、换道等几何模式。
2. velocity vocabulary 可以更专注于纵向控制，比如慢速跟车、稳定巡航、加速通过、低速转弯等速度模式。
3. 两者组合后，候选轨迹数变成 $N_p \times N_v$，在词表结构不爆炸的情况下获得组合级的覆盖率。

可以把它理解成一种“把驾驶意图拆成两个正交轴”的做法：**空间形状负责意图，速度曲线负责节奏。**

从原论文的方法章节看，这里还有两层实现细节值得补上：

1. **从 trajectory 到 path / velocity 的转换不是抽象概念，而是明确的重采样过程。**
	path 是沿累计路程按固定空间间隔 $\Delta s$ 重新采样得到的几何路径；velocity 则是按固定时间间隔 $\Delta t$ 计算得到的速度序列。
2. **从 path / velocity 回到 trajectory 也有明确的组合算子。**
	论文先由速度序列累积得到行驶距离 $s_t$，再沿 path 在对应距离上插值，最终恢复时空轨迹。

换句话说，SparseDriveV2 不是简单把一条轨迹“拆开存一下”，而是定义了一套可逆的表示方式：

$$
(p, v) = \mathcal{D}(\tau), \qquad \tau = \mathcal{C}(p, v)
$$

这使得 path vocabulary 和 velocity vocabulary 不只是两个辅助特征集合，而是真正可以组合出最终规划轨迹的基础表示。

### 4.1 词表是怎么来的：不是手工规则，而是数据驱动构建

论文在 `Scalable Vocabulary Construction` 里把这一点讲得很清楚：SparseDriveV2 的 path vocabulary 和 velocity vocabulary 都来自大规模人类驾驶数据，而不是手工写出来的规则模板。

具体过程可以概括为：

1. 从训练集未来轨迹中提取足够长的 driving demonstrations。
2. 把这些轨迹先转换成 path 和 velocity 两种表示。
3. 分别对 path 和 velocity 做 K-Means 聚类，得到代表性的 path anchors 和 velocity anchors。
4. 最终通过全组合形成 trajectory vocabulary：

$$
\mathcal{T} = \{ \mathcal{C}(p_i, v_j) \mid p_i \in \mathcal{P}, v_j \in \mathcal{V} \}
$$

这一步非常关键，因为它解释了 SparseDriveV2 为什么能在不引入动态生成的前提下，把词表密度推到论文里强调的 **32x** 水平。对于 NAVSIM v1，论文给出的典型配置是：

1. $N_p = 1024$ 条 path anchors。
2. $N_v = 256$ 条 velocity anchors。
3. 最终组合得到 $1024 \times 256 = 262{,}144$ 条 trajectory anchors。

相较于很多 prior scoring-based 方法常用的 8192 条 monolithic trajectory anchors，这个组合空间的密度提升不是边际优化，而是量级上的变化。

---

## 5. 核心设计二：为什么 path 和 velocity 要分开打分

如果直接对所有组合轨迹做打分，模型当然会更“直接”，但代价也会迅速变得不可接受。SparseDriveV2 的关键，就是把这件昂贵的事情拆成两层：

### 5.1 第一层：粗粒度 factorized scoring

先独立对 path 和 velocity 打分：

$$
s^p_i = f_p(E_p(p_i), F, E), \qquad s^v_j = f_v(E_v(v_j), F, E)
$$

这里：

1. $F$ 是 scene feature。
2. $E$ 是 status feature。
3. $E_p$ 和 $E_v$ 是对 path / velocity 的编码器。

直观上，这一层在做两件事：

1. path scoring 过滤掉明显不适合当前场景的空间形状。
2. velocity scoring 过滤掉明显不合理的速度模式。

例如：

1. 在拥堵场景里，高速 velocity profile 往往不合理。
2. 在直行道路上，强转弯 path 往往不合理。

它的价值在于，大量低质量组合会被提前排除在“真正组成轨迹之前”。

### 5.2 第二层：fine-grained trajectory scoring

在粗筛后，再对少量 composed trajectories 做精排。对应形式上就是：

$$
\tau_{i,j} = \mathcal{C}(p_i, v_j)
$$

然后对组合后的轨迹再进行一次 re-conditioning / interaction / scoring。

这一步的意义，不是重复前一步，而是补上一个在驾驶场景里非常关键的事实：**path 和 velocity 并不总是独立的。**

比如：

1. 一个急转弯 path，通常不能配非常高的速度。
2. 一个平直 path，通常可以容纳更高的速度上限。

所以粗粒度 scoring 负责“筛掉大多数不可能”，细粒度 scoring 负责“在剩下的候选里做最终排序和决策”。

### 5.3 论文里还有一个关键点：scene encoder 并不依赖显式 BEV 构建

这一点在博客里也值得单独强调。论文的 `Scene Encoding` 采用了和 SparseDrive 一脉相承的思路：直接从多视角图像中提取 scene features，再结合 ego status 做交互，而不是先显式构建一个厚重的 BEV 表示再去做规划。

这意味着 SparseDriveV2 的效率优势不只来自“词表结构更聪明”，也来自“前端表征没有把计算预算过早耗尽”。在方法上，它把大部分算力留给了真正影响规划质量的候选筛选和精排阶段。

### 5.4 Top-K 不是抽象概念，而是有明确的两级规模设置

论文在实现细节里给出了非常具体的 progressive filtering 配置，这也是理解 SparseDriveV2 推理效率的关键：

1. 第一层 decoder 先保留 top-128 的 path anchors 和 top-64 的 velocity anchors。
2. 第二层再进一步收缩到 20 条 path 和 20 条 velocity。
3. 最终只对 $20 \times 20 = 400$ 条 trajectory hypotheses 做 fine-grained scoring。

这个数字很重要，因为它说明 SparseDriveV2 的推理优势不是“把 26 万条候选都算一遍”，而是先用很便宜的 coarse scoring 把大盘子收缩，再把真正昂贵的细粒度推理限制在 400 条候选上。对于 NAVSIM v2，论文还专门把第二层 velocity 数量从 20 缩到 10，以进一步降低 metric ground-truth 计算开销。

---

## 6. 图 1 解读：整体架构在表达什么

如果把论文的整体框架图当成代码流程图，它基本就是下面这个顺序：

1. **Scene encoding**：从多相机图像和 ego status 中抽取场景特征。
2. **Factorized vocabulary**：构建 path vocabulary 和 velocity vocabulary。
3. **Coarse scoring**：分别给 path 和 velocity 打分。
4. **Top-K selection**：选出少量高质量 path/velocity。
5. **Composition**：把它们组合成候选轨迹。
6. **Fine-grained scoring**：对组合轨迹进行二次打分。
7. **Final decision**：选出最终执行轨迹。

这张图最容易被忽略的一点是：**SparseDriveV2 的难点不是“候选数量大”，而是“候选数量大却不需要逐个全量重算”。**

它把最重的 spatiotemporal reasoning 放在了最后的少量候选上，而不是一开始就对全量组合做昂贵交互。

---

## 7. 训练目标：为什么它不是一个单纯的分类问题

SparseDriveV2 的训练方式，本质上是一个分层监督系统。它的目标不是做单点回归，也不是只做一次分类，而是把多个阶段都对齐到最终规划质量上。

论文里最重要的监督项可以概括为三类：

### 7.1 Path-level supervision

对 ground-truth path 和 path anchors 做距离监督，然后构造 soft target distribution，再用 cross-entropy 训练 path scorer。

### 7.2 Velocity-level supervision

对 ground-truth velocity profile 和 velocity anchors 做距离监督，再训练 velocity scorer。

### 7.3 Trajectory-level supervision

对于组合后的 trajectory anchors，再做 trajectory-level imitation learning / metric supervision。

如果写成总目标，大致可以理解成：

$$
\mathcal{L} = \mathcal{L}_{path} + \mathcal{L}_{vel} + \mathcal{L}_{traj} + \alpha \mathcal{L}_{metric}
$$

其中：

1. $\mathcal{L}_{path}$ 负责几何覆盖。
2. $\mathcal{L}_{vel}$ 负责纵向节奏。
3. $\mathcal{L}_{traj}$ 负责最终轨迹选择。
4. $\mathcal{L}_{metric}$ 负责把闭环指标、规则约束或 teacher signal 引入训练。

这类分层损失的好处很明确：**你不是只告诉模型“最终那条轨迹对”，而是在逐层告诉它“路径怎么选、速度怎么选，以及组合后怎么选”。**

如果对照原文，这里还有两个细节值得补充：

1. coarse path scoring 用的是 masked average squared distance 构造 soft target。
2. velocity scoring 用的是 velocity profile 的 $L_1$ 距离构造 soft target。

也就是说，SparseDriveV2 并不是把词表学习做成一个硬标签分类问题，而是把“离 GT 更近的 anchor 应该拿更高概率”这种结构化偏好显式写进了监督信号里。这样做的直接好处是，模型学到的不只是“哪一个是对的”，而是“哪些更接近正确答案”。

---

## 8. 工程视角：为什么它比“直接扩大静态词表”更聪明

如果只是简单把 monolithic trajectory vocabulary 扩大，问题会很快暴露：

1. 词表越大，显存压力越高。
2. 逐候选打分越慢。
3. 模型越容易在训练和推理时都变得笨重。

SparseDriveV2 的 factorization 让词表密度显著提升，但并不需要按同样比例增加所有计算。

可以把这个思路记成一句话：

> **不是把每条轨迹都单独学一遍，而是先学“空间形状”和“速度节奏”两个基元，再组合。**

这和语言模型里“词表变大”和“子词分解”的思路很像，只是这里换成了轨迹空间。

更重要的是，因子化之后，模型更容易学习到跨场景共享的结构：

1. 路径几何可以跨速度复用。
2. 速度模式可以跨路径复用。
3. 高质量候选不需要在同一个大表里重复编码。

这也是 SparseDriveV2 能在不引入复杂生成器的前提下，依然把性能推高的根本原因。

如果把它翻译成一句工程判断：**SparseDriveV2 的强，不在于某个单点模块特别“花哨”，而在于它把表示、筛选和监督这三件事放在同一套计算预算下协同优化了。**

---

## 9. 实验结果：为什么它的领先更有说服力

SparseDriveV2 的结果之所以有说服力，不是因为它在某一个数据集上多涨了零点几分，而是因为它在 **NAVSIM v1、NAVSIM v2、Bench2Drive** 这几类代表性 benchmark 上，都给出了同一方向的结论：**scoring-based planning 仍然有明显的性能上限可以继续往上推。**

下面用论文 Table 2 的完整 NAVSIM v1 结果来对照，这样可以更完整地看出 SparseDriveV2 和不同路线之间的差异：

| Method | Img. Backbone | NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VADv2 [vadv2] | ResNet-34 | 97.2 | 89.1 | 91.6 | 100 | 76.0 | 80.9 |
| UniAD [uniad] | ResNet-34 | 97.8 | 91.9 | 92.9 | 100 | 78.8 | 83.4 |
| Transfuser [transfuser] | ResNet-34 | 97.7 | 92.8 | 92.8 | 100 | 79.2 | 84.0 |
| PARA-Drive [paradrive] | ResNet-34 | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| DRAMA [drama] | ResNet-34 | 98.0 | 93.1 | 94.8 | 100 | 80.1 | 85.5 |
| GoalFlow [goalflow] | ResNet-34 | 98.3 | 93.8 | 94.3 | 100 | 79.8 | 85.7 |
| Hydra-MDP [hydramdp] | ResNet-34 | 98.3 | 96.0 | 94.6 | 100 | 78.7 | 86.5 |
| Hydra-MDP++ [hydramdp++] | ResNet-34 | 97.6 | 96.0 | 93.1 | 100 | 80.4 | 86.6 |
| ARTEMIS [artemis] | ResNet-34 | 98.3 | 95.1 | 94.3 | 100 | 81.4 | 87.0 |
| FUMP [fump] | ResNet-34 | 98.1 | 96.2 | 94.2 | 100 | 82.0 | 87.8 |
| DiffusionDrive [diffusiondrive] | ResNet-34 | 98.2 | 96.2 | 94.7 | 100 | 82.2 | 88.1 |
| WoTE [wote] | ResNet-34 | 98.5 | 96.8 | 94.9 | 99.9 | 81.9 | 88.3 |
| DIVER [diver] | ResNet-34 | 98.5 | 96.5 | 94.9 | 100 | 82.6 | 88.3 |
| DriveSuprim [drivesuprim] | ResNet-34 | 97.8 | 97.3 | 93.6 | 100 | 86.7 | 89.9 |
| DiffusionDriveV2 [diffusiondrivev2] | ResNet-34 | 98.3 | 97.9 | 94.8 | 99.9 | 87.5 | 91.2 |
| ipad [ipad] | ResNet-34 | 98.6 | 98.3 | 94.9 | 100 | 88.0 | 91.7 |
| Hydra-MDP [hydramdp] | V2-99 | 98.4 | 97.8 | 93.9 | 100 | 86.5 | 90.3 |
| GoalFlow [goalflow] | V2-99 | 98.4 | 98.3 | 94.6 | 100 | 85.0 | 90.3 |
| SparseDriveV2 | ResNet-34 | 98.5 | 98.4 | 95.0 | 99.9 | 88.6 | **92.0** |

> 表 2：论文中的 NAVSIM v1 leaderboard。为避免 markdown 渲染问题，这里去掉了原表的底色命令，仅保留数值；SparseDriveV2 的 PDMS 用加粗表示。

如果把论文 Table 2 里的信息再多看一层，还能发现 SparseDriveV2 的领先并不是靠某一个子指标“单点拉高”。它在 NC、DAC、TTC、Comfort 和 EP 这些组成 PDMS 的核心维度上都表现稳定，尤其是 **EP（Ego Progress）** 指标更高，这和论文想强调的“更密的动作空间覆盖”是完全一致的。

### 9.1 NAVSIM v1

论文报告了：

1. 92.0 PDMS。
2. 90.1 EPDMS。

在 v1 上，这意味着它不仅领先于典型 static scoring 方法，也超过了当下更受关注的 dynamic generation 和 hybrid 路线。

### 9.2 NAVSIM v2

在更严格的 v2 协议下，它仍然能拿到很强的 EPDMS，说明 factorized vocabulary + scalable scoring 并不是只对旧协议有效，而是对更复杂的评价体系也成立。

如果对照原论文 Table 3，SparseDriveV2 在更新后的官方实现下达到 **90.1 EPDMS**，相比文中的 DiffusionDriveV2 **87.5 EPDMS** 高出 **2.6 分**。原文还专门指出，EP 指标的显著提升正是更高词表覆盖带来的直接收益。

### 9.3 Bench2Drive

在闭环驾驶里，论文给出了：

1. 89.15 Driving Score。
2. 70.00 Success Rate。

如果只写这两个数字还不够完整，Bench2Drive 部分其实还体现了另一层信息：SparseDriveV2 不只是“能跑起来”，而是在 closed-loop benchmark 上展现了明显的泛化能力。论文还给出了 multi-ability 结果，其 mean score 达到 **67.67**，说明它在 merging、overtaking、emergency brake、give way 和 traffic sign 等不同交互能力上都不只是偶然命中。

为了让这部分更完整，下面补一张 Bench2Drive 的简化对比表：

| Method | Driving Score | Success Rate | Mean Ability |
| --- | --- | --- | --- |
| Hydra-NeXt | 73.86 | 50.00 | 53.22 |
| SimLingo | 86.02 | 67.27 | - |
| HiP-AD | 86.77 | 69.09 | 65.98 |
| SparseDriveV2 | 89.15 | 70.00 | 67.67 |

这说明它的“候选足够密 + 打分足够省”的设计，不只是 open-loop 指标漂亮，而是真的能够转化成闭环性能。

对工程读者来说，这组结果最有价值的地方在于：**它证明了纯 scoring 路线并不天然落后于动态生成路线，关键差别在于词表覆盖和层次化打分有没有真正做到位。**

### 9.4 论文里不可忽略的一部分：消融实验也支持主结论

当前博客如果只写主结果，其实还少了一块很重要的证据链：论文的 ablation studies。

Table 6 主要说明了两件事：

1. **词表规模继续增大，性能就继续提升。**
	例如从 $(N_p, N_v) = (512, 128)$ 到 $(1024, 256)$，EPDMS 从 88.7 提升到 90.1，没有出现明显饱和。
2. **scalable scoring 里的具体实现选择同样重要。**
	把 path-scene interaction 从普通 MHA 换成 deformable aggregation 会更好；再加入 trajectory re-conditioning，又会进一步提升结果。

这组消融的价值在于，它证明 SparseDriveV2 的提升并不是单纯因为“词表大了”，而是“更大的词表”和“更合适的打分机制”是一起工作的。

### 9.5 定性结果和失败案例也值得补一句

论文附录里的定性图其实也很有代表性，至少能支持三条观察：

1. 在 sharp-turn 场景里，SparseDriveV2 的轨迹更平滑。
2. 在效率场景里，它比 baseline 更不容易无谓停车。
3. 在高层意图对齐上，它更接近 expert trajectory。

同时，论文并没有回避失败案例。附录中的 Figure 5 明确指出，在某些场景里 SparseDriveV2 仍然会做出错误的导航决策，作者给出的解释是 **navigation information 可能不足**。这一点很重要，因为它说明 SparseDriveV2 的瓶颈已经不完全是“候选空间不够密”，而开始转向更上层的任务条件和语义信息。

---

## 10. 落地路径：训练和评估入口该怎么找

从仓库 README 和 `docs/train_eval.md` 来看，实际跑实验时你主要关心三件事：

1. 环境准备。
2. 数据缓存。
3. 训练与评估脚本。

如果按仓库结构来理解，推荐这样看：

### 10.1 数据缓存

先做缓存的目的很直接：把不需要每次重复计算的内容预先算好，避免训练时把时间浪费在 I/O 和重复特征抽取上。

### 10.2 训练入口

训练脚本统一放在 `scripts/` 下，README 里明确给了 `docs/train_eval.md` 作为主入口说明。

### 10.3 评估入口

评估同样从脚本层启动，再根据 NAVSIM 或 Bench2Drive 的协议输出最终指标。

对这类项目来说，更高效的读法并不是“先扎进最底层模型代码”，而是先把 **数据准备、训练入口、评估协议** 这三件事串起来。SparseDriveV2 的仓库已经把这条链路尽量收敛到 README、docs 和 scripts 三个部分。

如果进一步对照原文和仓库说明，NAVSIM 的典型实现配置也值得在博客里留一笔：

1. path anchors 采样间隔为 1m，最大空间 horizon 为 50m。
2. velocity profile 的时间间隔为 0.5s，规划 horizon 为 4s。
3. scene feature 使用 ResNet-34，输入相机为 `l0 / f / r0`，分辨率 256×512。
4. NAVSIM 训练通常使用 8 张 NVIDIA L20，total batch size 128，训练 10 epochs，learning rate 为 $1 \times 10^{-4}$，weight decay 为 0。

这些数字不是无关紧要的实现细节，而是 SparseDriveV2 把“超密词表”和“可控算力”同时成立的关键约束条件。

---

## 11. 横向对比：如果和 DiffusionDriveV2 放在一起看，差异是什么

虽然两者都属于端到端自动驾驶里的多模态规划路线，但思想重心明显不同：

1. **DiffusionDriveV2** 更像是“生成式多模态规划”：重点是采样、扩散、RL、selector。
2. **SparseDriveV2** 更像是“极致的 scoring-based planning”：重点是词表结构、候选密度、层次化打分。

如果说 DiffusionDriveV2 关注的是“怎么生成出更好的候选”，那么 SparseDriveV2 关注的是“怎么把候选空间组织得更合理、把打分做得更高效”。

这两条路线并不是互相否定，而是两个不同工程约束下的优化方向：

1. 生成式路线更强调灵活性和样本质量提升。
2. Scoring 路线更强调可控性、稳定性和推理效率。

SparseDriveV2 的贡献在于，它把“静态词表”这条看似保守的路线，重新做成了一个既可以规模化、也可以逼近动态生成性能的强方案。

如果再往前推一步看，SparseDriveV2 真正有价值的地方是：它迫使整个领域重新回答一个基础问题: **高性能规划到底必须依赖动态生成，还是足够好的结构化评分已经足够？** 论文给出的答案非常明确，至少在当前 benchmark 上，后者依然大有空间。

---

## 12. 阅读提示：读代码时最容易迷路的几个点

如果你后面准备真的去翻仓库，最容易卡住的通常是这几个地方：

1. **为什么 path 和 velocity 要先独立监督**：因为它们对应不同层次的驾驶语义，先分开学更容易收敛。
2. **为什么要 top-k 两阶段筛选**：因为全量组合太贵，先粗排再精排是唯一合理的工程折中。
3. **为什么 trajectory re-conditioning 还要再看 scene**：因为 path/velocity 组合只提供先验，真实可行性仍然要结合场景再判断。
4. **为什么静态词表能赢动态生成方法**：因为一旦词表够密，动态生成的优势就不再是唯一解法，而高效打分和更强覆盖更关键。

可以把 SparseDriveV2 的实现理解成一个三段式系统：

1. 先把动作空间切开。
2. 再把大候选集过滤掉。
3. 最后只对少量高质量候选做高成本判别。

---

## 13. 总结：关于 SparseDriveV2，最值得带走的判断

SparseDriveV2 证明了一件很重要的事：**当候选空间足够密、结构足够合理、打分足够高效时，scoring-based planning 不仅没有过时，反而仍然可以做到领先。**

它不是靠更复杂的生成过程取胜，而是靠更好的词表结构和更聪明的候选筛选，把“静态候选集”做成了一个真正可扩展的规划系统。

如果你从工程角度看自动驾驶规划，这篇论文最值得记住的不是某个单独公式，而是下面这条经验：

> **要扩大规划能力，不一定先加大模型复杂度，也可以先重构动作空间。**

---

## 14. 后续阅读建议

如果你接下来还想继续深挖，建议按这个顺序：

1. 先把论文里的 Figure 1 和方法章节通读一遍，理解 path / velocity / trajectory 的三层关系。
2. 再看仓库 `README.md` 和 `docs/train_eval.md`，确认训练和评估链路。
3. 最后去对照 `navsim/` 里的实现，把 coarse scoring、top-k 选择和 fine-grained scoring 对上。

这样读下来，你会比“先看公式、再找代码”更容易建立完整心智模型。

---
## 15. 附录：NAVSIM v1 指标定义

上面第 9 章里的表格使用的是 NAVSIM v1 的 leaderboard 指标。为了方便对照，这里把 `NC ↑ | DAC ↑ | TTC ↑ | Comf. ↑ | EP ↑ | PDMS ↑` 逐一解释一下。

### 15.1 指标总览

| 缩写 | English name | 解释 | 含义 |
| --- | --- | --- | --- |
| NC | No Collision | 无碰撞率 | 表示规划轨迹是否避免与其他车辆、行人或静态障碍发生碰撞。数值越高越安全。 |
| DAC | Drivable Area Compliance | 可行驶区域符合率 | 表示车辆轨迹有多大比例保持在可行驶区域内，避免压线、越界或驶出车道。数值越高越规范。 |
| TTC | Time-To-Collision | 碰撞时间安全性 | 反映轨迹在前向安全距离和碰撞时间上的表现，数值越高表示越不容易进入危险接近状态。 |
| Comf. | Comfort | 舒适性 | 衡量轨迹是否平滑，通常和加速度、加加速度（jerk）等有关。数值越高表示驾驶动作越平顺。 |
| EP | Ego Progress | 自车进展 | 表示车辆沿任务路线向前推进的程度，越高说明规划越能有效完成路径进度。 |
| PDMS | PDM Score | 综合规划评分 | NAVSIM v1 的最终聚合分数，由安全、合规、舒适和进展等子指标组合而成。 |

### 15.2 PDMS 的公式

论文给出的 NAVSIM v1 PDMS 计算方式为：

$$
\mathrm{PDMS} = \mathrm{NC} \times \mathrm{DAC} \times \frac{5\mathrm{TTC} + 2\mathrm{C} + 5\mathrm{EP}}{12}
$$

这里：

1. `NC` 和 `DAC` 作为前置门控项，分别约束安全性和可行驶区域合规性。
2. `TTC`、`C`、`EP` 则共同构成后面的加权项，分别对应安全性、舒适性和任务进展。
3. 权重 `5 : 2 : 5` 表示 NAVSIM v1 更看重安全与进展，同时也保留舒适性的约束。

### 15.3 每个指标怎么看

1. **NC - No Collision**
	- 这个指标最直观，关注的是“有没有撞上”。
	- 在自动驾驶规划里，它是最基础的安全门槛。
	- 如果 NC 低，说明轨迹虽然可能前进更快，但安全性已经不够。

2. **DAC - Drivable Area Compliance**
	- 这个指标关注“有没有跑出可行驶区域”。
	- 它强调的是规则与车道约束，而不是单纯速度。
	- 对端到端规划来说，DAC 高通常意味着轨迹更像人类驾驶。

3. **TTC - Time-To-Collision**
	- TTC 本质上衡量“距离潜在碰撞还有多远的时间”。
	- 数值越高，说明系统在前向安全距离上更保守，也更不容易形成危险接近。
	- 在 leaderboard 里，它常被看作安全裕度的一部分。

4. **Comf. - Comfort**
	- 舒适性通常和轨迹的平滑程度有关。
	- 如果轨迹频繁急加速、急减速、急转向，Comfort 往往会下降。
	- 这也是为什么论文强调 path / velocity 的结构化建模：它有助于生成更平稳的动作曲线。

5. **EP - Ego Progress**
	- EP 表示自车沿任务目标前进的程度。
	- 这个指标不是“跑得越快越好”，而是看轨迹是否真正帮助车辆有效向前推进。
	- SparseDriveV2 的提升里，EP 往往是最能体现“词表更密”收益的部分。

6. **PDMS - PDM Score**
	- PDMS 是 NAVSIM v1 的最终综合得分。
	- 它把安全、合规、舒适和进展放在同一个评分体系里，因此比单一指标更接近真实驾驶质量。
	- 论文里 SparseDriveV2 的 92.0 PDMS，说明它不是只在某一个维度上好看，而是在整体规划质量上也比较均衡。

### 15.4 读表时可以怎么理解

如果把这组指标放在一起看，可以把它们理解成四层约束：

1. **先安全**：NC 和 TTC 先保证不出危险。
2. **再合规**：DAC 保证轨迹没有跑偏。
3. **再平顺**：Comfort 保证动作不突兀。
4. **最后看进展**：EP 衡量是否真的推动了任务完成。

这也是为什么 SparseDriveV2 的 table 里，某些方法单项分数很高，但最终 PDMS 仍然会拉开差距：leaderboard 看的是“能不能综合地开得好”，不是只看某一个局部指标。
