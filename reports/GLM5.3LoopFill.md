# GLM-5.4 下 LoopFill 自投机解码方案可行性分析

> 文件名沿用任务约定 `GLM5.3LoopFill.md`。本文讨论对象是 GLM-5.4 的 Loop Transformer，不把 GLM-5.3 的普通解码实现直接当作目标实现。

## 结论

**LoopFill 可以在 GLM-5.4 上实现，但它不是把现有 LoopFill 代码换一个模型配置就能得到的即插即用优化。** 更准确的判断是：GLM-5.4 的架构满足 LoopFill 的基本前提，适合做免训练的自投机原型；要获得可重复的端到端收益，必须同时改造候选树调度、循环状态缓存、稀疏注意力索引和分布式批处理。

建议按下面的边界推进：

- **研究原型可行性：高。** GLM-5.4 复用一组共享权重运行两次，天然存在 `T=2` 的浅层—完整深度层次，LoopFill 的后缀草稿和浅层草稿都可以接入。
- **精确贪心输出：可行，但有前提。** 最终 Token 必须由 GLM-5.4 的完整两次循环重新决定。任何候选树、第一轮 readout 或缓存复用都只能改变计算顺序，不能替代第二轮验证。
- **精确采样分布：可行，但实现难度更高。** 需要使用完整目标分布做接受和残差修正，不能把第一轮 logits 当作最终采样分布。
- **生产级收益：尚未能从公开资料推出。** `T=2` 比 LoopFill 论文中常见的三到四次循环更浅，收益更依赖后缀命中率、第一轮预测质量、节点预算和 GPU/通信开销。

一句话概括：**先把 LoopFill 当作 GLM-5.4 的调度层，而不是模型层改造。先证明状态和验证语义完全正确，再测量它是否值得承担额外的候选计算。**

## 1. GLM-5.4 的相关架构事实

Z.ai 公开说明 GLM-5.4 将 GLM-5 基座的中间 28 层共享并运行两次。为控制 750B 规模模型的部署成本，官方还引入了三项与 LoopFill 直接相关的设计。

1. **门控渐进更新。** 每个 Token、每次迭代都有独立门控。第一轮主要建立全局上下文，第二轮利用已有状态做细化和纠错，门控大小还会随 Token 难度变化。
2. **混合 KV Cache。** 第二轮复用第一轮选出的稀疏 KV，同时保留第二轮自己的局部滑窗 KV。额外循环只增加一个滑窗，而不是整段上下文的第二份 KV。
3. **稀疏注意力索引复用。** 第二轮直接复用第一轮生成的稀疏注意力索引，避免再次执行索引器。

这三个设计说明 GLM-5.4 的循环并非两个相互独立的普通 Transformer forward。第二轮状态依赖第一轮的选择、门控和缓存布局。LoopFill 的候选树因此必须保存“节点自身的循环进度”和“该节点可合法复用的索引与 KV”，不能只保存一个普通 Transformer 的 past-key-values 指针。

官方还指出，GLM-5.4 在 1B、4B、30B 和 750B 尺度上验证了循环结构，并在同参数量 30B 对照中使用约 10% 的额外训练计算换取数学、科学推理和长上下文任务收益。这个结果支持架构迁移的方向，但没有给出 GLM-5.4 与 LoopFill 组合后的实测数据。

## 2. LoopFill 的机制与 GLM-5.4 的对应关系

LoopFill 维护一棵统一候选树。候选来自两条路径：后缀树可以一次加入多个历史续写，浅层模型在节点达到深度 `d` 后用 LM head 产生 Top-k 候选。所有节点按路径分数排序，只保留每个请求的 Top-N 节点。不同节点在不同循环深度上同时前进，根节点完成完整深度后提交真实输出，并提升已验证的匹配子树。

对 GLM-5.4，最自然的映射是：

| LoopFill 抽象 | GLM-5.4 映射 | 需要注意的地方 |
|---|---|---|
| 完整循环深度 `T` | `T=2` | 28 层共享块执行两轮，不能按普通 106 层独立权重模型处理 |
| 浅层起草深度 `d` | 首轮结束，即 `d=1` | 首轮 logits 是草稿信号，不是最终分布 |
| 完整验证 | 第二轮结束后的目标 logits | 必须保留门控、稀疏索引和混合 KV 的原生语义 |
| 后缀候选 | 当前会话和历史已验证输出中的 Token 续写 | 全局树需要租户隔离、过期和隐私策略 |
| 节点状态 | 第一轮隐藏状态、两轮 KV、索引元数据、门控相关中间状态 | 需要 Copy-on-Write 和祖先共享 |
| Top-N | 每请求候选节点上限 | 还需要全局候选预算，避免 750B 分布式批次膨胀 |
| 提交路径 | 完整两轮 logits 接受的最长连续子树 | 只有已验证路径可以成为新根 |

因此，**模型结构的兼容性高，缓存和调度的兼容性是主要工作量。**

## 3. 推荐的 GLM-5.4 LoopFill 执行流程

设已验证前缀为 `X`，候选节点为 `v`，其 Token 为 `x_v`。对每个节点至少维护下面的状态：

```text
parent(v), token(v), depth(v)
H1(v)                 第一轮循环结束的隐藏状态
I1(v)                 第一轮产生的稀疏注意力索引或其可复用句柄
K1(v), V1(v)          第一轮可共享的稀疏 KV
K2win(v), V2win(v)    第二轮局部滑窗 KV
logits1(v), logits2(v)第一轮草稿 logits 与第二轮验证 logits
score(v), status(v)    路径分数和候选状态
```

每个全局调度步可以按以下顺序执行。

### 3.1 创建候选

1. 以根节点的已验证前缀查询局部后缀树和可选的全局后缀树。
2. 对命中的后缀续写加入连续 Token 路径，单次加入量受 `Q` 限制。
3. 当节点完成第一轮 28 层后，用首轮 LM head 产生 Top-k 候选。
4. 对首轮候选继续做后缀查询，把“历史续写”和“首轮预测”合并到同一棵树。相同父节点、相同 Token 的节点必须合并，避免重复计算。
5. 用路径分数和节点预算保留候选。初始实验建议 `N=2` 或 `N=4`，不要直接沿用小模型论文中的 `N=16`。

### 3.2 推进循环

- 根节点和已有候选节点可以在同一个批次中推进，但每个节点的循环进度不同。
- 新候选先完成第一轮，随后才能进入第二轮。
- 第二轮使用 GLM-5.4 的混合 KV 规则。共享前缀的第一轮稀疏 KV 和索引可以别名复用，分支新增 Token 的第二轮滑窗必须独立维护。
- 需要按循环深度分桶或使用带深度元数据的连续批次，否则不同节点的状态会被错误地拼接。

### 3.3 验证和提交

根节点完成第二轮后，用 `logits2(root)` 决定真实 Token。若真实 Token 匹配某个子节点，则：

1. 将该节点提升为新根。
2. 保留其已完成的第一轮和第二轮状态，以及后代候选。
3. 删除竞争分支及其独占缓存。
4. 如果后续连续节点也已完成第二轮验证，则在同一全局步连续提交多个 Token。

如果根节点输出不匹配任何候选，则只提交完整目标模型的结果，并从新前缀重新建立候选。这个回退路径是正确性的必要条件。

## 4. 最关键的技术风险

### 4.1 稀疏索引复用不能简单复制到所有分支

官方的索引复用描述针对同一个 Token 的第一轮和第二轮。LoopFill 会同时计算多个不同候选 Token，因此需要区分两种情况：

- **已验证公共前缀。** 第一轮索引、稀疏 KV 和相关句柄可以在所有后代之间共享。
- **分支新增位置。** 候选 Token 改变了局部隐藏状态，不能默认认为另一条分支的索引仍然等价。可以先保守地为分支重新计算索引，或者只复用公共前缀索引并为分支尾部追加索引。

如果为了省索引计算而把某一条分支的索引强行用于另一条分支，第二轮不再是 GLM-5.4 的目标模型计算。即使最终 Token 偶尔相同，也不能据此声称保持严格无损。第一版实现应优先选择保守策略，待测量证明后再扩大复用范围。

### 4.2 混合 KV Cache 需要树形 Copy-on-Write

GLM-5.4 的第二轮同时读取第一轮选出的长程 KV 和第二轮新产生的滑窗 KV。候选树中的兄弟节点必须共享公共祖先缓存，同时隔离各自新增的滑窗内容。适合的实现是页式缓存或块级 Copy-on-Write，而不是为每个候选复制一份完整上下文。

提交一个子树后，祖先页可以继续保留，竞争分支的私有页回收。回收逻辑必须与异步 GPU kernel 和分布式通信完成事件绑定，不能根据 CPU 侧的“候选已剪枝”状态立即释放仍在使用的页。

### 4.3 `d=1` 的首轮信号可能不够稳定

LoopFill 论文中的收益依赖浅层 readout 与完整模型有较高一致性。GLM-5.4 的门控会根据 Token 难度改变每轮更新幅度，固定 `d=1` 不能保证所有 Token 都有相同质量的草稿。

建议记录以下信号并用于自适应控制：

- 首轮 Top-1 与最终 Top-1 的一致率。
- 首轮 Top-k 覆盖最终 Token 的比例。
- 每个 Token 的第一轮门控值或其聚合统计。
- 后缀命中率、候选路径接受率和被丢弃节点的平均已执行循环数。

当首轮置信度低或门控显示 Token 难度高时，应缩小 `k`、暂停浅层扩展或只保留后缀候选。GLM-5.4 官方提到的自适应计算方向，也支持把 LoopFill 的候选预算做成 Token 级动态策略，但这属于后续增强，不应放入第一版正确性验证。

### 4.4 `T=2` 限制流水线收益上限

LoopFill 论文的示例和实验包含三到四次循环。GLM-5.4 只有两次循环，因此首轮起草到完整验证之间只有一个循环间隔。后缀树仍然可以一次铺出多个未来 Token，但流水线填充和排空时间更短，额外候选计算更容易抵消收益。

在理想的无限并行和高接受率条件下，`T=2` 允许每个全局循环步完成接近两个 Token 的工作。真实吞吐还要扣除：

- 750B MoE 的路由和跨卡通信。
- 候选树的索引、排序、剪枝和缓存管理。
- 分支节点的重复第一轮计算。
- GPU batch 中不同节点深度不一致带来的填充浪费。

因此不能把 LoopFill 在 Ouro 或 Huginn 上的 5× 级别结果外推到 GLM-5.4。GLM-5.4 的收益需要按单请求首 Token 延迟、生成阶段 ITL、并发吞吐和单位 Token 算力分别测量。

### 4.5 全局后缀树会带来数据隔离问题

GLM-5.4 面向代码、工具调用、视频和 3D 任务。代码模板、JSON、工具参数和重复的代理轨迹可能有较高后缀命中率，开放式推理和新任务则未必如此。全局树不应无条件混合所有用户输出。

生产实现至少需要：

- 默认使用当前会话的局部树。
- 全局树按租户、项目或明确授权的语料域隔离。
- 对敏感 Token、工具参数、凭据和用户数据禁用索引。
- 对全局树设置版本、容量、过期和命中质量阈值。

## 5. 可行性分级和建议配置

| 方案 | 目标 | 建议 | 判断 |
|---|---|---|---|
| A：离线轨迹验证 | 不改线上服务，只回放 GLM-5.4 两轮 logits | 统计首轮一致率、后缀命中率、最长接受路径 | 必做 |
| B：单请求后缀 LoopFill | 只启用局部后缀树，`N=2` 或 `4` | 保守重算分支索引，完整两轮验证 | 最适合首个原型 |
| C：后缀加首轮 Top-k | `d=1` 产生 `k=1` 或 `2`，与后缀树合并 | 记录每个候选的额外 FLOPs 和接受率 | 第二阶段 |
| D：自适应预算 | 用首轮置信度、门控统计和后缀命中率调节 `N`、`k`、`Q` | 先在单卡或小规模仿真中校准 | 第三阶段 |
| E：高并发分布式服务 | 深度分桶、页式 KV、跨卡候选合并 | 设置全局候选上限和请求级公平策略 | 最后实施 |

首版实验不建议同时引入全局后缀树、采样残差修正和索引激进复用。这样无法区分收益来源，也难以定位输出不一致的原因。

## 6. 验证指标

### 6.1 正确性

- 贪心输出与关闭 LoopFill 的 GLM-5.4 逐 Token 对比，要求逐 Token 一致。
- 采样模式比较随机种子固定时的目标分布、接受率和残差修正结果。
- 对每个候选记录使用的第一轮索引、第二轮索引、共享 KV 页和分支私有滑窗，检查是否存在跨分支污染。
- 对剪枝和异步回收做长序列压力测试，确认不会出现偶发错误或非法内存访问。

### 6.2 性能

至少同时报告以下指标：

```text
TTFT                         首 Token 延迟
ITL                          生成阶段每 Token 延迟
Throughput                   不同并发下的输出 Token/s
Accepted path length         每次提交的连续 Token 数
Loop steps per output token  每个输出 Token 平均消耗的全局循环步
Draft waste                  被剪枝节点已经执行的循环比例
KV memory                    公共页、分支页和滑窗页的峰值占用
Index overhead               分支索引重算带来的额外时间
```

实验矩阵应覆盖代码、工具调用、自然语言推理、长上下文检索和随机采样。至少比较以下基线：

1. 原生 GLM-5.4 两轮自回归解码。
2. 只有后缀树的非流水线验证。
3. 只有首轮自投机的流水线。
4. 完整 LoopFill。

在相同硬件、相同并发、相同输出长度和相同采样设置下比较，不能把官方“最高 200 tokens/s”与研究原型的单请求结果直接比较。

## 7. 最终判断

GLM-5.4 是 LoopFill 很合适的研究对象，因为它公开确认了共享循环层、首轮到次轮的状态复用和混合 KV Cache。LoopFill 也正好能利用这些中间状态，把后缀多 Token 候选和首轮模型候选放进同一棵树。

但 GLM-5.4 的两个循环、门控更新、稀疏注意力和分布式 MoE 同时提高了实现门槛。最大的风险不是“能不能生成候选”，而是“候选分支是否仍然执行了正确的第二轮目标模型，以及额外候选是否真的减少了端到端等待”。

因此，建议结论写成：

> **GLM-5.4 上可以采用 LoopFill 的自投机思路，推荐先做保守、免训练、完整两轮验证的局部后缀原型。首轮 Top-k 扩展、分支索引复用、自适应预算和高并发优化应分阶段加入。只有在逐 Token 输出一致、缓存无污染且并发吞吐仍有收益后，才适合进入生产推理引擎。**

## 8. LoopSpec 代码实现细读

本节只解释 `kaist-flexml-lab/loopspec` 当前代码如何组织一次 LoopSpec 推理，不把这些实现改写成 LoopFill 方案。

代码仓库：<https://github.com/kaist-flexml-lab/loopspec>

### 8.1 模型被明确拆成 `pre → recurrent → post`

LoopSpec 没有把完整模型简单复制成多个独立模型，而是把一次前向拆成三个执行角色。

| 角色 | 代码位置 | 主要工作 | 是否在每个循环迭代执行 |
|---|---|---|---|
| `pre` | `PreExecutor` | Token Embedding、Prelude、Raven 初始隐藏状态 | 否，新 lane 创建时执行一次 |
| `recurrent` | `OuroRecurrentExecutor` 或 `RavenRecurrentExecutor` | 共享的核心循环块 | 是，每个 lane 每个循环阶段执行 |
| `post` | `PostExecutor` | Coda、最终归一化、LM Head、LogitsProcessor | 否，只在 q1、q2、final 读出点执行 |

`RecurrentServingModel.forward()` 根据 `ForwardBatch.recurrent_role` 分派这三个角色。[`components.py` 的 `PreExecutor` 和 `PostExecutor`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/modeling/components.py#L49-L194) 只负责张量变换，循环调度和 KV 时间线由 LoopSpec runtime 管理。

`LoopedRecurrentServingModel` 是固定深度基线。它先执行一次 `pre`，再执行 `recurrent_steps` 次共享核心块，最后执行一次 `post`。共享的 recurrent 模块不复制权重，而是通过重新编号 attention layer ID，让每次循环使用不同的逻辑 KV 层。[`serving.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/modeling/serving.py#L185-L240)

### 8.2 Embedding 和 Prelude 的实际位置

#### Ouro

Ouro 的 `build_ouro()` 构造一个空的 `PreExecutor`，因此 Token 经过 `VocabParallelEmbedding` 后直接作为 recurrent hidden。每次 recurrent 迭代执行核心层并做 `step_norm`。Checkpoint 中的 `early_exit_gate` 会被加载并保留，但固定深度 serving 路径明确不会执行它。[`ouro.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/modeling/ouro.py#L88-L137)

#### Raven

Raven 的 `PreExecutor` 包含 Embedding 和 `n_layers_in_prelude` 个 Prelude block。Prelude 输出保存为 `state.injected`，同时根据 `init_std` 单独初始化 recurrent hidden。之后每个 recurrent 迭代都通过 adapter 拼接当前 hidden 和这份 injected Prelude 输出，再运行共享核心块。因此 Prelude 只计算一次，但它的结果会被每个循环迭代消费。[`raven.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/modeling/raven.py#L61-L100)

需要区分两个概念。`raven_pre` 和 `olmo_post` 是 Raven 单层内部的归一化布局名称，`pre`、`recurrent`、`post` 才是 LoopSpec 的执行角色。

`PreExecutor.forward()` 的行为可以概括为：

```text
hidden = input_embeds 或 embed_tokens(input_ids)
hidden *= embed_scale
hidden = Prelude(hidden)
Ouro  -> recurrent_hidden = hidden
Raven -> recurrent_hidden = 独立初始化值，injected = hidden
```

所以 Embedding 并不在每个循环中重复，也没有按 Token ID 产生不同的矩阵乘法量。它只在一个新 lane 首次进入 `pre` 时执行一次。实际 LoopSpec server 还拒绝 `input_embeds`、多模态输入和 `replace_embeds` 路径，因此支持的 serving 路径使用普通 Token ID 查表。[`loopspec_scheduler.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/server/loopspec_scheduler.py#L51-L75) [`cuda_graph.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/cuda_graph.py#L427-L444)

### 8.3 LM Head 不在 recurrent loop 内

`PostExecutor.forward()` 先运行可选的 post blocks，再做 `final_norm`，最后调用 `LogitsProcessor(input_ids, hidden_states, lm_head, forward_batch)`。因此 LM Head 是 `post` 角色的一部分，不是共享 recurrent block 的一部分。[`components.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/modeling/components.py#L153-L194)

LoopSpec 的 `PipelineSchedule` 将循环阶段映射为三个端点：

```text
K - 1       -> q1，第一提案
K * X - 1   -> q2，可选第二提案
S - 1       -> final，完整深度验证
```

其中 `S` 是每个 Token 的总循环次数，`K` 是一个 pipeline cycle 推进的循环次数，`X` 是 q2 相对 q1 的深度倍率。[`pipeline.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/pipeline.py#L124-L170)

一次 `executor.forward(lanes)` 的顺序是：

1. `stage == 0` 的新 lane 执行一次 `pre`。
2. 所有 live lane 执行 `K` 次 `recurrent`。
3. 只有本次结束阶段命中 q1、q2 或 final 的 lane 执行 `post`。
4. `post` 输出交给决策策略。贪心模式在 GPU 上 `argmax`，只把 Token ID 回传 CPU。采样模式保存 q1/q2 概率分布，最后用完整分布执行接受和残差修正。

[`executor.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/executor.py#L292-L367) [`decisions.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/decisions.py#L96-L177) [`sampling/policy.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/sampling/policy.py#L197-L304)

这意味着 q1/q2 的 LM Head 不是“提前得到一个近似 logits 后跳过 final”，而是额外的中间读出。最终 Token 仍由 final endpoint 的完整深度 logits 决定。若 q1 或 q2 被接受，代码只复用已经推进的 lane 和其 KV 时间线。

### 8.4 Post/Coda 是否会放大 LM Head 成本

如果模型没有 post attention，`request.post()` 使用 attention-free direct graph，从 state bank 读取选中的 hidden，执行 Coda、norm 和 LM Head，不创建新的 KV timeline。

如果模型存在 post attention，`request.post()` 会为每个 q1、q2、final 深度维护独立的 post cursor。prefill 阶段先为这些端点填充各自的 Prompt KV，decode 阶段只对端点 lane 推进 post cursor。[`request.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/request.py#L111-L192) [`request.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/request.py#L292-L323)

因此：

- Ouro 的 q1/q2 主要是 hidden state 到词表的读出和采样处理，post 为空，成本相对集中在 LM Head。
- Raven 的 q1/q2 可能还要执行 Coda 和 post attention，并为每个端点维护 KV 时间线。
- 共享 Embedding 权重和 LM Head 权重只减少参数存储。`lm_head.tie_weights(embed_tokens)` 不会把 Embedding 查表和 LM Head 计算融合，也不会减少 endpoint 的词表投影次数。[`serving.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/modeling/serving.py#L146-L178)

### 8.5 同一个 pipeline cycle 内为什么会出现位置和阶段负载不均

`PipelineLane` 同时记录三个不同维度：

```text
stage  当前要执行的 recurrent 迭代
index  预测的 completion 序号
token  当前作为输入的 Token
```

`stage` 不是物理 layer ID，`index` 也不是循环深度。[`pipeline.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/pipeline.py#L13-L43)

设一个 cycle 开始时有 `B_t` 个 live lane，只有 `P_t` 个 lane 的 `stage == 0`，有 `E_t` 个 lane 在本次结束阶段命中 q1/q2/final。那么这个 cycle 的执行结构近似为：

```text
recurrent：K × B_t 行
pre：      P_t 行，每行包含 Embedding 和可选 Prelude
post：     E_t 行，每行包含可选 Coda、norm、LM Head
```

这三个集合并不相等。一个 cycle 可能同时包含：

- 一个刚由 q1 创建、需要 Embedding 和 Prelude 的新 lane。
- 若干已经完成部分循环、只需要继续 recurrent 的旧 lane。
- 一个到达 final 的 lane，需要完整 post 和 LM Head。
- 另一些到达普通阶段的 lane，不需要任何 post。

代码不会把所有 lane 填充到同一个“大而全”的 forward。`_prepare_cycle()` 只把 `stage == 0` 的行放入 `pre_rows`，再用结束阶段筛选 `endpoints`。[`executor.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/executor.py#L205-L237)

这解决了角色之间的无效计算，却没有消除角色内部的全部负载不均。具体表现如下。

#### 位置长度不均

不同 lane 的 `index` 不同，所对应的 Token 位置和 prefix length 也不同。`DecodeStagingSlot.stage()` 为每行分别写入 `REQUEST`、`SEQUENCE_LENGTH`、`POSITION` 和 `TOKEN`。同一个 recurrent batch 内可以同时存在不同 `seq_len` 的 lane。[`metadata.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/metadata.py#L86-L137)

Attention backend 根据每行的 request mapping 和 sequence length 读取 KV。代码保留了这种 ragged metadata，但没有按照长度再分桶，也没有为短序列单独缩小 CUDA block。于是同一个 recurrent graph width 中，长 prefix 行会做更多 attention 访存，短 prefix 行可能先完成，形成 kernel 内部的尾部等待。

#### Embedding 和 Prelude 的阶段不均

Embedding 的成本属于 `P_t` 行，而 recurrent 的成本属于 `B_t` 行。新 lane 的数量随 q1/q2 接受情况变化，`P_t` 不是固定值。代码的处理方式是单独运行 pre role，而不是让所有 recurrent lane 再做一次 Embedding。对于没有 pre attention 的模型，pre 使用 direct graph，只有位置、Token 和 state-output slot 被上传，不建立 KV timeline。[`request.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/request.py#L241-L270)

这是一种角色级的负载隔离，不是位置级负载均衡。Embedding 查表本身不会因 Token ID 或位置产生不同 FLOPs，但新 lane 进入时间不同会让同一 cycle 的 pre batch 宽度波动。

#### LM Head 和端点不均

LM Head 的成本属于 `E_t` 行。`post()` 只接收命中端点的 state slots，非端点 lane 不会执行 LM Head。q1、q2 和 final 在不同 cycle 到达，端点 batch 可能很小，尤其在 pipeline 启动和排空阶段。

代码为 `pre`、`recurrent`、`post` 分别捕获不同 batch width 的 CUDA Graph，并把 graph key 写成 `(role, width, variant)`。`DecodeBatchPool` 也按角色和精确宽度复用 `ForwardBatch`。[`cuda_graph.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/cuda_graph.py#L36-L121) [`batch_pool.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/batch_pool.py#L25-L186)

这避免了把一个小的 LM Head endpoint batch 强行 padding 到全部 live lane，也避免了 recurrent lane 被 post 角色拖慢。代价是需要预捕获多种 width，而且小 width 的词表投影和独立 graph launch 可能利用率较低。代码没有对 q1/q2/final 的 endpoint 做跨 cycle 合并，也没有让 LM Head 等待更多 lane 以换取更大的矩阵乘。

### 8.6 分支和 KV 时间线如何保持这些不同位置的状态

每个逻辑 branch 在 `pre`、每个 recurrent iteration 和每个 post endpoint 都有自己的 `TimelineCursor`。新 branch 不复制完整 K/V，而是先借用父 branch 的 prefix mapping，在第一次 graph replay 时复制映射并追加新的 KV slot。[`branching/manager.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/branching/manager.py#L45-L190)

隐藏状态则由独立的 state bank 管理。`STATE_INPUT` 和 `STATE_OUTPUT` 只是 bank 行号，不能和 request-pool row 或 KV slot 混淆。recurrent 每次从指定 bank 行 gather hidden，执行一个循环后 scatter 回同一 lane 的 state slot。[`cuda_graph.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/cuda_graph.py#L317-L359)

这种设计把三种索引空间分开：

| 索引空间 | 表示 | 作用 |
|---|---|---|
| `stage` | recurrent 逻辑迭代 | 决定使用哪条 core 时间线和是否触发 q1/q2/final |
| `index`、`position` | Token 在序列中的位置 | 决定输入 Token、RoPE/位置和 attention prefix |
| `state slot` | GPU hidden bank 行 | 保存 lane 跨 cycle 的 hidden/injected |
| `request row` | SGLang mapping 行 | 保存某条 branch/role/depth 的 prefix 到 KV 映射 |
| KV slot | 物理 K/V 存储位置 | 保存一次具体 attention decode 写入 |

因此，不同 Token 位置可以在同一个 recurrent batch 中并行，但它们不会共享同一条“逻辑时间线”。分支剪枝时，`BranchManager` 只回收不再被 live lane 或后代依赖的 timeline，并把唯一子节点可以继续使用的 cursor ownership 转移过去。

### 8.7 代码实际采取的负载均衡边界

LoopSpec 的负载处理可以分成三层。

1. **角色隔离。** `pre`、`recurrent`、`post` 分开执行，避免每个 lane 每轮都重复 Embedding、Prelude 或 LM Head。
2. **端点稀疏化。** 只有 q1/q2/final 行执行 post，避免非读出阶段计算词表 logits。
3. **精确宽度 Graph。** 对每个角色捕获多个 exact-width graph，避免用最大 live lane 数 padding 所有角色。

它没有实现以下机制：

- 按 `seq_len` 或 `position` 对同一角色的 lane 再分桶。
- 按 Token 的 Embedding、attention 或 LM Head 实际耗时做 work stealing。
- 把多个 cycle 的小 endpoint batch 延迟合并成更大的词表矩阵乘。
- 在 GPU kernel 内根据每行的上下文长度动态重新分配 block。

所以，LoopSpec 解决的是**流水线阶段之间的结构性负载不均**，没有完全解决**同一阶段内部由序列长度和端点稀疏性造成的负载不均**。这也是它需要统计 `pre_rows`、`recurrent_rows`、`post_rows`、已验证 proposal 和拒绝数量的原因。[`executor.py`](https://github.com/kaist-flexml-lab/loopspec/blob/main/sglang_recurrent/loopspec/executor.py#L33-L45)

### 8.8 直接回答

- **LM Head 怎么处理。** 它位于 `post`，只在 q1/q2/final 端点对选中 lane 执行。q1/q2 是中间提案，final 是完整验证。贪心只回传 argmax，采样保存概率并在 final 做无损接受与残差处理。
- **Prelude 和其他非循环部分怎么处理。** Embedding 和 Raven Prelude 属于 `pre`，每个新 lane 只执行一次。Raven 的 Prelude 输出作为 injected state 在每次 recurrent 迭代参与 adapter。Coda 和 final norm 属于 `post`，按端点重复执行。Ouro 没有 Prelude/Coda，`early_exit_gate` 在固定深度 serving 中不执行。
- **Embedding 造成的同一 Loop 不同位置负载不均怎么处理。** 代码没有把 Embedding 放进 recurrent loop，因此不会在每个循环位置重复查表。新 lane 的 Embedding 由独立 pre batch 承担，旧 lane 继续执行 recurrent。不同序列位置造成的 attention 长度差异仍保留在同一 role batch 中，代码靠逐行 metadata 和 exact-width graph 管理，没有做长度级动态均衡。

## LoopSpec 代码参考

- [模型组件](https://github.com/kaist-flexml-lab/loopspec/tree/main/sglang_recurrent/modeling)
- [LoopSpec pipeline](https://github.com/kaist-flexml-lab/loopspec/tree/main/sglang_recurrent/loopspec)
- [分支与 KV 时间线](https://github.com/kaist-flexml-lab/loopspec/tree/main/sglang_recurrent/loopspec/branching)
- [服务调度器](https://github.com/kaist-flexml-lab/loopspec/tree/main/sglang_recurrent/server)

## 参考资料

1. Z.ai，**GLM-5.4: Toward Generalized Coding with Our First Loop Transformer**，2026-11-23。<https://z.ai/blog/glm-5.4>
2. 本仓库，**Looped Transformer 投机解码方法分析报告**。<https://github.com/YzTongNiar/GLM5.4-Analysis/blob/main/reports/looped-transformer-decoding-analysis.md>
3. 本仓库，**LoopFill 论文导读**。<https://github.com/YzTongNiar/GLM5.4-Analysis/blob/main/reports/loopfill-reading-guide.md>
4. 本仓库，**LoopFill 原始论文文件**。<https://github.com/YzTongNiar/GLM5.4-Analysis/blob/main/papers/2611.19649v1_LoopFill.pdf>
5. GLM-5.4 架构公开整理，**LLM Architecture Gallery**。<https://sebastianraschka.com/llm-architecture-gallery/>

## 证据边界

本文使用了 Z.ai 官方博客和本仓库已有 LoopFill 分析作为主要依据。Z.ai 尚未公开完整的 GLM-5.4 推理接口、候选树 API、分支索引语义或 LoopFill 组合实验，因此文中涉及分支索引策略、缓存布局和性能上限的部分属于工程推导，实施前需要以实际权重和推理运行时的 profiling 结果校正。
