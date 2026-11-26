# LoopSpec 实验洞察：接受长度、流水排布与 GLM-5.4 自投机设计

核对日期：2026-11-26。源码固定为官方仓库提交 [`41aa710ed7c9ce908cffa03597be5d1df19d7b56`](https://github.com/kaist-flexml-lab/loopspec/tree/41aa710ed7c9ce908cffa03597be5d1df19d7b56)，停止处理同时核对其依赖的 SGLang `0.5.14`。本文讨论 LoopSpec 的现有代码，不将 LoopFill 的批量路径提交机制套用到它上面。

## 1. 先回答三个问题

1. **接受长度如何计算？** 官方报表的 `accept_len` 是按第一提案接受、第二提案接受、最终回退的频数估算出的等效长度。它不等于一次验证接受的连续 Token 数，也不是最长成功草稿链。
2. **投机走多少步？** 一次 `forward(lanes)` 对所有活跃 lane 推进 `K` 个 recurrent iteration；单 lane 总共执行 `S` 次，经历 `D=S/K` 个 pipeline cycle。q1 深度为 `K`，可选 q2 深度为 `KX`。没有另一个“先连续起草 γ 个 Token、再统一验证”的固定长度参数。
3. **什么时候停？** 单 lane 达到完整深度 `S` 后结束；候选分支被完整验证淘汰后剪枝；整条请求只有在已提交输出满足长度、EOS、stop 等条件时正常结束。q1/q2 置信度、拒绝、草稿 EOS、候选树容量都不是正常的请求停止阈值。

最关键的实现事实：`continue_pipelined()` 每个 cycle 允许 **零个或一个 final lane**，多于一个会报错；只有 final 才向 `committed` 追加一个 Token。第一枚 completion Token 已在 prefill 结束后确定，不属于 decode pipeline 的接受统计。见 [pipeline.py](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/pipeline.py#L254-L423)。

## 2. 把三个“步数”分开

| 量 | 实际含义 | 不能混同的概念 |
|---|---|---|
| recurrent iteration | 同一个输入 Token 再运行一次共享核心块 | 单个 Transformer layer；一个新输出 Token |
| pipeline cycle | 一次 `executor.forward(lanes)`，所有 live lane 各推进 `K` 次核心迭代 | 固定长度草稿的一轮批量验证 |
| completion index | 该 lane **预测的下一 Token** 在输出中的序号 | lane 输入 Token 的序号；循环深度 |
| `stage` | 下一次要执行的核心迭代，零基编号 | 当前 Token 的序列位置 |

`S` 从模型配置读取：Ouro 使用 `total_ut_steps`，Raven 使用 `mean_recurrence`，两者都转换为固定整数；不是按 Token 动态抽样循环次数。报表对标准模型名称另行硬编码 Ouro 为 4、Raven 的 `Recurrent-` 为 32。若修改 checkpoint 的实际循环深度，需要同步修改报表映射，否则报表估算会失真。[contracts.py](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/modeling/contracts.py#L56-L65)；[result_parser.py](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/experiment_tools/result_parser.py#L27-L30)。

参数对应关系如下：

| 参数 | 含义 | 约束 |
|---|---|---|
| `--first` | `K`：第一提案的绝对循环深度，同时也是每 cycle 的推进量 | `1 ≤ K < S` 且 `S % K == 0` |
| `--second` | `d₂`：第二提案的**绝对**循环深度 | `K < d₂ < S` 且 `d₂ % K == 0` |
| 内部 `second_step` | `X=d₂/K`，第二端点的 cycle 序号 | `1 < X < D` |
| `max_new_tokens` | 包含 prefill 首 Token 的总输出预算 | 服务拒绝无界预算；0 和 1 单独处理 |

例如 Raven 的 `--first 2 --second 8` 表示 `K=2、d₂=8、X=4`；不是第二提案在第 16 次循环。对于 `S=32`，单 lane 需要 16 个 cycle，q1/q2/final 分别在第 1/4/16 个 cycle 读出。见 [cli.py](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/server/cli.py#L65-L94) 及 [参数转换](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/server/cli.py#L161-L168)。

Embedding/Prelude 不增加 `stage`，post/Coda/LM Head 也不增加 `stage`。循环进度只计算 recurrent 核心；这些非循环工作的时间仍属于真实延迟。[executor.forward](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/executor.py#L295-L366)。

## 3. “接受”到底接受了什么

### 3.1 接受一个提案，复用一条已经推进的子分支

父 lane 在 q1 或 q2 产生候选 Token 后，会为这个 Token 创建一个从 `stage=0` 开始的子 lane。子 lane 用它作为输入，预测再下一个 Token。父 lane 继续向完整深度前进，同时子 lane 和更深后代也在推进。

到 final 时，父 lane 确定一个真实输出 Token。随后只有同时满足以下条件的直接子 lane 才能复用：

- `parent_branch` 匹配父分支；
- 子 lane 的输入 `token` 匹配最终输出；
- `origin_stage` 匹配 `accepted_stage`，即确实接受了它来源的 q1 或 q2 提案。

因此，**采样回退碰巧抽到与某个旧草稿相同的 Token，也不能仅凭 Token 相同复用它**：`accepted_stage=None` 时会创建全新的 final fallback lane。选定子 lane 后保留其后代，剪掉竞争子树。接受的是当前输出位置的一个提案及其可复用计算，不是提前批准整棵后代树。[分支选择与剪枝](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/pipeline.py#L364-L423)。

### 3.2 贪心模式

q1 总会提出它的 argmax；q2 只有在 argmax 与 q1 不同时才创建第二提案。final 依次判断：

1. final argmax 等于 q1 Token：接受 q1；
2. 否则，存在 q2 且 final argmax 等于 q2 Token：接受 q2；
3. 否则：提交 final argmax，重新建立续接分支。

q2 不会在出生时替换或删除 q1 分支；两者一直保留到 final 选择。贪心规则不需要概率接受率阈值。[GreedyDecisionPolicy.resolve](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/decisions.py#L137-L183)。

### 3.3 采样模式

设 `Q₁` 是 q1 端点分布，`Q̃₂` 是 q2 端点原始分布，`P` 是完整深度目标分布；它们均经过请求指定的 temperature/top-k/top-p 处理和归一化。

- 第一候选 `x ~ Q₁`，在 final 以 `min(1, P(x)/Q₁(x))` 接受。
- q2 出生时，仅当 `Q̃₂(x) < Q₁(x)` 且正残差质量非零，才创建第二候选。实际提案分布是 `R₂(v) ∝ max(Q̃₂(v)−Q₁(v), 0)·1[v≠x]`，并归一化；不是直接从 `Q̃₂` 抽样。
- 第一候选被拒后，条件目标改为 `F = normalize(max(P−Q₁,0))`。
- 若存在第二候选 `y ~ R₂`，以 `min(1,F(y)/R₂(y))` 接受；若也拒绝，则从 `normalize(max(F−R₂,0))` 抽取最终回退 Token。
- 若没有 q2，直接从 `F` 回退。残差分布不再次做 top-p 截断；源码还对浮点导致的零残差设置有限回退。

q2 的门控只决定是否增开候选分支，**不会让父 lane 提前结束 recurrent，也不是 final 接受判断**。以上是代码的分布校正规则；本文未以 GPU 实验验证其数值误差或端到端分布一致性。见 [q2 residual kernel](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/sampling/sampler.py#L284-L345)、[q1 verify kernel](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/sampling/sampler.py#L430-L459)、[RejectionSampler](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/sampling/rejection.py#L41-L98)。

## 4. 官方 `accept_len` 的精确定义

先按一个报告聚合范围，把所有请求的计数相加：

| 符号 | 日志字段 | 含义 |
|---|---|---|
| `N` | `decode_tokens` | 已提交输出数减去每条请求的 prefill 首 Token |
| `A₁` | `first_accept` | final 接受 q1 的次数 |
| `R₁` | `first_reject` | final 拒绝 q1 的次数 |
| `A₂` | `second_accept` | q1 被拒后，final 接受 q2 的次数 |
| `F` | `second_skip + second_reject` | q1 被拒且没有接受 q2 的次数 |

有效日志满足 `N=A₁+R₁`、`R₁=A₂+F`。这里 `second_skip` 只统计 **q1 已被拒且没有第二提案可验证** 的 final 事件，不是所有 q2 门控跳过事件；在 q1 已接受时，q2 不验证也不会记作 `second_skip`。报表的 `second_acceptance` 分母是 `R₁`，包含 skip，并非只以实际 q2 验证次数为分母。[metrics.py](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/server/metrics.py#L32-L45)。

代码先计算：

```text
C_est = N·K + R₁·(d₂−K) + F·(S−d₂)
      = A₁·K + A₂·d₂ + F·S

accept_len = S·N / C_est
```

`C_est` 是按接受结果估算的串行 recurrent 深度总量，单位是“核心迭代步”，并不是实际 `pipeline_cycles`，更不是 GPU 上所有 lane 的总计算量。未开启 q2 时，解析器令 `d₂=S`，公式化为 `C_est=A₁K+R₁S`。`C_est≤0` 时返回 `None`，报告不显示有效数值。源码入口为 [cycle_metric / cycle_accept_length](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/experiment_tools/result_parser.py#L195-L224)，深度解析见 [cycle_depths](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/experiment_tools/result_parser.py#L401-L423)。

直观理解：接受 q1，对下一 Token 的推进间隔按 `K` 计费；接受 q2，按 `d₂` 计费；全部回退，按 `S` 计费。把平均间隔除进完整深度 `S`，得到一个等效长度。

例如 `S=4、K=1、d₂=2`，若 `N=100、A₁=70、A₂=20、F=10`，则 `C_est=70+40+40=150`，`accept_len=400/150≈2.667`。全部 q1 接受时为 4，全部 q2 接受时为 2，全部回退时为 1。这三个值都不是“一次 commit 了 4/2/1 个 Token”。

### 4.1 有限长度下，它和实际 cycle 数不同

设单条请求在 prefill 首 Token 之后提交了 `N≥1` 个 Token。对第 `i` 个 final 结果，记 `ℓᵢ` 为 q1/q2/fallback 对应的 `K/d₂/S`。由当前同步调度器可推导：

```text
C_est                 = ℓ₁ + ℓ₂ + ... + ℓ_N
K·pipeline_cycles     = S + ℓ₁ + ℓ₂ + ... + ℓ_(N−1)
K·pipeline_cycles     = C_est + S − ℓ_N
```

第一枚 decode Token 必须先走完整 `S` 次循环；最后一枚 decode Token 的接受结果不再用于推进下一个输出。因此报表未计入初始填充，却把最后一次接受带来的潜在收益也计算进去了。此关系针对当前正常完成的调度路径，不包括异常退出；多个请求要逐请求加边界项，不能当成一条长流。

因此实验中至少区分：

- **官方等效接受长度**：`S·N/C_est`，基于端点接受频数；
- **有限序列循环步数比**：`S·N/(K·pipeline_cycles)`，计入启动阶段；
- **真实速度比**：同硬件、请求、模型和采样配置下，基线与 LoopSpec 的实际时间比。

真实耗时还受到 pre/post、不同 batch width、attention 长度、分支与 KV 管理、采样、同步和 CPU 调度影响。`recurrent_rows = Σ(K·live_lanes)` 才接近“执行了多少 lane 次核心工作”的计数；它依然不是按序列长度加权的 FLOPs。executor 内部维护 `pre_rows/recurrent_rows/post_rows`，但当前 `generation_metrics` 日志只输出其中的 `pipeline_cycles` 和接受统计；要做角色成本分析，还需扩展日志或 profiling，不能假设报表已经包含所有计数。[executor 计数](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/executor.py#L339-L345)。

### 4.2 时间线：`S=4、K=1`，q1 全部接受

prefill 已提交 `c₀`，输出预算为 6，还需要生成 `c₁…c₅`。下表用预测目标给 lane 命名，例如 `c₄` lane 以第三个输出位置的 Token 为输入，预测第四个位置。采用贪心模式，开启深度 2 的 q2，并假设每次 q2 与 q1 的 argmax 相同。此时每个 q2 端点仍运行 LM Head，但不会开第二条分支。所有 final 都接受 q1。

| 拍 | pre：新 lane 的 Embedding / Prelude | recurrent：各 lane 本拍完成的深度 | post：本拍合批做 LM Head 的 lane | 提交 |
|---|---|---|---|---|
| 1 | c₁ | c₁→1 | c₁(q1) | — |
| 2 | c₂ | c₁→2，c₂→1 | c₁(q2)，c₂(q1) | — |
| 3 | c₃ | c₁→3，c₂→2，c₃→1 | c₂(q2)，c₃(q1) | — |
| **4** | **c₄** | **c₁→4，c₂→3，c₃→2，c₄→1** | **c₁(final)，c₃(q2)，c₄(q1)** | **c₁** |
| 5 | c₅ | c₂→4，c₃→3，c₄→2，c₅→1 | c₂(final)，c₄(q2)，c₅(q1) | c₂ |
| 6 | — | c₃→4，c₄→3，c₅→2 | c₃(final)，c₅(q2) | c₃ |
| 7 | — | c₄→4，c₅→3 | c₄(final) | c₄ |
| 8 | — | c₅→4 | c₅(final) | c₅ |

这里 `N=5`，官方 `accept_len=4×5/5=4`；实际执行 8 个 cycle，有限序列步数比为 `20/8=2.5`。每个有 final 的 cycle 仍然只提交一个 Token。

### 4.3 第 4 拍，哪些 lane 在等

第 4 拍开始时，c₁ 距离提交只剩一次核心迭代，c₂ 和 c₃ 的旧状态也已就绪。单卡执行器先为新来的 c₄ 做 Embedding 和模型配置的全部 Prelude 层。**c₁、c₂、c₃ 都要等 c₄ 的 pre 完成**，之后四行才一起运行 recurrent。到 post 时轮到 c₂ 等待：它刚到深度 3，可以保存状态，但下一次 recurrent 要等 c₁、c₃、c₄ 三个端点完成读出、决策返回并选定存活分支。c₁ 本身要用 final 的输出才能提交。GPU 在这两段等待期间仍分别执行 pre 和 post。[执行器的阶段顺序和决策回读](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/executor.py#L295-L366)

在这套实现中，lane 共享同一张卡；精确宽度的 batch 让第 4 拍只为一行做 pre、为三行做 post。如果另行设计多卡流水，把 Prelude、循环块和 LM Head 固定分到不同设备，阶段耗时差才会直接表现为某些卡空等，重叠调度也要重新实现。官方 LoopSpec 服务要求并行规模均为 1。[服务端并行约束](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/server/cli.py#L95-L108)

### 4.4 深 Prelude 在什么情况下会吃掉收益

按上表数行，prefill 之后共执行 **5 行 pre、20 行 recurrent 和 15 行 post**。普通自回归生成同样的 5 个 Token，也需要 5 行 pre 和 20 行 recurrent，只需要 5 行最终 post。LoopSpec 把这 20 行核心计算合并到 8 拍里。节省多少时间取决于批宽为 2、3、4 时一次核心计算的耗时。与此同时，5 次有用的 Prelude 仍在关键路径上，新增的 10 行中间读出也要占用 GPU。这里的行数是工作量计数，同拍多行可以合批，不能直接用行数乘单行延迟来预测耗时。

如果改变第 4 拍的结果，让 c₁ 的 final 拒绝 q1，而 q2 没有备用提案，调度器仍提交正确的 c₁。原先预测 c₂、c₃、c₄ 的三条 lane 都依赖错误前缀，它们已经执行过的 Prelude 和部分 recurrent 随剪枝丢弃。下一拍以正确的 c₁ 为输入，重新为预测 c₂ 的 lane 做 pre。Prelude 越深，一次拒绝可能浪费的时间越多。[选择续接分支的条件](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/pipeline.py#L364-L419)

**没有固定的 Prelude 层数阈值。** 假设普通解码每个 Token 耗时 10 ms，其中 Prelude 占 8 ms。如果在相同负载下，每个已提交 Token 的 Prelude 仍占至少 8 ms 关键路径，理想加速上限就是 `10/8=1.25×`。额外读出、错误分支和决策同步足以把速度比拉到 1 以下。这个算例给出的是条件上限。实测时应在相同请求与并发下比较每个已提交 Token 的毫秒数，再按 batch 宽度分解 pre、core、post 及同步耗时。

## 5. 前瞻可以多长，是否有固定投机长度上限

对于只有 q1、接受持续成功且输出预算足够的情况，`D=S/K` 个 lane 可以在稳态形成深度错开的链。根 lane 到达 final 的那个 cycle，链尾最多又产生一个新的零深度子 lane；从该 final lane 的预测位置看，刚产生的子 lane 最多领先 `D` 个 completion index，但它尚未执行。下一 cycle 旧 final lane 已退出，正常活跃 lane 数上界仍为 `D`。

启用 q2 会增加同一位置上的备选路径和总宽度，不会把 q1 的最早出生间隔缩短到一个 cycle 以内。`PipelineSchedule.max_rows` 用保守的完整候选树边界分配资源：

```text
无 q2：b[t]=1，max_rows=D
有 q2：前 X 项 b[t]=1；之后 b[t]=b[t−1]+b[t−X]
max_rows = sum(b[0:D])
```

例如 `S=4、K=1、X=2`，births 为 `[1,1,2,3]`，`max_rows=7`。这是活跃 lane 容量上界，不是 7 个 Token 的接受长度，也不是“投机 7 步后停止”。若实际树超过它，代码抛出 `RuntimeError`，没有实现一个达到容量后正常暂停扩展的动态策略。[容量推导](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/pipeline.py#L174-L193)。

候选扩展另外受 `lane.index+1 < max_new_tokens` 约束：如果子 lane 预测的输出位置已超出预算，就不再创建它。这解释了靠近输出末尾时为何流水线自然变窄。**整个生成可以跨越任意多个固定深度窗口，直到输出停止条件成立；`S/K` 只规定一个 lane 的寿命，不规定整条生成的总 cycle 数。**

## 6. 三个层级的停止条件

| 层级 | 触发条件 | 后续动作 |
|---|---|---|
| 单 lane | `end_stage == S−1` | 执行 final readout、提交其预测；父 lane 不再推进，选择续接子树 |
| 候选子树 | final 选中了竞争分支，或没有接受任何提案 | 剪掉不属于选中续接分支的后代；必要时从 stage 0 重启 |
| 整条请求 | 已提交输出达到 `max_new_tokens`，或服务端 stop 回调判断完成 | 停止生成并释放剩余分支 |
| 异常退出 | 多个 final、提交乱序、容量越界、未完成却无 lane 等 | 抛出错误；不代表正常 EOS 或预算完成 |

### 6.1 服务端只检查已经确认的输出

`generate_loopspec()` 在 prefill 首 Token 后调用一次 `should_stop`；之后 `continue_pipelined()` 每提交一个 final Token 就调用一次。回调把增量输出写入 SGLang `Req.output_ids`，再调用 `Req.update_finish_state(new_accepted_len)`。正常路径该增量是 1，**变量名 `new_accepted_len` 不是报表的 `accept_len`**。草稿还没有写入 `Req.output_ids`，即使草稿是 EOS，也不会触发这个停止检查。[生成入口](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/loopspec/executor.py#L401-L458)；[服务回调](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/server/loopspec_scheduler.py#L331-L354)。

锁定版本 SGLang 0.5.14 的实际检查顺序包括：已有结束状态/待结束状态、输出长度、grammar、非法 Token、stop 字符串或 stop regex、EOS/stop Token。LoopSpec 服务拒绝 constrained decoding，因此不能把上游存在的 grammar 处理说成 LoopSpec 已支持的功能。[SGLang update_finish_state](https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/managers/schedule_batch.py#L1403-L1438)。

还有两个容易遗漏的细节：

- 该 SGLang 版本的 `ignore_eos=True` 会跳过整个 Token 型停止检查，包括显式 `stop_token_ids`、模型 EOS、tokenizer EOS 和 additional stop IDs；字符串停止与长度停止仍会检查。[Token 型停止](https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/managers/schedule_batch.py#L1302-L1324)。
- 流式输出遇到 stop 字符串的可能前缀，会暂缓发送，等后续 Token 消除歧义；这本身不等于 recurrent pipeline 停止。

`max_new_tokens=0` 在服务端直接结束，不执行模型；`=1` 只需要 prefill 的首 Token；首 Token 已满足 EOS/stop 时，同样不启动 decode pipeline。无界 `max_new_tokens=None` 和非零 `min_new_tokens` 在此服务实现中不受支持。

### 6.2 q1、q2 和 early-exit gate 都不会替代完整深度

q1/q2 只让后继 Token 提前开始，当前父 lane 仍需走到 `S`。Ouro 加载了 `early_exit_gate` 权重但固定深度 serving 不执行它；Raven 使用固定 `mean_recurrence`，也没有按训练时的循环分布随机停止。候选被剪枝会让一部分未完成 lane 提前终止，但这是丢弃错误投机计算，不是让最终目标分布变浅。[OuroRecurrentExecutor](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/modeling/ouro.py#L83-L104)。

## 7. 与已有文档的核对及验证范围

远端 `001ee3d` 确实包含 [`GLM5.3LoopFill.md` 第 8 节](GLM5.3LoopFill.md#8-loopspec-代码实现细读)，此前本地落后于远端。已先快进同步，再补充本文；根目录另一个未跟踪的同名文件不是此次更新对象。

旧文中 `pre/recurrent/post` 角色拆分、端点 LM Head、q1/q2/final 深度和 KV 分支复用的方向成立。本次同时收紧两处表述：exact-width graph 避免无关 lane 做 post，但 **post 仍在同一 cycle 内执行并参与同步**，不能说 recurrent 完全不受 LM Head 耗时影响；LoopSpec 调度层没有长度分桶，不足以证明底层 attention kernel 没有动态切分或调度优化，不能仅从该层源码断言 kernel 的具体尾部等待行为。

在 CPU 上直接导入上述提交的 `pipeline.py` 和 `result_parser.py`，用模拟 endpoint 决策执行官方调度逻辑，得到以下结果（`S=4、K=1、q2=2`，prefill 后输出 5 个 Token）：

| final 决策序列 | 实际 commit cycle | 官方 `accept_len` | 有限序列步数比 |
|---|---|---|---|
| q1 × 5 | 4, 5, 6, 7, 8 | 4 | 2.5 |
| q2 × 5 | 4, 6, 8, 10, 12 | 2 | 1.667 |
| fallback × 5 | 4, 8, 12, 16, 20 | 1 | 1 |
| q1, q2, fallback, q1, q2 | 4, 5, 7, 11, 12 | 2 | 1.667 |

另验证了：回退 Token 即使与 q1 相同也不复用被拒来源的 lane、单 Token 预算不执行 cycle、提交后停止回调清空分支、`K=2` 的推进、非法 schedule 拒绝、无 q2 的报表深度替代，以及零 decode Token 不产生有效接受长度。模拟决策用于隔离和检验调度语义，不代表模型实际接受率。

本次没有运行模型权重、CUDA Graph 或真实 GPU benchmark；因此这里给出的是源码事实、公式推导和 CPU 调度验证，不能用表中数字声称 RTX 5080 的实测加速。

## 8. GLM-5.4 能怎样使用两轮循环做自投机

截至 2026-11-27，Z.ai 公开的结构给出了一个自然的提案位置。GLM-5.4 将中间 28 层作为共享块运行两轮，每轮通过独立的逐 Token 门控更新残差流。第二轮复用第一轮生成的稀疏注意力索引和选取的稀疏 KV，同时维护自己的局部滑窗 KV。官方指南未给出循环块前后各有多少层、首轮是否有可用的 next-token 读出头，或如何在服务接口上访问中间状态。下文是需要模型权重与推理引擎内部接口的原型设计。当前 LoopSpec 服务只接入 Ouro 和 Raven。[GLM-5.4 官方架构说明](https://docs.z.ai/guides/llm/glm-5.4) · [LoopSpec 模型检查](https://github.com/kaist-flexml-lab/loopspec/blob/41aa710ed7c9ce908cffa03597be5d1df19d7b56/sglang_recurrent/server/cli.py#L125-L139)

### 8.1 一个中间提案，完整两轮决定输出

若按完整的 28 层块选读出点，调度取 `S=2、K=1`，两个端点分别是首轮 q1 和次轮 final，没有容纳 q2 的中间循环深度。以已提交的输入 Token `x` 为例：

| 拍 | 当前 Token 的目标路径 | 投机路径 | 拍末决策 |
|---|---|---|---|
| 1 | 对 x 执行前置层和第 1 轮，从门控更新后的状态读出候选 y | 获得 y 后方可建立子分支 | 暂存 y |
| 2 | 对 x 执行第 2 轮和完整后置层，得到目标分布 P | 先做 y 的前置层，在数据依赖允许时把 y 的第 1 轮与 x 的第 2 轮合批 | 贪心时提交 argmax(P)。若等于 y，保留 y 的首轮进度，否则丢弃并用正确 Token 重启 |
| 3 | 命中则 y 已可运行第 2 轮并预测再下一个 Token | 若 y 已产生自己的 q1，可启动其后继的第 1 轮 | 未命中则从正确 Token 的第 1 轮重新填充流水 |

表中合批需要内核支持同批两行使用不同的轮次、门控、KV 映射和索引来源。还要确认子分支第 1 轮所需的父前缀状态都已就绪。按第 4 节的公式，设单提案接受比例为 `a`，则在长序列下等效接受长度为 `2/(2−a)`。即使 `a=80%` 也只有约 1.67，全部命中时循环步数比才趋近 2。两行共享权重带来的批处理收益、多卡通信和后置层耗时决定这些循环步数能换来多少毫秒。

### 8.2 先解决读出和缓存的正确性，再衡量收益

1. **用原生两轮解码收集配对状态。** 在同一已验证前缀上，分别记录第一轮门控后的隐藏状态与完整后置层输出的目标 logits。先试用模型自身的归一化和 LM Head 作为便宜的 q1 读出，测其与 final 的 top-1 一致率。如果读出必须重跑多层后置网络才够准，将这些层和词表投影计入每次提案的成本。有 attention 的浅层读出必须使用独立缓存或隔离的临时状态。必要时再比较轻量蒸馏读出头，此方案需要训练。门控值控制状态更新幅度，其预测接受概率的能力需要单独校准。
2. **按前缀、轮次和位置隔离状态。** 共享已提交前缀的只读页，为候选后缀保留独立的首轮隐藏状态、索引、KV 映射和次轮滑窗写入位置。一个 Token 的第二轮应消费该分支同一位置第一轮选出的索引与稀疏 KV。命中时提升正确子分支，拒绝时回收该分支私有页。用同一前缀从头重算的原生 final logits 作对照，覆盖接受、拒绝重启和滑窗滚动边界。
3. **先验证贪心，再接采样。** 贪心模式逐 Token 对照原生输出。采样模式保存实际抽出草稿的过滤后分布 Q，按 `min(1,P(y)/Q(y))` 接受，被拒时从归一化的正残差 `[P−Q]₊` 回退。若错误草稿也进入分布式执行，各 rank 要按同一决策提交或回收分支。
4. **单独测两轮能否合得划算。** 对同一模型、硬件拓扑和输入分布，比较原生解码、只加首轮读出、启用单候选流水三组。分别测单行和双行循环块耗时、前后置层与浅层读出耗时、缓存与跨卡通信开销，再报告实际接受率、每个已提交 Token 的延迟和并发吞吐。第二组可量出纯提案开销，第三组能验证跨位置合批是否足以收回它。公开的服务速度不能替代这组三方对照。
