# Looped Transformer 解码方法分析

本仓库整理 **LoopSpec、LoopAhead、LoopFill** 三种面向 Looped Transformer 的解码加速方法，并收录对应原始论文、中文分析报告和方法对比图。

报告重点回答四个问题：

1. 为什么 Looped Transformer 的自回归解码延迟较高？
2. 三种方法分别如何产生草稿、调度候选并复用计算？
3. 它们为什么能够保持完整深度模型的输出或输出分布？
4. 两篇同名 “LoopAhead” 论文的方法有什么本质区别？

## 阅读入口

**[中文分析报告：Looped Transformer 投机解码方法分析](reports/looped-transformer-decoding-analysis.md)**

建议按以下顺序阅读：

1. 背景与统一问题框架；
2. LoopSpec：浅层原生预测、双提案与级联验证；
3. LoopAhead：学习型 LAD 多 Token 草稿与中层反馈调度；
4. LoopFill：后缀树与浅层预测组成的双草稿源；
5. 三者横向对比及同名 LoopAhead 辨析。

## 仓库结构

```text
GLM5.4-Analysis/
├── README.md
├── reports/
│   └── looped-transformer-decoding-analysis.md
├── papers/
│   ├── 2609.17184v1_LoopSpec.pdf
│   ├── 2610.21594v1_LoopAhead.pdf
│   ├── 2611.19649v1_LoopFill.pdf
│   └── 33690_LoopAhead_Parallelling_R.pdf
├── assets/
│   ├── loop-decoding-comparison.png
│   └── loop-decoding-comparison.html
└── diagrams/
    └── loop-decoding-comparison.json
```

- `papers/`：原始论文，仅用于学习与研究，版权归原作者及出版方所有。
- `reports/`：中文分析报告，包含公式、流程和 `France／Paris` 具体示例。
- `assets/`：方法对比图；HTML 文件下载后可用浏览器打开，支持缩放与深浅色切换。
- `diagrams/`：对比图的可编辑 JSON 源文件。

## 核心结论

| 方法 | 草稿机制 | 是否训练 | 候选控制 |
|---|---|---|---|
| LoopSpec | 目标模型自身的浅层原生预测 | 否 | 更深的第二提案、门控与级联验证 |
| LoopAhead | 学习型 LAD，根据浅层上下文和 Anchor 多 Token 起草 | 是 | 中层冲突反馈、暂停／恢复、执行预算 |
| LoopFill | 后缀树检索＋浅层预测 | 否 | 统一路径评分、Top-N 节点预算 |

三者的共同原则是：**草稿决定提前计算什么，完整深度模型决定最终输出什么。**

> 注意：`2610.21594v1` 的 LoopAhead 与 `33690_LoopAhead_Parallelling_R.pdf` 是同名但不同的工作。后者并行的是同一 Token 前缀的不同循环深度，采用预测器与校正器，并非本报告所比较的多 Token 投机解码基线。
