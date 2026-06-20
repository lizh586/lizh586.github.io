---
title: "项目"
description: "手写实现与开源工作"
date: "2026-06-20"
---

## RL from Scratch — 强化学习算法手写

[GitHub](https://github.com/lizh586/rl-from-scratch)

基于 PyTorch 从零实现三个主流强化学习算法，覆盖 value-based / policy gradient / max-entropy 三类方法，gymnasium 标准环境全部 solved：

| 算法 | 环境 | 结果 |
|------|------|------|
| DQN | CartPole-v1 / LunarLander-v2 | 728ep / 938ep solved |
| PPO | CartPole-v1 / LunarLander-v2 | 62iter / 405iter solved |
| SAC | Pendulum-v1 | eval return 稳定于 -120 |
| REINFORCE ± baseline | CartPole-v1 | ~800ep / ~400ep |

## DL 核心组件 — 从零手写

与 PyTorch 参考实现逐项验证：

- **NumPy 反向传播框架**：Tensor 类 + 计算图 + topological sort + backward，41 参数 MLP 梯度误差 < 1e-5
- **Transformer Encoder**：ScaledDotProductAttention → Multi-Head Attention → Positional Encoding → EncoderBlock
- **CNN / ResNet**：conv2d / LeNet (Fashion-MNIST 90.08%) / ResNet 残差块 (92.48%)
- **RNN 家族**：LSTM cell (diff 2.98e-8) / GRU cell (diff 1.2e-7)
- **BPE Tokenizer + nn.Embedding**

代码见 [GitHub rl-from-scratch](https://github.com/lizh586/rl-from-scratch) `dl/` 目录。

## 技术博客

[从零手写 DQN / PPO / SAC：三个强化学习算法的实战记录](https://zhuanlan.zhihu.com/p/2046555983841908330)

## MAB 缓存策略优化

校级大创项目（负责人，优秀结题前 15%）。将缓存决策建模为在线学习问题，推导遗憾上界 O(√n log n)，C++ 仿真系统命中率较 LRU 提升 5%–8%。
