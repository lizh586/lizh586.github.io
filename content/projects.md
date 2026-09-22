---
title: "项目"
description: "手写实现与开源工作"
date: "2026-09-22"
---

## LLaMA from Scratch — 从零手写 LLaMA 与训练管线

[GitHub](https://github.com/lizh586/llama-from-scratch)

不用 `transformers`，不用 `trl`：从零手写 LLaMA 全部组件，并在自写模型上跑通 **Pretrain → SFT** 两阶段训练管线。

| 组件 | 说明 |
|------|------|
| RoPE 旋转位置编码 | 复数推导 → 旋转矩阵 → 频率计算 |
| RMSNorm | 去中心化归一化 |
| SwiGLU 门控 FFN | 门控 + Swish + 线性投影（3 矩阵） |
| GQA 分组查询注意力 | KV 头共享 + RoPE + causal mask |
| Decoder Layer / Full Model | Pre-Norm + residual 组装，端到端前向 + `attn_mask` + `generate` |

每个组件与 PyTorch 参考实现逐项数值比对，误差 < 1e-5。仓库内附 3 个自包含测试：RMSNorm 的数值与梯度、全模型的形状与 `generate` 不变式、attention mask 判别（含一个正控 —— 不 mask 时必须显著不同，否则测试本身失去区分力）。

### 训练管线

| 阶段 | 数据 | 结果 |
|------|------|------|
| Pretrain | shakespeare_char（字符级） | 800 步，val **1.5242** nats |
| SFT | Alpaca-cleaned | 1782 步，val **2.9280 → 1.5597** |

SFT 这块最容易错也最值得看：prompt 段 label 置 `-100`，只在 response 段算 loss；核心不变式 `labels[t] == x[t+1]`（非 `-100` 处）被落成可失败的断言而不是 print。RM → PPO（RLHF 后两阶段）推进中。

数据与 checkpoint 不进仓库（体积原因），README 里写了需要自备的路径。

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
