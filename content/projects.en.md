---
title: "Projects"
description: "Hand-written implementations and open-source work"
date: "2026-09-22"
---

## LLaMA from Scratch — LLaMA and its training pipeline, written from scratch

[GitHub](https://github.com/lizh586/llama-from-scratch)

No `transformers`, no `trl`: every LLaMA component hand-written, with the **Pretrain → SFT** stages of the training pipeline running on my own model.

| Component | Notes |
|-----------|-------|
| RoPE | Complex-number derivation → rotation matrix → frequency computation |
| RMSNorm | Mean-free normalization |
| SwiGLU FFN | Gating + Swish + linear projections (3 matrices) |
| GQA | Grouped KV heads + RoPE + causal mask |
| Decoder Layer / Full Model | Pre-Norm + residual assembly, end-to-end forward + `attn_mask` + `generate` |

Each component is verified numerically against a PyTorch reference, error < 1e-5. The repo ships 3 self-contained tests: RMSNorm values and gradients, full-model shapes and `generate` invariants, and an attention-mask discriminator (with a positive control — unmasked, pad content *must* change the output, otherwise the test has no discriminating power).

### Training Pipeline

| Stage | Data | Result |
|-------|------|--------|
| Pretrain | shakespeare_char (character-level) | 800 steps, val **1.5242** nats |
| SFT | Alpaca-cleaned | 1782 steps, val **2.9280 → 1.5597** |

SFT is where the subtle bugs live: prompt tokens get `-100` labels so loss is computed on the response span only, and the invariant `labels[t] == x[t+1]` (wherever `labels != -100`) is enforced as a failing assertion rather than a print. RM → PPO (the two RLHF stages) are in progress.

Datasets and checkpoints are not shipped (size), so the README documents the paths you need to supply.

## RL from Scratch

[GitHub](https://github.com/lizh586/rl-from-scratch)

Three mainstream RL algorithms implemented from scratch in PyTorch, covering value-based / policy-gradient / max-entropy methods, all solved on standard gymnasium environments:

| Algorithm | Environment | Result |
|-----------|-------------|--------|
| DQN | CartPole-v1 / LunarLander-v2 | 728ep / 938ep solved |
| PPO | CartPole-v1 / LunarLander-v2 | 62iter / 405iter solved |
| SAC | Pendulum-v1 | eval return steady at -120 |
| REINFORCE ± baseline | CartPole-v1 | ~800ep / ~400ep |

## Deep Learning Building Blocks — from scratch

Verified item by item against PyTorch references:

- **NumPy autograd framework**: Tensor class + computation graph + topological sort + backward; gradient error < 1e-5 on a 41-parameter MLP
- **Transformer Encoder**: ScaledDotProductAttention → Multi-Head Attention → Positional Encoding → EncoderBlock
- **CNN / ResNet**: conv2d / LeNet (Fashion-MNIST 90.08%) / ResNet residual block (92.48%)
- **RNN family**: LSTM cell (diff 2.98e-8) / GRU cell (diff 1.2e-7)
- **BPE Tokenizer + nn.Embedding**

Code lives in the `dl/` directory of [rl-from-scratch](https://github.com/lizh586/rl-from-scratch).

## Blog

[Implementing DQN / PPO / SAC from scratch: notes from three reinforcement learning algorithms](https://zhuanlan.zhihu.com/p/2046555983841908330) (in Chinese)

## MAB Cache Replacement

University-level undergraduate research project (PI, top 15% on completion). Modeled cache replacement as an online learning problem, derived a regret upper bound of O(√n log n); the C++ simulator improved hit rate by 5%–8% over LRU.
