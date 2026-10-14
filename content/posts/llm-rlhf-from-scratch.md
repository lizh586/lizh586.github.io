---
title: "从零手写 RLHF 四阶段：Pretrain → SFT → RM → PPO 的实战记录"
description: "在自写的字符级 LLaMA 上把四个阶段串成一条线，结果 PPO 没提升对齐，反而把它 SFT 学到的收尾能力练没了"
date: "2026-10-14"
tags: ["RLHF", "LLM", "SFT", "PPO", "奖励模型", "PyTorch"]
---

## 背景

我是上海交通大学计算机学院 2026 级硕士研究生，本科毕业于中山大学数学学院。LLaMA 的全部组件（RoPE / RMSNorm / SwiGLU / GQA / attn_mask / generate）都是手写的，与 PyTorch 参考实现逐项比对数误差 < 1e-5。

这篇文章记录我在这套自写模型上把 Pretrain → SFT → RM → PPO 四个阶段串成一条线的过程——包括失败的那一半。

## 成果一览

| 阶段 | 配置 | 结果 |
|------|------|------|
| Pretrain | 字符级 Shakespeare，~9.5M 参数 / 1.00M token | val 拐点落在 800 步——瓶颈是数据量，不是步数 |
| SFT | Alpaca，lr 1e-5，1782 步 | val 2.93 → 1.56；含 EOS 0/60 → 58/60 |
| RM | 成对偏好，softplus(-Δ) | heldout 准确率 64.6%，只比抛硬币好 14.6 个点 |
| PPO | β=0.1，K=4 | 没提升对齐，反而退化 |

## 关键收获

**最容易翻车的是跨阶段接口，不是阶段内部。** `PAD_ID == EOS_ID`、SFT 右 pad 而 RM 左 pad、ckpt 键空间——整条线出过返工的三处，全都是「错了不报错、数值静默变」。

**KL 惩罚的梯度根本没接上。** β 从 0.1 扫到 100（跨 1000 倍），KL 与参数漂移零趋势。根因是一行 `.detach()` 把 ∂KL/∂θ 整条切断了——β 只改 Â 的数值、不回传一点梯度。

**「参数没动」和「输出没变」是两件事。** 参数层空转的时候，输出层照样能全部分岔。

## 完整文章

包含四阶段的接口设计、KL 梯度断链的完整证据链与踩坑记录，详见 [知乎专栏](https://zhuanlan.zhihu.com/p/2093725043406660747)。

代码开源：[github.com/lizh586/llama-from-scratch](https://github.com/lizh586/llama-from-scratch)
