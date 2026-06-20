---
title: "从零手写 DQN / PPO / SAC：三个强化学习算法的实战记录"
description: "一个数学转 CS 的研 0 学生，一个月从零实现 RL 算法并开源的全过程"
date: "2026-06-06"
tags: ["RL", "DQN", "PPO", "SAC", "PyTorch"]
---

## 背景

我是中山大学数学学院大四，保研上海交大计算机。2026 年 5 月开始系统自学强化学习，目标是在入学前完成从理论到代码的闭环。

这篇文章记录了我从零手写 DQN、PPO、SAC 三个算法并全部 solved 的过程。

## 成果一览

| 算法 | 类型 | 环境 | 收敛结果 |
|------|------|------|----------|
| DQN | Value-based | CartPole / LunarLander | 728ep / 938ep solved |
| PPO | Policy gradient | CartPole / LunarLander | 62iter / 405iter solved |
| SAC | Max-entropy | Pendulum | eval return ~ -120 稳定 |

## 关键收获

**DQN**：ReplayBuffer + target network 打破自举不稳定性。ε-greedy 不能每 episode 衰减——要每 step 衰减，否则探索不足。

**PPO**：clipped surrogate objective 限制 policy update 幅度。调参中独立定位 3 个 bug——advantage 符号反了、GAE λ 写错、entropy coef 初始值太大。

**SAC**：双 Q + 自动熵调节，不需要手动设探索率。lr 从 1e-3 降到 3e-4 加梯度裁剪后稳定。

## 完整文章

包含算法原理解析、调参记录、收敛曲线，详见 [知乎专栏](https://zhuanlan.zhihu.com/p/2046555983841908330)。

代码开源：[github.com/lizh586/rl-from-scratch](https://github.com/lizh586/rl-from-scratch)
