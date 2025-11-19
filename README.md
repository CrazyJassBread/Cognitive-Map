# Cognitive-Map
### 环境介绍
**base_env**：环境抽象类的基础，提供基础的环境定义和渲染功能，子类可在其基础上自定义game规则与环境

**Task1**：基础任务设置，agent初始位置在（0，0），目标位置在（7，7）

**Task2**：修改了背景颜色，从0修改为4（干扰state观测值）
### 方法简介
由于Task1和Task2在任务上基本一致，因此将测试在Task1训练、Task2中迁移的表现

**PPO**、**A2C**、**DQN**等文件则是在不引入cognitive map的抽象机制前测试的文件

only A2C：train average 14 test average = 100steps

only PPO：train average 14 test average = 23steps

only DQN: train average 14 test average = 100steps

human: train average 14 test average = 14

PPO with cognitive map: train average =14 test average 14