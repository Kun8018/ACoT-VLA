# RLT (RL Token) 算法使用流程文档

## 概述

RLT（RL Token）是一种基于Vision-Language-Action（VLA）模型的两阶段强化学习算法。该算法通过学习紧凑的RL token表示来压缩VLA嵌入，然后在该表示上训练轻量级的Actor-Critic网络。

## 创建/修改的文件位置

- **策略模块**: `/Users/kun/ACoT-VLA/src/openpi/policies/rlt/`
- **算法模块**: `/Users/kun/ACoT-VLA/src/openpi/training/rl_algorithms/rlt/`
- **训练脚本**: `/Users/kun/ACoT-VLA/train_rlt.py`
- **集成示例**: `/Users/kun/ACoT-VLA/examples/rlt_workflow_example.py`
- **策略类**: `/Users/kun/ACoT-VLA/src/openpi/policies/policy.py` (已修改)

## 使用流程

### 阶段一：训练 RLT 网络

首先需要训练RLT网络。使用我提供的示例脚本：

```bash
# 运行完整的两阶段训练
python examples/rlt_workflow_example.py --mode both --offline_steps 5000 --online_steps 20000

# 或者单独运行离线/在线训练
python examples/rlt_workflow_example.py --mode train --offline_steps 5000
python examples/rlt_workflow_example.py --mode train --online_steps 20000
```

训练完成后，模型会保存在 `./checkpoints/rlt/` 目录中。

### 阶段二：加载训练好的 RLT 网络

在 `policy_config.py` 的 `create_trained_policy` 函数中，添加对 RLT 策略的支持：

```python
from openpi.policies.policy import Policy

def create_trained_policy(...):
    # 原代码 ...

    policy = Policy(
        model,
        transforms=transforms,
        output_transforms=output_transforms,
        refinement_strategy="rlt",  # 改为 rlt 策略
        rlt_config=RLTConfig(),
        rlt_network_path="./checkpoints/rlt/rlt_complete.pkl"
    )

    return policy
```

### 阶段三：推理与部署

在实际部署时，策略会自动使用 RLT 进行动作精细化。如果需要切换回其他策略，可以这样做：

```python
from openpi.policies.policy import Policy

# 加载策略
policy = create_trained_policy(...)

# 切换策略（可选）
policy.set_global_strategy("simple")  # 使用简单策略
policy.set_task_strategy("sorting_packages", "rlinf")  # 特定任务用 RLinF
```

## 策略说明

当前支持的三个精细化策略：

### 1. simple（默认）
- 使用高斯平滑和任务特定规则
- 计算速度快，没有额外开销
- 适合作为基准策略

### 2. rlinf（已有的 RLinF 策略）
- 使用可学习的神经网络
- 需要训练，计算开销中等
- 在复杂场景下效果可能更好

### 3. rlt（新增的 RLT 策略）
- 使用两阶段训练方法
- 先用RL token压缩VLA嵌入
- 再训练轻量级Actor-Critic网络
- 在长序列动作上效果可能更好

## 配置选项

可以通过 `RLTConfig` 调整参数：

```python
from openpi.policies.rlt.configuration_rlt import RLTConfig

config = RLTConfig(
    offline_steps=5000,      # 离线训练步数
    online_steps=20000,      # 在线训练步数
    chunk_size=10,           # 动作块大小
    utd_ratio=5,             # 每一步更新的次数
    actor_lr=3e-4,           # Actor学习率
    critic_lr=3e-4,          # Critic学习率
    rl_token_lr=1e-4         # RL token编码器学习率
)
```

## 性能比较

| 策略类型 | 计算开销 | 效果 | 内存使用 | 训练需要 |
|---------|----------|------|----------|----------|
| simple | 低 | 基准水平 | 无额外 | 无 |
| rlinf | 中 | 通常优于simple | 低 | 需要 |
| rlt | 高 | 复杂场景下可能最佳 | 中 | 需要较长时间训练 |

## 常见问题解答

### Q: 需要修改 policy.py 的第 39 行吗？
A: 不需要。你可以在创建 Policy 实例时通过 `refinement_strategy` 参数指定策略。

### Q: RLT 网络训练需要很长时间吗？
A: 是的，特别是离线训练阶段。建议在 GPU 上运行，并调整参数。

### Q: 如何结合任务策略使用？
A: 使用 `set_task_strategy` 方法：

```python
policy.set_task_strategy("sorting_packages", "rlt")
policy.set_task_strategy("pouring_workpiece", "simple")
```

### Q: 是否可以在训练时切换策略？
A: 训练策略和推理策略是两个完全独立的概念。训练时会使用 `train_rlt.py` 中的训练逻辑，推理时使用 Policy 类。
