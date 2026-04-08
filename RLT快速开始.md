# RLT 快速开始指南

## 步骤1：查看项目结构

确保以下文件/目录已正确创建：

- `src/openpi/policies/rlt/` - RLT 策略实现
- `src/openpi/training/rl_algorithms/rlt/` - RLT 算法实现
- `examples/rlt_workflow_example.py` - 使用示例

## 步骤2：查看 Policy 类

在 `src/openpi/policies/policy.py` 中已添加对 RLT 的支持：

- `__init__` 方法新增 `rlt_network_path` 和 `rlt_config` 参数
- `_rlt_refine_actions` 方法实现 RLT 动作精细化
- `set_task_strategy` 等方法支持 `rlt` 选项

## 步骤3：运行示例

```bash
# 快速测试（使用较小的训练步数）
python examples/rlt_workflow_example.py --mode train --offline_steps 100 --online_steps 200

# 查看输出目录
ls -la ./checkpoints/rlt/
```

## 步骤4：在你的项目中使用

在你的代码中这样使用：

```python
from openpi.policies.policy import Policy
from openpi.policies.rlt.configuration_rlt import RLTConfig

# 1. 加载你的模型
model = ...  # 加载 ACoT-VLA 模型

# 2. 创建配置和策略
config = RLTConfig()
policy = Policy(
    model,
    refinement_strategy="rlt",  # 在这里指定策略
    rlt_config=config,
    rlt_network_path="./checkpoints/rlt/rlt_complete.pkl"
)

# 3. 推理
outputs = policy.infer(observation)

# 4. 为特定任务切换策略
policy.set_task_strategy("sorting_packages", "rlt")
policy.set_task_strategy("pouring_workpiece", "simple")
```

## 常见问题

### 我不需要修改 `policy.py` 的第 39 行吗？

不需要！第 39 行只是默认值，你应该通过创建 Policy 实例时的参数来指定策略。

### 训练和推理是分开的吗？

是的！先使用 `examples/rlt_workflow_example.py` 训练网络，然后在 Policy 中加载训练好的网络进行推理。

### 策略说明

- **simple**: 高斯平滑 + 任务规则
- **rlinf**: 神经网络强化学习
- **rlt**: RL token + Actor-Critic
