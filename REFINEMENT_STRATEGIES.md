# ACoT-VLA 动作精细化策略指南

本项目提供了两种动作精细化策略，用于优化机器人策略的输出动作。

## 策略概述

### 方案1：简单精细化（Simple Refinement）- 默认启用

- **状态**: ✅ 已实现并测试通过
- **特点**: 高斯平滑 + 任务特定规则
- **适用场景**: 快速优化，简单有效
- **测试脚本**:
  - `test_simple_refinement.py` - 完整测试（依赖项目）
  - `test_simple_refinement_no_deps.py` - 简化测试（无依赖）

### 方案2：RLinF（Reinforcement Learning in the Loop）- 备用方案

- **状态**: ✅ 已实现，需要训练
- **特点**: 可学习的神经网络 + 强化学习训练
- **适用场景**: 方案1效果不够时使用
- **训练脚本**:
  - `train_rlinf.py` - RLinF 网络训练脚本

## 如何切换策略

### 方法1：创建时为每个任务单独指定策略

在创建 Policy 时，可以为每个任务单独指定策略：

```python
# 方案A：全局默认用简单精细化，特定任务用 RLinF
policy = Policy(
    model,
    refinement_strategy="simple",  # 全局策略
    task_strategy_map={
        "sorting_packages": "rlinf",   # 包裹分拣用 RLinF
        "pouring_workpiece": "simple",  # 倾倒工件用简单精细化
    }
)

# 方案B：全局默认用 RLinF，特定任务用简单精细化
policy = Policy(
    model,
    refinement_strategy="rlinf",   # 全局策略
    task_strategy_map={
        "opening_door": "simple",   # 开门用简单精细化
    },
    rlinf_network_path="./rlinf_net.pkl"  # 训练好的 RLinF 网络
)

# 方案C：所有任务都用同一种策略
policy = Policy(model, refinement_strategy="simple")
```

### 方法2：运行时动态设置单个任务的策略

```python
# 先创建策略
policy = create_trained_policy(config, checkpoint_path)

# 为某个任务单独设置策略
policy.set_task_strategy("sorting_packages", "rlinf")  # 包裹分拣用 RLinF
policy.set_task_strategy("pouring_workpiece", "simple")  # 倾倒工件用简单精细化

# 查看当前任务策略映射
print(policy.get_task_strategy_map())
```

### 方法3：运行时动态切换全局策略

```python
# 先创建策略
policy = create_trained_policy(config, checkpoint_path)

# 切换到 RLinF（所有没有单独设置策略的任务都会用这个）
policy.set_global_strategy("rlinf")

# 切换回简单精细化
policy.set_global_strategy("simple")
```

### 方法4：混合使用（最灵活）

```python
# 创建策略，全局用 simple，部分任务用 rlinf
policy = Policy(
    model,
    refinement_strategy="simple",
    task_strategy_map={
        "sorting_packages": "rlinf",
    },
    rlinf_network_path="./rlinf_net.pkl"
)

# 运行时发现另一个任务也需要 rlinf
policy.set_task_strategy("pouring_workpiece", "rlinf")

# 发现全局策略需要改
policy.set_global_strategy("rlinf")

# 某个任务改回 simple
policy.set_task_strategy("opening_door", "simple")
```

## 如何训练 RLinF 网络

### 步骤1：生成训练数据并训练

```bash
# 使用默认参数训练
python3 train_rlinf.py

# 指定参数
python3 train_rlinf.py --samples 2000 --epochs 200 --output ./my_rlinf_net.pkl
```

### 步骤2：测试训练好的网络

```bash
# 测试已保存的网络
python3 train_rlinf.py --test ./rlinf_net.pkl
```

### 步骤3：使用训练好的网络

```python
# 创建策略时指定网络路径
policy = Policy(
    model,
    refinement_strategy="rlinf",
    rlinf_network_path="./rlinf_net.pkl"
)
```

## 运行测试

### 测试方案1（简单精细化）

```bash
# 简化测试（无需项目依赖）
python3 test_simple_refinement_no_deps.py

# 完整测试（需要项目依赖）
python3 test_simple_refinement.py
```

### 测试方案2（RLinF）

```bash
# 先训练 RLinF 网络
python3 train_rlinf.py --samples 500 --epochs 50

# 然后测试
python3 test_rlinf_refinement.py
```

## 策略详情

### 方案1：简单精细化

```
动作序列 → 高斯平滑 → 任务特定规则 → 输出动作
```

- **高斯平滑**: 5点高斯窗口，减少动作噪声
- **任务特定规则**:
  - 包裹分拣: 抓握力度 ≥ 0.4，减少腕部晃动
  - 倾倒工件: 手腕倾斜 ≥ 0.6，避免过度提升
  - 开门: 优化手腕角度至 [0.1, -0.2, 0.3]

### 方案2：RLinF（Reinforcement Learning in the Loop）

```
动作序列 → 可学习网络优化 → 物理约束 → 高斯平滑 → 输出动作
              ↓
         动作质量评估
```

- **可学习网络**: 状态编码器 + 动作质量评估网络 + 动作改进网络
- **强化学习训练**: 使用简单精细化方法作为监督信号
- **物理约束**: 应用任务特定的物理限制
- **高斯平滑**: 最终平滑处理

## 文件位置

- 核心实现: `src/openpi/policies/policy.py`
- 方案1测试: `test_simple_refinement.py`, `test_simple_refinement_no_deps.py`
- 方案2训练: `train_rlinf.py`
- 使用文档: `REFINEMENT_STRATEGIES.md`

## 下一步建议

1. **首先使用方案1**: 运行训练，看看效果如何
2. **如果效果不够**:
   - 训练 RLinF 网络: `python3 train_rlinf.py`
   - 切换到方案2（RLinF）
3. **进一步优化**:
   - 根据具体任务调整策略参数
   - 为 RLinF 网络提供更多训练数据

## 注意事项

- RLinF 网络在没有训练时会自动降级到简单精细化
- 可以随时切换策略，无需重启
- 每个任务可以有独立的策略选择
- RLinF 网络需要训练数据（使用简单精细化方法生成）
