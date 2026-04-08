# Copyright 2026 ACoT-VLA Authors. All rights reserved.
"""
RLT 与 ACoT-VLA 集成示例

这个示例展示了如何将 RLT (RL Token) 算法集成到 ACoT-VLA 项目中：
1. 使用 ACoT-VLA 作为冻结的 VLA 骨干网络
2. 在其上添加 RLT 的 RL-token 编码/解码器
3. 使用两阶段训练方法
"""

import logging
import sys
from pathlib import Path

# 添加src目录到PATH，这样可以找到openpi模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 导入我们的 RLT 模块
from openpi.policies.rlt.configuration_rlt import RLTConfig
from openpi.policies.rlt.modeling_rlt_jax import RLTPolicy
from openpi.policies.policy import Policy


def usage_example():
    """
    展示如何在 Policy 中使用 RLT
    """
    logger.info("=" * 60)
    logger.info("RLT + ACoT-VLA 集成使用示例")
    logger.info("=" * 60)

    logger.info("""

在你的代码中这样使用 RLT 策略：

from openpi.policies.policy import Policy
from openpi.policies.rlt.configuration_rlt import RLTConfig

# 1. 加载 ACoT-VLA 模型
model = load_your_acot_vla_model()

# 2. 创建 RLT 配置
rlt_config = RLTConfig()

# 3. 创建 Policy 实例，使用 RLT 策略
policy = Policy(
    model,
    refinement_strategy="rlt",  # 这里指定策略
    rlt_config=rlt_config,
    rlt_network_path="./checkpoints/rlt/rlt_complete.pkl"
)

# 4. 推理
outputs = policy.infer(observation)

# 5. 为特定任务单独设置策略
policy.set_task_strategy("sorting_packages", "rlt")
policy.set_task_strategy("pouring_workpiece", "simple")

# 或者设置全局策略
policy.set_global_strategy("rlt")
    """)

    # 注意：实际运行时需要加载模型，这里只展示用法
    return True


def rlt_workflow_summary():
    """
    完整的 RLT 使用流程总结
    """
    logger.info("\n" + "=" * 60)
    logger.info("完整的 RLT 工作流程")
    logger.info("=" * 60)
    logger.info("""

## 步骤 1：训练 RLT 网络

# 运行训练脚本
cd /path/to/ACoT-VLA
python examples/rlt_workflow_example.py --mode train --offline_steps 5000 --online_steps 20000

## 步骤 2：在 Policy 中使用训练好的 RLT

在你的应用代码中：

from openpi.policies.policy import Policy
from openpi.policies.rlt.configuration_rlt import RLTConfig

# 创建 Policy
policy = Policy(
    model,
    refinement_strategy="rlt",
    rlt_config=RLTConfig(),
    rlt_network_path="./checkpoints/rlt/rlt_complete.pkl"
)

# 推理
outputs = policy.infer(obs)

## 可用的策略选项

目前支持三个精细化策略：

1. simple - 高斯平滑 + 任务规则（默认）
2. rlinf - 可学习的神经网络
3. rlt - RL token 方法（新增）
    """)


if __name__ == "__main__":
    usage_example()
    rlt_workflow_summary()
