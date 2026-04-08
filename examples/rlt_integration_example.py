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
from typing import Optional

# 添加项目根目录到PATH，这样可以找到openpi模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import flax.nnx as nnx

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 导入我们的 RLT 模块
from openpi.policies.rlt.configuration_rlt import RLTConfig
from openpi.policies.rlt.modeling_rlt_jax import RLTPolicy


class ACOTVLAEmbeddingExtractor(nn.Module):
    """
    从 ACoT-VLA 模型中提取 VLA 嵌入的包装器

    在实际使用中，这会包装你预训练的 ACoT-VLA 模型
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        super().__init__()
        # 模拟一个 VLA 嵌入器
        # 在实际使用中，这里会加载你预训练的 ACoT-VLA 模型
        self.embedding_dim = 2048
        self.sequence_length = 50
        self.dummy = nn.Linear(3, self.embedding_dim)  # 占位层

        if checkpoint_path is not None:
            logger.info(f"Loading ACoT-VLA checkpoint from {checkpoint_path}")
            # 这里加载实际的模型权重

    def forward(self, images, task_name, state=None):
        """
        从输入中提取 VLA 嵌入

        Args:
            images: 图像输入 (B, C, H, W) 或 (B, N, C, H, W)
            task_name: 任务名称字符串
            state: 机器人状态 (可选)

        Returns:
            vla_embeddings: VLA 嵌入 (B, sequence_length, embedding_dim)
            ref_actions: 参考动作 (B, chunk_size, action_dim)
        """
        batch_size = images.shape[0] if len(images.shape) == 4 else images.shape[0]

        # 模拟 VLA 嵌入生成
        # 在实际使用中，这里会调用真实的 ACoT-VLA 模型
        vla_embeddings = torch.randn(
            batch_size, self.sequence_length, self.embedding_dim,
            device=images.device
        )

        # 模拟参考动作生成
        ref_actions = torch.randn(batch_size, 10, 7, device=images.device)  # 10个动作chunks, 7维动作

        return vla_embeddings, ref_actions


class RLTIntegratedPolicy(nn.Module):
    """
    RLT + ACoT-VLA 集成策略

    结合了冻结的 ACoT-VLA 骨干网络和 RLT 的 RL-token 处理
    """

    def __init__(self, config: RLTConfig, vla_checkpoint: Optional[str] = None, state_dim: int = 7, action_dim: int = 7):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 冻结的 VLA 骨干网络
        self.vla_backbone = ACOTVLAEmbeddingExtractor(vla_checkpoint)
        for param in self.vla_backbone.parameters():
            param.requires_grad = False

        # RLT 策略
        self.rlt_policy = RLTPolicy(config, state_dim=state_dim, action_dim=action_dim)

    def forward(self, images, task_name, proprioception=None):
        """
        前向推理

        Args:
            images: 图像输入
            task_name: 任务名称
            proprioception: 机器人本体感觉状态 (可选)

        Returns:
            refined_actions: RLT 优化后的动作
        """
        if proprioception is None:
            proprioception = torch.zeros(images.shape[0], self.state_dim, device=images.device)

        # 从 VLA 骨干网络获取嵌入和参考动作
        vla_embeddings, ref_actions = self.vla_backbone(images, task_name, proprioception)

        # 使用 RLT 策略优化动作
        refined_actions = self.rlt_policy(vla_embeddings, proprioception, ref_actions)

        return refined_actions


def create_integrated_policy(
    vla_checkpoint: Optional[str] = None,
    rlt_checkpoint: Optional[str] = None,
    state_dim: int = 7,
    action_dim: int = 7,
):
    """
    创建集成的 RLT + ACoT-VLA 策略

    Args:
        vla_checkpoint: ACoT-VLA 检查点路径
        rlt_checkpoint: RLT 检查点路径
        state_dim: 状态维度
        action_dim: 动作维度

    Returns:
        policy: 集成的策略
    """
    config = RLTConfig()

    policy = RLTIntegratedPolicy(
        config,
        vla_checkpoint=vla_checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    if rlt_checkpoint is not None:
        logger.info(f"Loading RLT checkpoint from {rlt_checkpoint}")
        checkpoint = torch.load(rlt_checkpoint)
        policy.rlt_policy.load_state_dict(checkpoint["policy_state_dict"])

    return policy


def example_inference():
    """
    示例推理流程
    """
    logger.info("=" * 60)
    logger.info("RLT + ACoT-VLA 集成示例: 推理")
    logger.info("=" * 60)

    # 创建策略
    policy = create_integrated_policy(
        # vla_checkpoint="./checkpoints/acot_vla.pt",  # 你的 ACoT-VLA 检查点
        # rlt_checkpoint="./checkpoints/rlt/rlt_checkpoint.pt",  # 你的 RLT 检查点
        state_dim=7,
        action_dim=7,
    )

    policy.eval()

    # 模拟输入
    device = policy.config.device
    batch_size = 1
    images = torch.randn(batch_size, 3, 224, 224, device=device)  # 单张图像
    task_name = "sorting_packages"
    proprioception = torch.randn(batch_size, 7, device=device)  # 7维状态

    # 推理
    with torch.no_grad():
        refined_actions = policy(images, task_name, proprioception)

    logger.info(f"Input images shape: {images.shape}")
    logger.info(f"Refined actions shape: {refined_actions.shape}")
    logger.info("✅ Inference example complete!")


def example_training():
    """
    示例训练流程
    """
    logger.info("=" * 60)
    logger.info("RLT + ACoT-VLA 集成示例: 训练")
    logger.info("=" * 60)

    # 创建配置
    config = RLTConfig(
        offline_steps=100,  # 为了演示，使用少量步数
        online_steps=200,
    )

    # 创建策略和算法
    rlt_policy = RLTPolicy(config, state_dim=7, action_dim=7)
    algorithm = RLTAlgorithm(rlt_policy, config)

    # 创建 VLA 嵌入器（实际使用中会用真的）
    vla_extractor = ACOTVLAEmbeddingExtractor()
    vla_extractor.eval()  # 冻结
    for param in vla_extractor.parameters():
        param.requires_grad = False

    # 初始化优化器 (离线阶段)
    algorithm.optimizers = {
        "rl_token": torch.optim.Adam(
            list(rlt_policy.rl_token_encoder.parameters()) +
            list(rlt_policy.rl_token_decoder.parameters()),
            lr=config.rl_token_lr,
        )
    }

    logger.info("Starting Stage 1: Offline RL-token training")

    # 模拟离线训练
    device = config.device
    for step in range(config.offline_steps):
        # 模拟获取 VLA 嵌入
        dummy_images = torch.randn(8, 3, 224, 224, device=device)
        with torch.no_grad():
            vla_embeddings, _ = vla_extractor(dummy_images, "sorting_packages")

        # 创建离线批次
        batch = {"vla_embeddings": vla_embeddings}

        # 简单的离线批次迭代器
        def batch_iter():
            yield batch

        stats = algorithm.offline_update(batch_iter())

        if step % 20 == 0:
            logger.info(f"Offline step {step}: {stats.to_log_dict()}")

    logger.info("Stage 1 complete. Transitioning to online phase.")

    # 切换到在线阶段
    algorithm.transition_to_online()

    logger.info("✅ Training example complete!")


if __name__ == "__main__":
    example_inference()
    print()
    example_training()
