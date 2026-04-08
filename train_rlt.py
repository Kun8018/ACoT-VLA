#!/usr/bin/env python3
"""
RLT (RL Token) 算法训练脚本
这个脚本用于训练RLT算法，使用两阶段训练方法：
1. 离线阶段：训练RL-token编码器/解码器
2. 在线阶段：训练Actor-Critic网络

使用方式：
    # 基础训练
    python train_rlt.py

    # 指定配置
    python train_rlt.py --offline_steps 5000 --online_steps 10000
"""

import argparse
import logging
import sys
import numpy as np
from pathlib import Path

# 添加src目录到PATH，这样可以找到openpi模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 导入我们新创建的RLT模块
from openpi.policies.rlt.configuration_rlt import RLTConfig
from openpi.policies.rlt.modeling_rlt_jax import RLTPolicy


class SimpleReplayBuffer:
    """简单的重放缓冲区实现"""

    def __init__(self, capacity: int = 100000, state_dim: int = 0, action_dim: int = 7, chunk_size: int = 10, vla_dim: int = 2048, vla_seq_len: int = 50):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.vla_dim = vla_dim
        self.vla_seq_len = vla_seq_len

    def add(self, vla_embeddings, proprioception, ref_actions, actions, next_vla_embeddings, next_proprioception, next_ref_actions, rewards, dones):
        """添加一个样本到缓冲区"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = {
            "vla_embeddings": vla_embeddings,
            "proprioception": proprioception,
            "ref_actions": ref_actions,
            "actions": actions,
            "next_vla_embeddings": next_vla_embeddings,
            "next_proprioception": next_proprioception,
            "next_ref_actions": next_ref_actions,
            "rewards": rewards,
            "dones": dones,
        }
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """随机采样一个批次"""
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]

        return {
            "vla_embeddings": torch.stack([torch.from_numpy(b["vla_embeddings"]) for b in batch]),
            "proprioception": torch.stack([torch.from_numpy(b["proprioception"]) for b in batch]),
            "ref_actions": torch.stack([torch.from_numpy(b["ref_actions"]) for b in batch]),
            "actions": torch.stack([torch.from_numpy(b["actions"]) for b in batch]),
            "next_vla_embeddings": torch.stack([torch.from_numpy(b["next_vla_embeddings"]) for b in batch]),
            "next_proprioception": torch.stack([torch.from_numpy(b["next_proprioception"]) for b in batch]),
            "next_ref_actions": torch.stack([torch.from_numpy(b["next_ref_actions"]) for b in batch]),
            "rewards": torch.tensor([b["rewards"] for b in batch], dtype=torch.float32),
            "dones": torch.tensor([b["dones"] for b in batch], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.buffer)


class MockDataGenerator:
    """生成模拟数据用于演示RLT算法"""

    def __init__(self, config: RLTConfig, state_dim: int = 0, action_dim: int = 7):
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.vla_seq_len = config.vla_chunk_size
        self.vla_dim = config.rl_token.input_dim

    def generate_offline_sample(self):
        """生成一个离线训练样本"""
        return {
            "vla_embeddings": np.random.randn(self.vla_seq_len, self.vla_dim).astype(np.float32),
        }

    def generate_online_sample(self):
        """生成一个在线训练样本"""
        return {
            "vla_embeddings": np.random.randn(self.vla_seq_len, self.vla_dim).astype(np.float32),
            "proprioception": np.random.randn(self.state_dim).astype(np.float32) if self.state_dim > 0 else np.zeros(0, dtype=np.float32),
            "ref_actions": np.random.randn(self.config.chunk_size, self.action_dim).astype(np.float32),
            "actions": np.random.randn(self.config.chunk_size, self.action_dim).astype(np.float32),
            "next_vla_embeddings": np.random.randn(self.vla_seq_len, self.vla_dim).astype(np.float32),
            "next_proprioception": np.random.randn(self.state_dim).astype(np.float32) if self.state_dim > 0 else np.zeros(0, dtype=np.float32),
            "next_ref_actions": np.random.randn(self.config.chunk_size, self.action_dim).astype(np.float32),
            "rewards": np.random.uniform(-1.0, 1.0).astype(np.float32),
            "dones": np.random.choice([0.0, 1.0], p=[0.95, 0.05]).astype(np.float32),
        }


def create_batch_iterator(buffer: SimpleReplayBuffer, batch_size: int = 32):
    """创建一个批次迭代器"""
    while True:
        yield buffer.sample(batch_size)


def train_rlt(config: RLTConfig, state_dim: int = 0, action_dim: int = 7):
    """
    RLT算法的完整训练流程
    """
    logger.info("=" * 60)
    logger.info("RLT (RL Token) Training")
    logger.info("=" * 60)

    # 创建策略和算法
    policy = RLTPolicy(config, state_dim=state_dim, action_dim=action_dim)
    algorithm = RLTAlgorithm(policy, config)

    # 初始化数据生成器和缓冲区
    data_generator = MockDataGenerator(config, state_dim=state_dim, action_dim=action_dim)
    offline_buffer = SimpleReplayBuffer(
        capacity=config.offline_buffer_capacity,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        vla_dim=config.rl_token.input_dim,
        vla_seq_len=config.vla_chunk_size,
    )
    online_buffer = SimpleReplayBuffer(
        capacity=config.online_buffer_capacity,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=config.chunk_size,
        vla_dim=config.rl_token.input_dim,
        vla_seq_len=config.vla_chunk_size,
    )

    # 为离线阶段初始化优化器
    algorithm.optimizers = {
        "rl_token": torch.optim.Adam(
            list(policy.rl_token_encoder.parameters()) + list(policy.rl_token_decoder.parameters()),
            lr=config.rl_token_lr,
        )
    }

    logger.info(f"Starting Stage 1: Offline training ({config.offline_steps} steps)")

    # 阶段1: 离线训练
    for step in range(config.offline_steps):
        # 生成模拟数据并添加到缓冲区
        sample = data_generator.generate_offline_sample()
        # 离线缓冲区只需要vla_embeddings，但我们需要填充完整结构
        offline_buffer.add(
            sample["vla_embeddings"],
            np.zeros(state_dim, dtype=np.float32) if state_dim > 0 else np.zeros(0, dtype=np.float32),
            np.zeros((config.chunk_size, action_dim), dtype=np.float32),
            np.zeros((config.chunk_size, action_dim), dtype=np.float32),
            sample["vla_embeddings"],
            np.zeros(state_dim, dtype=np.float32) if state_dim > 0 else np.zeros(0, dtype=np.float32),
            np.zeros((config.chunk_size, action_dim), dtype=np.float32),
            0.0,
            0.0,
        )

        # 训练
        batch_iterator = create_batch_iterator(offline_buffer, batch_size=32)
        # 修改批次以只包含vla_embeddings
        modified_batch = next(create_batch_iterator(offline_buffer, batch_size=32))
        offline_batch = {"vla_embeddings": modified_batch["vla_embeddings"]}

        # 创建临时的离线批次迭代器
        def offline_batch_iter():
            while True:
                b = offline_buffer.sample(32)
                yield {"vla_embeddings": b["vla_embeddings"]}

        stats = algorithm.offline_update(offline_batch_iter())

        if step % 100 == 0:
            logger.info(f"Offline step {step}/{config.offline_steps}: {stats.to_log_dict()}")

    logger.info("Stage 1 complete. Transitioning to online phase.")

    # 切换到在线阶段
    algorithm.transition_to_online()

    logger.info(f"Starting Stage 2: Online training ({config.online_steps} steps)")

    # 阶段2: 在线训练
    for step in range(config.online_steps):
        # 生成模拟数据并添加到缓冲区
        sample = data_generator.generate_online_sample()
        online_buffer.add(**sample)

        # 只有当缓冲区足够大时才开始训练
        if len(online_buffer) >= config.online_step_before_learning and step >= config.warmup_steps:
            batch_iterator = create_batch_iterator(online_buffer, batch_size=32)
            stats = algorithm.update(batch_iterator)

            if step % 100 == 0:
                logger.info(f"Online step {step}/{config.online_steps}: {stats.to_log_dict()}")
        else:
            if step % 100 == 0:
                logger.info(f"Online step {step}/{config.online_steps}: Filling buffer ({len(online_buffer)}/{config.online_step_before_learning})")

    logger.info("Stage 2 complete. Saving checkpoint.")

    # 保存检查点
    checkpoint_dir = Path("./checkpoints/rlt")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "policy_state_dict": policy.state_dict(),
        "algorithm_state_dict": algorithm.state_dict() if hasattr(algorithm, "state_dict") else None,
        "config": config,
    }, checkpoint_dir / "rlt_checkpoint.pt")

    logger.info(f"Checkpoint saved to {checkpoint_dir / 'rlt_checkpoint.pt'}")

    return policy


def main():
    parser = argparse.ArgumentParser(description="RLT (RL Token) Training Script")
    parser.add_argument("--offline_steps", type=int, default=500, help="Number of offline training steps")
    parser.add_argument("--online_steps", type=int, default=1000, help="Number of online training steps")
    parser.add_argument("--state_dim", type=int, default=7, help="Dimension of proprioceptive state")
    parser.add_argument("--action_dim", type=int, default=7, help="Dimension of action")
    parser.add_argument("--chunk_size", type=int, default=10, help="Size of action chunks")
    parser.add_argument("--vla_dim", type=int, default=2048, help="Dimension of VLA embeddings")

    args = parser.parse_args()

    # 创建配置
    config = RLTConfig(
        offline_steps=args.offline_steps,
        online_steps=args.online_steps,
        chunk_size=args.chunk_size,
    )
    config.rl_token.input_dim = args.vla_dim
    config.rl_token.rl_token_dim = args.vla_dim

    # 运行训练
    policy = train_rlt(config, state_dim=args.state_dim, action_dim=args.action_dim)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
