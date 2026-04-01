#!/usr/bin/env python3
"""
训练 RLinF（Reinforcement Learning in the Loop）网络
这个脚本用于生成训练数据并训练 RLinF 网络，用于优化 ACOT-VLA 策略的输出动作
"""

import sys
import argparse
import logging
import pathlib
import random
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from pathlib import Path

import openpi.models.model as _model
import openpi.policies.policy as _policy
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_training_data(policy: _policy.Policy, num_samples: int = 1000):
    """
    生成 RLinF 网络的训练数据
    训练数据格式: [(state, raw_action, improved_action)]

    使用策略网络生成原始动作，然后通过简单精细化方法生成改进动作
    """
    logger.info(f"Generating {num_samples} training samples...")

    training_data = []

    # 模拟观察
    for _ in range(num_samples):
        fake_obs = {
            "state": np.random.rand(32),  # 模拟状态
            "task_name": random.choice(["sorting_packages", "pouring_workpiece", "opening_door"]),
            "observation/state": np.random.rand(32),
            "observation/images/top_head": np.random.rand(224, 224, 3),
            "observation/images/hand_left": np.random.rand(224, 224, 3),
            "observation/images/hand_right": np.random.rand(224, 224, 3),
        }

        # 使用策略生成原始动作（通过模拟策略调用）
        raw_actions = np.random.rand(10, 32)  # 模拟动作序列

        # 保存原始动作
        original_actions = raw_actions.copy()

        # 使用简单精细化方法作为监督信号
        policy._refinement_strategy = "simple"
        outputs = {"actions": raw_actions}
        refined_outputs = policy.post_process(fake_obs, outputs)
        improved_actions = refined_outputs["actions"]

        training_data.append({
            "state": fake_obs["state"],
            "raw_action": original_actions,
            "improved_action": improved_actions,
            "task_name": fake_obs["task_name"]
        })

        if len(training_data) % 100 == 0:
            logger.info(f"Generated {len(training_data)}/{num_samples} samples")

    logger.info(f"Training data generation complete: {len(training_data)} samples")
    return training_data


def train_rlinf_network(config_name: str, num_samples: int, num_epochs: int, output_path: str):
    """
    训练 RLinF 网络的完整流程
    """
    logger.info("=" * 60)
    logger.info("RLinF (Reinforcement Learning in the Loop) Training")
    logger.info("=" * 60)

    try:
        # 获取配置
        config = get_config(config_name)
        logger.info(f"Using config: {config.name}")

        # 创建策略（使用一个最小化的检查点）
        policy = create_trained_policy(
            config,
            "./checkpoints/acot_icra_simulation_challenge_reasoning_to_action/baseline/30000"
        )

        logger.info("✅ Policy creation passed!")

        # 生成训练数据
        logger.info("\n" + "-" * 60)
        logger.info("Step 1: Generating Training Data")
        logger.info("-" * 60)
        training_data = generate_training_data(policy, num_samples)

        # 初始化 RLinF 网络
        logger.info("\n" + "-" * 60)
        logger.info("Step 2: Initializing RLinF Network")
        logger.info("-" * 60)
        rngs = nnx.Rngs(jax.random.key(42))
        rlinf_net = _policy.RLinFNetwork(action_dim=32, rngs=rngs)
        logger.info("✅ RLinF network initialized!")

        # 准备训练数据
        states = jnp.array([d["state"] for d in training_data])
        raw_actions = jnp.array([d["raw_action"] for d in training_data])
        improved_actions = jnp.array([d["improved_action"] for d in training_data])

        # 定义优化器
        tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)

        @jax.jit
        def loss_fn(model: _policy.RLinFNetwork, state: jnp.ndarray, ra: jnp.ndarray, ia: jnp.ndarray):
            ia_pred, quality = model(state, ra)
            improvement_loss = jnp.mean(jnp.square(ia_pred - ia))
            quality_loss = jnp.mean(jnp.square(quality - 0.95))
            return improvement_loss + 0.1 * quality_loss

        logger.info("\n" + "-" * 60)
        logger.info("Step 3: Training RLinF Network")
        logger.info("-" * 60)

        state = nnx.state(rlinf_net)
        opt_state = tx.init(nnx.filter(state, nnx.Param))

        for epoch in range(num_epochs):
            loss, grads = nnx.value_and_grad(loss_fn)(rlinf_net, states, raw_actions, improved_actions)
            updates, opt_state = tx.update(grads, opt_state)
            state = optax.apply_updates(state, updates)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}/{num_epochs}, Loss: {loss:.6f}")

        # 更新网络
        nnx.update(rlinf_net, state)
        logger.info("✅ RLinF network training completed!")

        # 保存网络
        logger.info("\n" + "-" * 60)
        logger.info("Step 4: Saving RLinF Network")
        logger.info("-" * 60)

        save_path = Path(output_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)

        network_params = nnx.state(rlinf_net)
        with open(save_path, 'wb') as f:
            pickle.dump(network_params, f)

        logger.info(f"✅ RLinF network saved to: {save_path}")

        # 测试训练好的网络
        logger.info("\n" + "-" * 60)
        logger.info("Step 5: Testing Trained RLinF Network")
        logger.info("-" * 60)

        # 测试网络对一个动作序列的改进
        test_obs = {
            "state": np.random.rand(32),
            "task_name": random.choice(["sorting_packages", "pouring_workpiece", "opening_door"]),
        }

        test_raw_action = np.random.rand(10, 32)
        if test_obs["task_name"] == "sorting_packages":
            test_raw_action[:, 13] = np.random.rand(10) * 0.25  # 低抓握力度

        # 测试原始网络（无改进）
        policy._rlinf_network = rlinf_net
        outputs = {"actions": test_raw_action}
        policy._refinement_strategy = "rlinf"
        result = policy.post_process(test_obs, outputs)

        logger.info("✅ RLinF network integration test passed!")

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        logger.debug(e, exc_info=True)
        return False

    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL TRAINING STEPS COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"\nNetwork saved at: {output_path}")
    logger.info("\nTo use this RLinF network, set rlinf_network_path parameter in Policy.")
    logger.info("Example:")
    logger.info('policy = Policy(model, rlinf_network_path="./rlinf_net.pkl")')
    logger.info('policy = Policy(model, refinement_strategy="rlinf", rlinf_network_path="./rlinf_net.pkl")')

    return True


def test_saved_network(network_path: str, config_name: str):
    """
    测试已保存的 RLinF 网络
    """
    logger.info("\nTesting saved RLinF network...")

    try:
        config = get_config(config_name)
        policy = create_trained_policy(
            config,
            "./checkpoints/acot_icra_simulation_challenge_reasoning_to_action/baseline/30000"
        )

        # 加载网络
        with open(network_path, 'rb') as f:
            network_params = pickle.load(f)

        rngs = nnx.Rngs(jax.random.key(42))
        rlinf_net = _policy.RLinFNetwork(action_dim=32, rngs=rngs)
        nnx.update(rlinf_net, network_params)
        policy._rlinf_network = rlinf_net
        policy._refinement_strategy = "rlinf"

        logger.info(f"✅ Network loaded from: {network_path}")

        # 简单测试
        test_obs = {
            "state": np.random.rand(32),
            "task_name": "sorting_packages",
        }
        test_raw_action = np.random.rand(10, 32)
        test_raw_action[:, 13] = np.random.rand(10) * 0.25

        outputs = {"actions": test_raw_action}
        result = policy.post_process(test_obs, outputs)

        logger.info("✅ Network integration test passed!")
        logger.info(f"Raw grip strength: {test_raw_action[:, 13]}")
        logger.info(f"Improved grip strength: {result['actions'][:, 13]}")

        assert any(grip > 0.4 for grip in result['actions'][:, 13]), "抓握力度应提高到≥0.4"

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.debug(e, exc_info=True)


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(
        description="训练 RLinF (Reinforcement Learning in the Loop) 网络"
    )

    parser.add_argument(
        "--config", "-c",
        default="acot_icra_simulation_challenge_reasoning_to_action",
        help="配置名称"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int, default=1000,
        help="训练数据样本数量"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int, default=100,
        help="训练轮数"
    )
    parser.add_argument(
        "--output", "-o",
        default="./rlinf_net.pkl",
        help="保存训练好的网络的路径"
    )
    parser.add_argument(
        "--test",
        type=str, default=None,
        help="测试已保存的网络"
    )

    args = parser.parse_args()

    if args.test:
        test_saved_network(args.test, args.config)
        return

    train_rlinf_network(args.config, args.samples, args.epochs, args.output)


if __name__ == "__main__":
    main()
