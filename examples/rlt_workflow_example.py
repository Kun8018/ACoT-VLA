#!/usr/bin/env python3
"""
RLT 算法完整工作流程示例

这个脚本展示了RLT算法的完整工作流程：
1. 阶段1：离线训练 RL-token 编码器/解码器
2. 阶段2：在线训练 Actor-Critic
3. 推理阶段：加载训练好的RLT网络，在Policy中使用

使用方式：
    # 训练RLT
    python examples/rlt_workflow_example.py --mode train

    # 使用训练好的RLT进行推理
    python examples/rlt_workflow_example.py --mode infer
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

# 添加项目根目录到PATH，这样可以找到openpi模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from openpi.policies.rlt.configuration_rlt import RLTConfig
from openpi.policies.rlt.modeling_rlt_jax import RLTokenEncoder, RLTokenDecoder, RLTActor, RLTPolicy


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_rlt_offline(config: RLTConfig, output_dir: Path = Path("./checkpoints/rlt")):
    """
    阶段1：离线训练 RL-token 编码器/解码器

    目标：学习从VLA嵌入中提取紧凑的RL表示
    """
    logger.info("=" * 60)
    logger.info("RLT 阶段1：离线训练 RL-token Encoder/Decoder")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建模型
    rng = jax.random.PRNGKey(42)
    encoder = RLTokenEncoder(
        input_dim=config.rl_token.input_dim,
        rl_token_dim=config.rl_token.rl_token_dim,
        num_layers=config.rl_token.num_encoder_layers,
        num_heads=config.rl_token.num_heads,
        ff_dim=config.rl_token.ff_dim,
        dropout=config.rl_token.dropout,
    )
    decoder = RLTokenDecoder(
        rl_token_dim=config.rl_token.rl_token_dim,
        output_dim=config.rl_token.input_dim,
        num_layers=config.rl_token.num_decoder_layers,
        num_heads=config.rl_token.num_heads,
        ff_dim=config.rl_token.ff_dim,
        dropout=config.rl_token.dropout,
    )

    # 初始化模型参数
    rng, init_rng = jax.random.split(rng)
    dummy_vla = jax.random.normal(init_rng, (1, 50, config.rl_token.input_dim))
    dummy_z_rl = encoder(dummy_vla)
    _ = decoder(dummy_z_rl, dummy_vla)

    # 优化器
    optimizer = optax.adam(config.rl_token_lr)
    opt_state = optimizer.init(nnx_utils.state(encoder) | nnx_utils.state(decoder))

    # 训练循环
    logger.info(f"Starting offline training for {config.offline_steps} steps...")

    for step in range(config.offline_steps):
        # 模拟训练数据
        batch_size = 32
        vla_embeddings = jax.random.normal(
            jax.random.PRNGKey(step),
            (batch_size, 50, config.rl_token.input_dim)
        )

        def loss_fn(encoder_state, decoder_state):
            temp_encoder = nnx.clone(encoder)
            temp_decoder = nnx.clone(decoder)
            nnx.update(temp_encoder, encoder_state)
            nnx.update(temp_decoder, decoder_state)

            # 前向传播
            z_rl = temp_encoder(vla_embeddings)
            z_reconstructed = temp_decoder(z_rl, vla_embeddings)

            # 重构损失
            loss = jnp.mean((z_reconstructed - vla_embeddings) ** 2)
            return loss

        # 计算梯度
        encoder_state = nnx_utils.state(encoder)
        decoder_state = nnx_utils.state(decoder)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(encoder_state, decoder_state)

        # 更新参数
        updates, opt_state = optimizer.update(grads[0] | grads[1], opt_state)
        optax.apply_updates(encoder_state | decoder_state, updates)

        if step % 100 == 0:
            logger.info(f"Offline step {step}/{config.offline_steps}: loss={loss:.6f}")

    # 保存训练好的模型
    logger.info("Saving offline trained models...")
    with open(output_dir / "rlt_offline_encoder.pkl", 'wb') as f:
        pickle.dump(nnx_utils.state(encoder), f)
    with open(output_dir / "rlt_offline_decoder.pkl", 'wb') as f:
        pickle.dump(nnx_utils.state(decoder), f)

    logger.info(f"✅ Offline training complete! Models saved to {output_dir}")
    return encoder, decoder


def train_rlt_online(config: RLTConfig, encoder, decoder, output_dir: Path = Path("./checkpoints/rlt")):
    """
    阶段2：在线训练 Actor-Critic

    目标：在RL-token表示上进行强化学习
    """
    logger.info("\n" + "=" * 60)
    logger.info("RLT 阶段2：在线训练 Actor-Critic")
    logger.info("=" * 60)

    # 冻结编码器/解码器
    for param in nnx_utils.state(encoder).values():
        param.requires_grad = False
    for param in nnx_utils.state(decoder).values():
        param.requires_grad = False

    # 创建Actor
    actor = RLTActor(
        state_dim=config.rl_token.rl_token_dim + 7,
        action_chunk_dim=config.chunk_size * 7,
        hidden_dims=config.actor.hidden_dims,
        std=config.actor.std,
    )

    # 初始化Actor
    dummy_state = jax.random.normal(jax.random.PRNGKey(43), (1, config.rl_token.rl_token_dim + 7))
    dummy_ref = jax.random.normal(jax.random.PRNGKey(44), (1, config.chunk_size * 7))
    _ = actor(dummy_state, dummy_ref)

    # 简单的在线训练模拟
    logger.info(f"Starting online training for {config.online_steps} steps...")

    for step in range(config.online_steps):
        if step % 100 == 0:
            logger.info(f"Online step {step}/{config.online_steps}")

    # 保存完整的RLT策略
    logger.info("Saving complete RLT policy...")

    complete_policy = RLTPolicy(config, state_dim=7, action_dim=7)

    # 加载离线训练好的编码器
    with open(output_dir / "rlt_offline_encoder.pkl", 'rb') as f:
        encoder_state = pickle.load(f)
    # 这里需要将encoder_state应用到complete_policy的编码器中

    with open(output_dir / "rlt_complete.pkl", 'wb') as f:
        pickle.dump(nnx_utils.state(complete_policy), f)

    logger.info(f"✅ Online training complete! Complete policy saved to {output_dir}")
    return complete_policy


def run_inference_with_rlt(config: RLTConfig, rlt_checkpoint_path: Path):
    """
    使用训练好的RLT策略进行推理

    注意：这里展示如何在Policy中使用RLT策略
    """
    logger.info("\n" + "=" * 60)
    logger.info("RLT 推理阶段")
    logger.info("=" * 60)

    logger.info(f"Loading RLT policy from {rlt_checkpoint_path}")

    try:
        # 加载训练好的RLT网络
        with open(rlt_checkpoint_path, 'rb') as f:
            rlt_state = pickle.load(f)

        rlt_policy = RLTPolicy(config, state_dim=7, action_dim=7)
        nnx.update(rlt_policy, rlt_state)

        logger.info("✅ RLT policy loaded successfully")

        logger.info("""
使用示例：
    from openpi.policies.policy import Policy
    from openpi.policies.rlt.configuration_rlt import RLTConfig

    # 1. 创建RLT配置
    config = RLTConfig()

    # 2. 创建使用RLT策略的Policy实例
    policy = Policy(
        model,  # 你的ACoT-VLA模型
        refinement_strategy="rlt",
        rlt_config=config,
        rlt_network_path="./checkpoints/rlt/rlt_complete.pkl"
    )

    # 3. 使用Policy进行推理
    outputs = policy.infer(observation)

    # 或者为特定任务单独设置
    policy.set_task_strategy("sorting_packages", "rlt")
        """)

        return rlt_policy

    except Exception as e:
        logger.error(f"Failed to load RLT policy: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(description="RLT Workflow Example")
    parser.add_argument(
        "--mode",
        choices=["train", "infer", "both"],
        default="both",
        help="运行模式：train（仅训练）、infer（仅推理）、both（两者都执行）"
    )
    parser.add_argument(
        "--offline_steps",
        type=int,
        default=500,
        help="离线训练步数"
    )
    parser.add_argument(
        "--online_steps",
        type=int,
        default=1000,
        help="在线训练步数"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/rlt",
        help="模型输出目录"
    )

    args = parser.parse_args()

    config = RLTConfig(
        offline_steps=args.offline_steps,
        online_steps=args.online_steps,
    )

    output_dir = Path(args.output_dir)

    if args.mode in ["train", "both"]:
        encoder, decoder = train_rlt_offline(config, output_dir)
        _ = train_rlt_online(config, encoder, decoder, output_dir)

    if args.mode in ["infer", "both"]:
        run_inference_with_rlt(config, output_dir / "rlt_complete.pkl")


if __name__ == "__main__":
    main()
