#!/usr/bin/env python3
"""
RLT 核心功能测试脚本
测试 RLT 算法的各个核心组件是否正常工作
"""

import sys
from pathlib import Path

# 添加项目根目录到PATH，这样可以找到openpi模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np

from openpi.policies.rlt.modeling_rlt_jax import (
    MLP,
    RLTokenEncoder,
    RLTokenDecoder,
    RLTActor,
    RLTPolicy,
)
from openpi.policies.rlt.configuration_rlt import (
    RLTokenConfig,
    RLTActorConfig,
    RLTCriticConfig,
    RLTConfig,
)
from openpi.training.rl_algorithms.rlt.rlt_algorithm import (
    RLTCritic,
    RLTAlgorithm,
    TrainingStats,
)


class TestRLTComponents:
    """测试 RLT 核心组件"""

    def test_mlp(self):
        """测试 MLP 网络"""
        print("\nTesting MLP...")
        input_dim = 10
        hidden_dims = [20, 30]
        output_dim = 5

        mlp = MLP(input_dim, hidden_dims, output_dim)
        x = torch.randn(2, input_dim)
        output = mlp(x)

        assert output.shape == (2, output_dim), f"Expected (2, {output_dim}), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        print("✅ MLP test passed")

    def test_rl_token_encoder(self):
        """测试 RL-token 编码器"""
        print("\nTesting RLTokenEncoder...")
        config = RLTokenConfig(input_dim=2048, rl_token_dim=2048)

        encoder = RLTokenEncoder(
            config.input_dim,
            config.rl_token_dim,
            config.num_encoder_layers,
            config.num_heads,
            config.ff_dim,
        )

        # 测试输入输出形状
        batch_size = 2
        seq_length = 50
        input_emb = torch.randn(batch_size, seq_length, config.input_dim)
        output = encoder(input_emb)

        assert output.shape == (batch_size, config.rl_token_dim), f"Expected {config.rl_token_dim}, got {output.shape}"
        print("✅ RLTokenEncoder test passed")

    def test_rl_token_decoder(self):
        """测试 RL-token 解码器"""
        print("\nTesting RLTokenDecoder...")
        config = RLTokenConfig(input_dim=2048, rl_token_dim=2048)

        decoder = RLTokenDecoder(
            config.rl_token_dim,
            config.input_dim,
            config.num_decoder_layers,
            config.num_heads,
            config.ff_dim,
        )

        batch_size = 2
        seq_length = 50
        z_rl = torch.randn(batch_size, config.rl_token_dim)
        z_vla = torch.randn(batch_size, seq_length, config.input_dim)

        decoded = decoder(z_rl, z_vla)

        assert decoded.shape == (batch_size, seq_length, config.input_dim)
        print("✅ RLTokenDecoder test passed")

    def test_rlt_actor(self):
        """测试 RL 演员网络"""
        print("\nTesting RLTActor...")
        config = RLTConfig()
        state_dim = 2048 + 7
        action_chunk_dim = config.chunk_size * 7

        actor = RLTActor(state_dim, action_chunk_dim, config.actor.hidden_dims)

        batch_size = 2
        state = torch.randn(batch_size, state_dim)
        ref_action = torch.randn(batch_size, action_chunk_dim)

        output = actor(state, ref_action)

        assert output.shape == (batch_size, action_chunk_dim)
        print("✅ RLTActor test passed")

    def test_rlt_critic(self):
        """测试 RL 评论家网络"""
        print("\nTesting RLTCritic...")
        config = RLTConfig()
        state_dim = 2048 + 7
        action_chunk_dim = config.chunk_size * 7

        critic = RLTCritic(state_dim, action_chunk_dim, config.critic.hidden_dims)

        batch_size = 2
        state = torch.randn(batch_size, state_dim)
        action = torch.randn(batch_size, action_chunk_dim)

        output = critic(state, action)

        assert output.shape == (batch_size, 1)
        print("✅ RLTCritic test passed")

    def test_rlt_policy(self):
        """测试完整的 RLT 策略"""
        print("\nTesting RLTPolicy...")
        config = RLTConfig()

        policy = RLTPolicy(config, state_dim=7, action_dim=7)

        batch_size = 2
        vla_emb = torch.randn(batch_size, config.vla_chunk_size, config.rl_token.input_dim)
        proprio = torch.randn(batch_size, 7)
        ref_actions = torch.randn(batch_size, config.chunk_size, 7)

        output = policy(vla_emb, proprio, ref_actions)

        assert output.shape == (batch_size, config.chunk_size, 7)
        print("✅ RLTPolicy test passed")

    def test_rlt_algorithm_init(self):
        """测试 RLT 算法初始化"""
        print("\nTesting RLTAlgorithm initialization...")
        config = RLTConfig()
        policy = RLTPolicy(config, state_dim=7, action_dim=7)
        algorithm = RLTAlgorithm(policy, config)

        assert algorithm is not None
        print("✅ RLTAlgorithm initialization test passed")

    def test_training_stats(self):
        """测试训练统计收集"""
        print("\nTesting TrainingStats...")
        stats = TrainingStats()

        # 添加损失
        stats.losses["loss1"] = 0.5
        stats.losses["loss2"] = 0.3
        stats.extra["accuracy"] = 0.95
        stats.grad_norms["actor"] = 0.12
        stats.grad_norms["critic"] = 0.15

        log_dict = stats.to_log_dict()

        assert len(log_dict) == 5
        assert "loss1" in log_dict
        assert "loss2" in log_dict
        assert "actor_grad_norm" in log_dict
        assert "critic_grad_norm" in log_dict
        assert "accuracy" in log_dict
        print("✅ TrainingStats test passed")

    def test_policy_training_mode(self):
        """测试策略训练/评估模式切换"""
        print("\nTesting policy training mode...")
        config = RLTConfig()
        policy = RLTPolicy(config, state_dim=7, action_dim=7)

        policy.eval()
        assert not policy.training

        policy.train()
        assert policy.training
        print("✅ Policy training mode test passed")


if __name__ == "__main__":
    # 运行所有测试
    print("=" * 60)
    print("RLT Core Components Tests")
    print("=" * 60)

    test_runner = TestRLTComponents()

    try:
        test_runner.test_mlp()
        test_runner.test_rl_token_encoder()
        test_runner.test_rl_token_decoder()
        test_runner.test_rlt_actor()
        test_runner.test_rlt_critic()
        test_runner.test_rlt_policy()
        test_runner.test_rlt_algorithm_init()
        test_runner.test_training_stats()
        test_runner.test_policy_training_mode()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
