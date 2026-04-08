"""
RLT 模型实现 - 与 flax.nnx 0.10.2 兼容
完全参考项目中 policy.py 的 RLinFNetwork 写法
"""
import math
from typing import Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.policies.rlt.configuration_rlt import RLTConfig


class MLP(nnx.Module):
    """简单的 MLP 网络 - 与 RLinFNetwork 写法一致"""
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, rngs: nnx.Rngs):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # 直接在 __init__ 中创建子模块，不使用 @nnx.compact
        prev_dim = input_dim
        self.layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            fc = nnx.Linear(prev_dim, hidden_dim, rngs=rngs)
            self.layers.append(fc)
            prev_dim = hidden_dim

        self.out_layer = nnx.Linear(prev_dim, output_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = nnx.swish(layer(x))
        x = self.out_layer(x)
        return x


class RLTokenEncoder(nnx.Module):
    """RL token 编码器"""
    def __init__(self, config: RLTConfig, rngs: nnx.Rngs):
        self.config = config
        rlt_config = config.rl_token

        # 直接创建参数，不使用 @nnx.compact
        self.rl_token_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, rlt_config.input_dim)) * 0.02
        )

        if rlt_config.input_dim != rlt_config.rl_token_dim:
            self.input_proj = nnx.Linear(rlt_config.input_dim, rlt_config.rl_token_dim, rngs=rngs)
        else:
            self.input_proj = nnx.identity

        # 简单的编码器 - 使用 MLP
        self.encoder_net = MLP(
            rlt_config.input_dim,
            [rlt_config.ff_dim],
            rlt_config.rl_token_dim,
            rngs=rngs
        )

    def __call__(self, vla_embeddings: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, dim = vla_embeddings.shape

        # 广播 rl_token 到批次大小
        rl_token = jnp.tile(self.rl_token_embedding, (batch_size, 1, 1))

        # 连接到 VLA 嵌入序列
        seq = jnp.concatenate([vla_embeddings, rl_token], axis=1)

        # 简单编码 - 平均池化 + MLP
        encoded = self.encoder_net(jnp.mean(seq, axis=1))
        return encoded


class RLTokenDecoder(nnx.Module):
    """RL token 解码器"""
    def __init__(self, config: RLTConfig, rngs: nnx.Rngs):
        self.config = config
        rlt_config = config.rl_token

        if rlt_config.rl_token_dim != rlt_config.input_dim:
            self.rl_proj = nnx.Linear(rlt_config.rl_token_dim, rlt_config.input_dim, rngs=rngs)
        else:
            self.rl_proj = nnx.identity

        # 简单的解码器 - 使用 MLP
        self.decoder_net = MLP(
            rlt_config.input_dim,
            [rlt_config.ff_dim],
            rlt_config.input_dim,
            rngs=rngs
        )

    def __call__(self, z_rl: jnp.ndarray, vla_embeddings: jnp.ndarray) -> jnp.ndarray:
        batch_size, seq_len, dim = vla_embeddings.shape

        # 投影 rl token
        rl_proj = self.rl_proj(z_rl)

        # 简单解码 - 广播到序列长度
        decoded = jnp.tile(rl_proj[:, jnp.newaxis, :], (1, seq_len, 1))
        decoded = self.decoder_net(decoded.reshape(batch_size * seq_len, -1))
        return decoded.reshape(batch_size, seq_len, -1)


class RLTActor(nnx.Module):
    """RLT 演员网络"""
    def __init__(self, config: RLTConfig, rngs: nnx.Rngs):
        self.config = config
        rlt_config = config.rl_token
        self.state_dim = rlt_config.rl_token_dim + 7  # state + proprio
        self.action_chunk_dim = config.chunk_size * 7

        self.actor_net = MLP(
            self.state_dim + self.action_chunk_dim,
            config.actor.hidden_dims,
            self.action_chunk_dim,
            rngs=rngs
        )

        self.log_std = math.log(config.actor.std)

    def __call__(self, state: jnp.ndarray, ref_action_chunk: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([state, ref_action_chunk], axis=-1)
        delta = self.actor_net(x)
        return ref_action_chunk + delta

    def get_std(self) -> float:
        return math.exp(self.log_std)


class RLTPolicy(nnx.Module):
    """RLT 策略"""
    def __init__(self, config: RLTConfig, rngs: nnx.Rngs):
        self.config = config
        rlt_config = config.rl_token

        self.rl_token_encoder = RLTokenEncoder(config, rngs=rngs)
        self.rl_token_decoder = RLTokenDecoder(config, rngs=rngs)
        self.actor = RLTActor(config, rngs=rngs)

    def __call__(self, vla_embeddings, proprio, ref_actions):
        batch_size, chunk_size, action_dim = ref_actions.shape

        z_rl = self.rl_token_encoder(vla_embeddings)
        state = jnp.concatenate([z_rl, proprio], axis=-1)

        ref_flat = ref_actions.reshape(batch_size, chunk_size * action_dim)
        refined_flat = self.actor(state, ref_flat)

        return refined_flat.reshape(batch_size, chunk_size, action_dim)
