# Copyright 2026 ACoT-VLA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
RLT (RL Token) policy networks - JAX/Flax implementation.

Reference: "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"
(Xu et al., Physical Intelligence, 2026)
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.policies.rlt.configuration_rlt import RLTConfig


# ── Building blocks ──────────────────────────────────────────────────


class MLP(nnx.Module):
    """Simple feedforward network with Swish activations (JAX/Flax version)."""

    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    activate_output: bool = True

    @nnx.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nnx.Linear(prev_dim, hidden_dim)(x)
            x = nnx.swish(x)
            prev_dim = hidden_dim
        x = nnx.Linear(prev_dim, self.output_dim)(x)
        if self.activate_output:
            x = nnx.swish(x)
        return x


# ── RL Token Encoder ─────────────────────────────────────────────────


class RLTokenEncoder(nnx.Module):
    """
    Compress VLA token embeddings into a single RL token via a small transformer.

    Appends a learnable `e_rl` embedding to the VLA token sequence, processes
    through transformer encoder layers, and returns the output at the `e_rl`
    position as the RL token `z_rl`.

    Paper Eq. 1: z_rl = g_phi([z_{1:M}, e_rl])_{M+1}
    """

    input_dim: int
    rl_token_dim: int
    num_layers: int
    num_heads: int
    ff_dim: int
    dropout: float = 0.0

    @nnx.compact
    def __call__(self, z_vla: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z_vla: VLA token embeddings, shape ``(B, M, D)``.

        Returns:
            RL token ``z_rl``, shape ``(B, rl_token_dim)``.
        """
        batch_size, seq_len, _ = z_vla.shape

        # Learnable RL token embedding
        e_rl = self.param(
            "e_rl",
            nnx.initializers.normal(stddev=0.02),
            (1, 1, self.input_dim),
        )
        e_rl = jnp.broadcast_to(e_rl, (batch_size, 1, self.input_dim))

        # Concatenate to input sequence
        seq = jnp.concatenate([z_vla, e_rl], axis=1)  # (B, M+1, D)

        # Input projection if needed
        if self.input_dim != self.rl_token_dim:
            seq = nnx.Linear(self.input_dim, self.rl_token_dim)(seq)

        # Transformer encoder
        for i in range(self.num_layers):
            seq = TransformerEncoderLayer(
                d_model=self.rl_token_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_dim,
                dropout=self.dropout,
                name=f"encoder_layer_{i}",
            )(seq)

        # Return output at e_rl position
        return seq[:, -1, :]


# ── RL Token Decoder ─────────────────────────────────────────────────


class RLTokenDecoder(nnx.Module):
    """
    Autoregressively reconstruct VLA embeddings from z_rl (JAX/Flax version).

    Used only during Stage 1 (offline RL-token training).
    """

    rl_token_dim: int
    output_dim: int
    num_layers: int
    num_heads: int
    ff_dim: int
    dropout: float = 0.0

    @nnx.compact
    def __call__(self, z_rl: jnp.ndarray, z_vla_stopped: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            z_rl: RL token, shape ``(B, D_rl)``.
            z_vla_stopped: Stop-gradient VLA embeddings, shape ``(B, M, D)``.

        Returns:
            Reconstructed embeddings, shape ``(B, M, D)``.
        """
        batch_size, seq_len, _ = z_vla_stopped.shape

        # RL token projection if needed
        if self.rl_token_dim != self.output_dim:
            z_rl_proj = nnx.Linear(self.rl_token_dim, self.output_dim)(z_rl)
        else:
            z_rl_proj = z_rl

        z_rl_proj = z_rl_proj[:, None, :]  # (B, 1, D)

        # Build target sequence: [z_rl, z_vla[0], z_vla[1], ..., z_vla[M-2]]
        target = jnp.concatenate([z_rl_proj, z_vla_stopped[:, :-1, :]], axis=1)

        # Transformer decoder layers
        seq = target
        for i in range(self.num_layers):
            seq = TransformerDecoderLayer(
                d_model=self.output_dim,
                nhead=self.num_heads,
                dim_feedforward=self.ff_dim,
                dropout=self.dropout,
                name=f"decoder_layer_{i}",
            )(seq, z_rl_proj)

        # Output head
        return nnx.Linear(self.output_dim, self.output_dim)(seq)


# ── Actor ────────────────────────────────────────────────────────────


class RLTActor(nnx.Module):
    """
    Lightweight actor that refines VLA reference action chunks (JAX/Flax version).

    The actor is conditioned on both the RL state and the VLA's proposed action
    chunk, acting as a "VLA-guided action editor".
    """

    state_dim: int
    action_chunk_dim: int
    hidden_dims: list[int]
    std: float = 0.1

    @nnx.compact
    def __call__(self, state: jnp.ndarray, ref_action_chunk: jnp.ndarray) -> jnp.ndarray:
        """
        Return the mean action chunk.

        Args:
            state: RL state (z_rl + proprioception), shape ``(B, D)``.
            ref_action_chunk: VLA's reference action chunk, shape ``(B, C*D_a)``.

        Returns:
            Refined action chunk, shape ``(B, C*D_a)``.
        """
        x = jnp.concatenate([state, ref_action_chunk], axis=-1)
        delta = MLP(
            input_dim=self.state_dim + self.action_chunk_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.action_chunk_dim,
            activate_output=False,
        )(x)
        return ref_action_chunk + delta


# ── Transformer Layers ──────────────────────────────────────────────


class TransformerEncoderLayer(nnx.Module):
    """Simple transformer encoder layer."""

    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float = 0.0
    name: str = ""

    @nnx.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # Self-attention
        attn = nnx.MultiHeadAttention(
            num_heads=self.nhead,
            in_features=self.d_model,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout,
        )
        attn_out = attn(x, x, x, mask=mask)
        x = x + attn_out
        x = nnx.LayerNorm()(x)

        # Feed-forward
        ff = MLP(
            input_dim=self.d_model,
            hidden_dims=[self.dim_feedforward],
            output_dim=self.d_model,
            activate_output=False,
        )
        ff_out = ff(x)
        x = x + ff_out
        x = nnx.LayerNorm()(x)

        return x


class TransformerDecoderLayer(nnx.Module):
    """Simple transformer decoder layer (simplified)."""

    d_model: int
    nhead: int
    dim_feedforward: int
    dropout: float = 0.0
    name: str = ""

    @nnx.compact
    def __call__(self, tgt: jnp.ndarray, memory: jnp.ndarray) -> jnp.ndarray:
        # Self-attention with causal mask
        batch_size, seq_len, _ = tgt.shape
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

        self_attn = nnx.MultiHeadAttention(
            num_heads=self.nhead,
            in_features=self.d_model,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout,
        )
        self_attn_out = self_attn(tgt, tgt, tgt, mask=causal_mask)
        x = tgt + self_attn_out
        x = nnx.LayerNorm()(x)

        # Cross-attention with memory
        cross_attn = nnx.MultiHeadAttention(
            num_heads=self.nhead,
            in_features=self.d_model,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=self.dropout,
        )
        cross_attn_out = cross_attn(x, memory, memory)
        x = x + cross_attn_out
        x = nnx.LayerNorm()(x)

        # Feed-forward
        ff = MLP(
            input_dim=self.d_model,
            hidden_dims=[self.dim_feedforward],
            output_dim=self.d_model,
            activate_output=False,
        )
        ff_out = ff(x)
        x = x + ff_out
        x = nnx.LayerNorm()(x)

        return x


# ── Policy ──────────────────────────────────────────────────────────


class RLTPolicy(nnx.Module):
    """
    RLT (RL Token) policy for inference (JAX/Flax version).

    Combines RL-token encoder + actor into a single module that:
        1. Takes VLA embeddings and proprioception
        2. Extracts RL token
        3. Returns action chunk distribution over time
    """

    config: RLTConfig
    state_dim: int = 0
    action_dim: int = 7

    @nnx.compact
    def __call__(
        self,
        vla_embeddings: jnp.ndarray,
        proprioception: jnp.ndarray,
        ref_action_chunks: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Inference call: takes VLA embeddings, proprioception, and reference action chunks,
        returns refined action chunks.

        Args:
            vla_embeddings: VLA embeddings from frozen backbone, shape ``(B, M, D)``.
            proprioception: Proprioceptive state, shape ``(B, D_p)``.
            ref_action_chunks: VLA reference action chunks, shape ``(B, C, D_a)``.

        Returns:
            Refined action chunks, shape ``(B, C, D_a)``.
        """
        # RL-token encoder
        rl_token_encoder = RLTokenEncoder(
            input_dim=self.config.rl_token.input_dim,
            rl_token_dim=self.config.rl_token.rl_token_dim,
            num_layers=self.config.rl_token.num_encoder_layers,
            num_heads=self.config.rl_token.num_heads,
            ff_dim=self.config.rl_token.ff_dim,
            dropout=self.config.rl_token.dropout,
        )

        # Actor
        actor = RLTActor(
            state_dim=self.config.rl_token.rl_token_dim + self.state_dim,
            action_chunk_dim=self.config.chunk_size * self.action_dim,
            hidden_dims=self.config.actor.hidden_dims,
            std=self.config.actor.std,
        )

        # Encode VLA embeddings to RL token
        z_rl = rl_token_encoder(vla_embeddings)

        # Concatenate RL token with proprioception
        state = jnp.concatenate([z_rl, proprioception], axis=-1)

        # Flatten action chunks for actor input
        batch_size, chunk_size, action_dim = ref_action_chunks.shape
        flattened_ref_actions = ref_action_chunks.reshape(batch_size, chunk_size * action_dim)

        # Generate refined action chunk
        flattened_refined_actions = actor(state, flattened_ref_actions)

        # Reshape back to chunk format
        return flattened_refined_actions.reshape(batch_size, chunk_size, action_dim)

    def encode_vla_to_rl_token(self, vla_embeddings: jnp.ndarray) -> jnp.ndarray:
        """Encode VLA embeddings to RL token."""
        rl_token_encoder = RLTokenEncoder(
            input_dim=self.config.rl_token.input_dim,
            rl_token_dim=self.config.rl_token.rl_token_dim,
            num_layers=self.config.rl_token.num_encoder_layers,
            num_heads=self.config.rl_token.num_heads,
            ff_dim=self.config.rl_token.ff_dim,
            dropout=self.config.rl_token.dropout,
        )
        return rl_token_encoder(vla_embeddings)
