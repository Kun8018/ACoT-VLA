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
RLT (RL Token) policy networks.

Reference: "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"
(Xu et al., Physical Intelligence, 2026)

Architecture:
  - RLTokenEncoder: compresses VLA token embeddings into a single compact RL token
  - RLTokenDecoder: reconstructs VLA embeddings from the RL token (Stage 1 training only)
  - RLTActor: refines VLA reference action chunks conditioned on (z_rl, proprioception, ref_action)
  - RLTCritic: Q(x, action_chunk) where x = (z_rl, proprioception)
  - RLTPolicy: bundles RL-token modules + actor into a policy for inference
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from openpi.policies.rlt.configuration_rlt import RLTConfig


# ── Building blocks ──────────────────────────────────────────────────


class MLP(nn.Module):
    """Simple feedforward network with ReLU activations."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ── RL Token Encoder ─────────────────────────────────────────────────


class RLTokenEncoder(nn.Module):
    """
    Compress VLA token embeddings into a single RL token via a small transformer.

    Appends a learnable `e_rl` embedding to the VLA token sequence, processes
    through transformer encoder layers, and returns the output at the `e_rl`
    position as the RL token `z_rl`.

    Paper Eq. 1: z_rl = g_phi([z_{1:M}, e_rl])_{M+1}
    """

    def __init__(
        self,
        input_dim: int,
        rl_token_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rl_token_dim = rl_token_dim

        self.e_rl = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        if input_dim != rl_token_dim:
            self.input_proj = nn.Linear(input_dim, rl_token_dim)
        else:
            self.input_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=rl_token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, z_vla: Tensor) -> Tensor:
        """
        Args:
            z_vla: VLA token embeddings, shape ``(B, M, D)``.

        Returns:
            RL token ``z_rl``, shape ``(B, rl_token_dim)``.
        """
        batch_size = z_vla.shape[0]
        e_rl = self.e_rl.expand(batch_size, -1, -1)
        seq = torch.cat([z_vla, e_rl], dim=1)  # (B, M+1, D)
        seq = self.input_proj(seq)
        out = self.transformer(seq)
        z_rl = out[:, -1, :]  # output at e_rl position
        return z_rl


# ── RL Token Decoder ─────────────────────────────────────────────────


class RLTokenDecoder(nn.Module):
    """
    Autoregressively reconstruct VLA embeddings from z_rl.

    Used only during Stage 1 (offline RL-token training).

    Paper Eq. 2: L_ro = E[sum_i || h(d([z_rl, z_bar_{1:i-1}]))_i - z_bar_i ||^2]
    """

    def __init__(
        self,
        rl_token_dim: int,
        output_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.output_dim = output_dim

        if rl_token_dim != output_dim:
            self.rl_proj = nn.Linear(rl_token_dim, output_dim)
        else:
            self.rl_proj = nn.Identity()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(output_dim, output_dim)

    def forward(self, z_rl: Tensor, z_vla_stopped: Tensor) -> Tensor:
        """
        Args:
            z_rl: RL token, shape ``(B, D_rl)``.
            z_vla_stopped: Stop-gradient VLA embeddings, shape ``(B, M, D)``.

        Returns:
            Reconstructed embeddings, shape ``(B, M, D)``.
        """
        seq_len = z_vla_stopped.shape[1]
        z_rl_proj = self.rl_proj(z_rl).unsqueeze(1)

        target = torch.cat([z_rl_proj, z_vla_stopped[:, :-1, :]], dim=1)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=z_rl.device)

        decoded = self.transformer(
            tgt=target,
            memory=z_rl_proj,
            tgt_mask=causal_mask,
        )
        return self.output_head(decoded)  # (B, M, D)


# ── Actor ────────────────────────────────────────────────────────────


class RLTActor(nn.Module):
    """
    Lightweight actor that refines VLA reference action chunks.

    Paper Eq. 4: pi_theta(a_{1:C} | x, a_tilde_{1:C}) = N(mu_theta(x, a_tilde), sigma^2 I)

    The actor is conditioned on both the RL state and the VLA's proposed action
    chunk, acting as a "VLA-guided action editor".
    """

    def __init__(self, state_dim: int, action_chunk_dim: int, hidden_dims: list[int], std: float = 0.1):
        super().__init__()
        input_dim = state_dim + action_chunk_dim
        self.net = MLP(input_dim, hidden_dims, action_chunk_dim)
        self.log_std = math.log(std)

    def forward(self, state: Tensor, ref_action_chunk: Tensor) -> Tensor:
        """
        Return the mean action chunk.

        Args:
            state: RL state (z_rl + proprioception), shape ``(B, D)``.
            ref_action_chunk: VLA's reference action chunk, shape ``(B, C*D_a)``.

        Returns:
            Refined action chunk, shape ``(B, C*D_a)``.
        """
        x = torch.cat([state, ref_action_chunk], dim=-1)
        delta = self.net(x)
        return ref_action_chunk + delta

    def get_std(self) -> float:
        """Get the standard deviation for the Gaussian policy."""
        return math.exp(self.log_std)


# ── Policy ──────────────────────────────────────────────────────────


class RLTPolicy(nn.Module):
    """
    RLT (RL Token) policy for inference.

    Combines RL-token encoder + actor into a single module that:
        1. Takes VLA embeddings and proprioception
        2. Extracts RL token
        3. Returns action chunk distribution over time
    """

    def __init__(self, config: RLTConfig, state_dim: int = 0, action_dim: int = 7):
        super().__init__()
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # RL-token encoder
        self.rl_token_encoder = RLTokenEncoder(
            input_dim=config.rl_token.input_dim,
            rl_token_dim=config.rl_token.rl_token_dim,
            num_layers=config.rl_token.num_encoder_layers,
            num_heads=config.rl_token.num_heads,
            ff_dim=config.rl_token.ff_dim,
            dropout=config.rl_token.dropout,
        )

        # RL-token decoder (for offline training only)
        self.rl_token_decoder = RLTokenDecoder(
            rl_token_dim=config.rl_token.rl_token_dim,
            output_dim=config.rl_token.input_dim,
            num_layers=config.rl_token.num_decoder_layers,
            num_heads=config.rl_token.num_heads,
            ff_dim=config.rl_token.ff_dim,
            dropout=config.rl_token.dropout,
        )

        # Actor
        self.actor = RLTActor(
            state_dim=config.rl_token.rl_token_dim + state_dim,
            action_chunk_dim=config.chunk_size * action_dim,
            hidden_dims=config.actor.hidden_dims,
            std=config.actor.std,
        )

        self._device = config.device
        self.to(self._device)

    def forward(
        self,
        vla_embeddings: Tensor,
        proprioception: Tensor,
        ref_action_chunks: Tensor,
    ) -> Tensor:
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
        # Encode VLA embeddings to RL token
        z_rl = self.rl_token_encoder(vla_embeddings)

        # Concatenate RL token with proprioception
        state = torch.cat([z_rl, proprioception], dim=-1)

        # Flatten action chunks for actor input
        batch_size, chunk_size, action_dim = ref_action_chunks.shape
        flattened_ref_actions = ref_action_chunks.reshape(batch_size, chunk_size * action_dim)

        # Generate refined action chunk
        flattened_refined_actions = self.actor(state, flattened_ref_actions)

        # Reshape back to chunk format
        return flattened_refined_actions.reshape(batch_size, chunk_size, action_dim)

    def encode_vla_to_rl_token(self, vla_embeddings: Tensor) -> Tensor:
        """
        Encode VLA embeddings to RL token (for training purposes).
        """
        return self.rl_token_encoder(vla_embeddings)

    def decode_rl_token_to_vla(self, z_rl: Tensor, z_vla_stopped: Tensor) -> Tensor:
        """
        Decode RL token back to VLA embeddings (for training purposes).
        """
        return self.rl_token_decoder(z_rl, z_vla_stopped)
