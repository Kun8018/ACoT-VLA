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
RLT (RL Token) algorithm.

Implements the two-stage training from "RL Token: Bootstrapping Online RL
with Vision-Language-Action Models" (Xu et al., Physical Intelligence, 2026).

Stage 1 (offline): Train RL-token encoder/decoder via reconstruction loss.
Stage 2 (online):  Train actor-critic with chunked TD, BC regularization,
                   reference-action pass-through, and reference-action dropout.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from openpi.policies.rlt.modeling_rlt import MLP, RLTPolicy
from openpi.policies.rlt.configuration_rlt import RLTConfig


@dataclass
class TrainingStats:
    """Returned by ``algorithm.update()`` for logging and checkpointing."""

    # Generic containers for all algorithms
    losses: dict[str, float] = field(default_factory=dict)
    grad_norms: dict[str, float] = field(default_factory=dict)
    extra: dict[str, float] = field(default_factory=dict)

    def to_log_dict(self) -> dict[str, float]:
        """Flatten all stats into a single dict for logging."""
        d: dict[str, float] = {}
        for name, val in self.losses.items():
            d[name] = val
        for name, val in self.grad_norms.items():
            d[f"{name}_grad_norm"] = val
        for name, val in self.extra.items():
            d[name] = val
        return d


class RLTCritic(nn.Module):
    """
    Q-function over (state, action_chunk) pairs.

    Paper Eq. 3: Q_psi(x, a_{1:C})

    Training-only component — lives on the algorithm side, not in the policy.
    """

    def __init__(self, state_dim: int, action_chunk_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = MLP(state_dim + action_chunk_dim, hidden_dims, output_dim=1)

    def forward(self, state: Tensor, action_chunk: Tensor) -> Tensor:
        x = torch.cat([state, action_chunk], dim=-1)
        return self.net(x)


class RLTAlgorithm:
    """
    RL Token: lightweight actor-critic on frozen VLA features.

    Owns the ``RLTPolicy`` (RL-token encoder/decoder + actor), a critic
    ensemble, and target networks.
    """

    def __init__(self, policy: RLTPolicy, config: RLTConfig):
        self.policy = policy
        self.config = config
        self.optimizers: dict[str, Optimizer] = {}
        self._optimization_step: int = 0
        self._device = config.device
        self._is_online = False

        self._init_critics()
        self._move_to_device()

    # ── Initialization ───────────────────────────────────────────────

    def _init_critics(self) -> None:
        state_dim = self.config.rl_token.rl_token_dim + self.policy.state_dim
        action_chunk_dim = self.config.chunk_size * self.policy.action_dim
        hidden_dims = self.config.critic.hidden_dims

        self.critics = torch.nn.ModuleList(
            [RLTCritic(state_dim, action_chunk_dim, hidden_dims) for _ in range(self.config.num_critics)]
        )
        self.critic_targets = torch.nn.ModuleList([copy.deepcopy(c) for c in self.critics])
        for ct in self.critic_targets:
            ct.requires_grad_(False)

    def _move_to_device(self) -> None:
        self.critics.to(self._device)
        self.critic_targets.to(self._device)

    # ── Offline phase (Stage 1): RL-token training ───────────────────

    def supports_offline_phase(self) -> bool:
        return True

    def offline_update(self, batch_iterator: Iterator[dict[str, Any]]) -> TrainingStats:
        """
        Train RL-token encoder/decoder on demonstration data.

        Paper Eq. 2: L_ro = E[ sum_i || h(d([z_rl, z_bar_{1:i-1}]))_i - z_bar_i ||^2 ]
        """
        batch = next(batch_iterator)

        vla_embeddings = batch["vla_embeddings"].to(self._device)
        z_vla = vla_embeddings.detach()  # stop-gradient on VLA embeddings

        z_rl = self.policy.encode_vla_to_rl_token(z_vla)
        z_reconstructed = self.policy.decode_rl_token_to_vla(z_rl, z_vla)

        loss_ro = F.mse_loss(z_reconstructed, z_vla)

        self.optimizers["rl_token"].zero_grad()
        loss_ro.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.rl_token_encoder.parameters()) + list(self.policy.rl_token_decoder.parameters()),
            max_norm=self.config.clip_grad_norm,
        )
        self.optimizers["rl_token"].step()

        self._optimization_step += 1
        return TrainingStats(losses={"loss_rl_token": loss_ro.item()})

    def transition_to_online(self) -> None:
        """Freeze RL-token modules; rebuild optimizers for actor-critic only."""
        self.policy.rl_token_encoder.requires_grad_(False)
        self.policy.rl_token_decoder.requires_grad_(False)
        self._is_online = True

        self.optimizers = {
            "actor": torch.optim.Adam(self.policy.actor.parameters(), lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critics.parameters(), lr=self.config.critic_lr),
        }
        self._optimization_step = 0

    # ── Online phase (Stage 2): Actor-Critic ─────────────────────────

    def update(self, batch_iterator: Iterator[dict[str, Any]]) -> TrainingStats:
        """
        One full RLT update step with UTD critic warm-up.

        Pulls ``utd_ratio`` batches. First ``utd_ratio - 1`` are critic-only;
        the last batch also updates the actor (every ``policy_update_freq`` steps).
        """
        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            fb = self._prepare_forward_batch(batch)
            self._critic_step(fb)
            self._update_target_networks()

        batch = next(batch_iterator)
        fb = self._prepare_forward_batch(batch)
        critic_loss = self._critic_step(fb)

        stats = TrainingStats(losses={"loss_critic": critic_loss})

        if self._optimization_step % self.config.policy_update_freq == 0:
            actor_loss, bc_loss, q_val = self._actor_step(fb)
            stats.losses["loss_actor"] = actor_loss
            stats.extra["bc_loss"] = bc_loss
            stats.extra["q_value_mean"] = q_val

        self._update_target_networks()
        self._optimization_step += 1
        return stats

    def _prepare_forward_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a replay batch into algorithm-ready tensors.

        Extracts RL-token from VLA embeddings, builds RL state, reads
        reference action from complementary_info.
        """
        vla_emb = batch["vla_embeddings"].to(self._device)
        next_vla_emb = batch["next_vla_embeddings"].to(self._device)
        proprio = batch.get("proprioception", torch.zeros(vla_emb.shape[0], self.policy.state_dim, device=self._device)).to(self._device)
        next_proprio = batch.get("next_proprioception", torch.zeros(next_vla_emb.shape[0], self.policy.state_dim, device=self._device)).to(self._device)
        ref_actions = batch["ref_actions"].to(self._device)
        next_ref_actions = batch["next_ref_actions"].to(self._device)
        actions = batch["actions"].to(self._device)
        rewards = batch["rewards"].to(self._device)
        dones = batch["dones"].to(self._device)

        with torch.no_grad():
            z_rl = self.policy.encode_vla_to_rl_token(vla_emb)
            z_rl_next = self.policy.encode_vla_to_rl_token(next_vla_emb)

        state = torch.cat([z_rl, proprio], dim=-1)
        next_state = torch.cat([z_rl_next, next_proprio], dim=-1)

        batch_size = vla_emb.shape[0]

        # Apply reference dropout
        if self.training and torch.rand(1).item() < self.config.ref_dropout:
            ref_actions = torch.zeros_like(ref_actions)

        return {
            "state": state,
            "next_state": next_state,
            "ref_actions": ref_actions.reshape(batch_size, -1),
            "next_ref_actions": next_ref_actions.reshape(batch_size, -1),
            "actions": actions.reshape(batch_size, -1),
            "rewards": rewards,
            "dones": dones,
        }

    def _critic_step(self, fb: dict[str, Any]) -> float:
        """Update critics with chunked TD-learning."""
        state = fb["state"]
        action = fb["actions"]
        next_state = fb["next_state"]
        next_ref_actions = fb["next_ref_actions"]
        rewards = fb["rewards"]
        dones = fb["dones"]

        # Current Q-values
        qs = [critic(state, action) for critic in self.critics]

        # Target Q-values (using target networks)
        with torch.no_grad():
            # Sample next action from policy
            next_action = self.policy.actor(next_state, next_ref_actions)
            # Use minimum of target critics to avoid overestimation
            next_qs = [critic_target(next_state, next_action) for critic_target in self.critic_targets]
            next_q = torch.min(torch.cat(next_qs, dim=-1), dim=-1, keepdim=True)[0]
            target_q = rewards + self.config.discount * (1 - dones) * next_q

        # Critic loss
        critic_loss = 0.0
        for q in qs:
            critic_loss += F.mse_loss(q, target_q)
        critic_loss /= len(qs)

        self.optimizers["critic"].zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=self.config.clip_grad_norm)
        self.optimizers["critic"].step()

        return critic_loss.item()

    def _actor_step(self, fb: dict[str, Any]) -> tuple[float, float, float]:
        """Update actor with policy gradient, BC regularization, and pass-through."""
        state = fb["state"]
        ref_actions = fb["ref_actions"]

        # Get current Q values for the policy action
        pi_action = self.policy.actor(state, ref_actions)

        # Use minimum of critics
        qs = [critic(state, pi_action) for critic in self.critics]
        q = torch.min(torch.cat(qs, dim=-1), dim=-1)[0]

        # Actor loss (maximize Q)
        actor_loss = -q.mean()

        # BC regularization
        bc_loss = F.mse_loss(pi_action, ref_actions)

        # Combined loss
        total_loss = actor_loss + self.config.bc_reg_coeff * bc_loss

        self.optimizers["actor"].zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=self.config.clip_grad_norm)
        self.optimizers["actor"].step()

        return total_loss.item(), bc_loss.item(), q.mean().item()

    def _update_target_networks(self) -> None:
        """Soft-update target networks with Polyak averaging."""
        tau = self.config.tau
        for critic, critic_target in zip(self.critics, self.critic_targets):
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
