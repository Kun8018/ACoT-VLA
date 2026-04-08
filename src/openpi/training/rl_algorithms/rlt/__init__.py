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
RLT (RL Token) algorithm implementation for ACoT-VLA.

Implements the two-stage training from "RL Token: Bootstrapping Online RL
with Vision-Language-Action Models" (Xu et al., Physical Intelligence, 2026).

Stage 1 (offline): Train RL-token encoder/decoder via reconstruction loss.
Stage 2 (online):  Train actor-critic with chunked TD, BC regularization,
                   reference-action pass-through, and reference-action dropout.
"""
from openpi.training.rl_algorithms.rlt.rlt_algorithm import (
    RLTCritic,
    RLTAlgorithm,
    TrainingStats,
)

__all__ = [
    "RLTCritic",
    "RLTAlgorithm",
    "TrainingStats",
]
