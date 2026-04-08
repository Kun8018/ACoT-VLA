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
RLT (RL Token) policy implementation for ACoT-VLA.

Reference: "RL Token: Bootstrapping Online RL with Vision-Language-Action Models"
(Xu et al., Physical Intelligence, 2026)
"""
from openpi.policies.rlt.configuration_rlt import (
    RLTActorConfig,
    RLTCriticConfig,
    RLTokenConfig,
    RLTConfig,
)
from openpi.policies.rlt.modeling_rlt_jax import (
    MLP,
    RLTActor,
    RLTokenDecoder,
    RLTokenEncoder,
    RLTPolicy,
)

__all__ = [
    "RLTokenConfig",
    "RLTActorConfig",
    "RLTCriticConfig",
    "RLTConfig",
    "MLP",
    "RLTokenEncoder",
    "RLTokenDecoder",
    "RLTActor",
    "RLTPolicy",
]
