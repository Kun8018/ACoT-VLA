from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias, Optional
import copy
import flax
import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import optax
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy
logger = logging.getLogger(__name__)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        refinement_strategy: str = "simple",
        task_strategy_map: Optional[dict[str, str]] = None,
        rlinf_network_path: Optional[str] = None
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._refinement_strategy = refinement_strategy
        self._task_strategy_map = task_strategy_map or {}

        # 初始化 RLinF 网络
        self._rlinf_network = None
        if rlinf_network_path:
            try:
                with open(rlinf_network_path, 'rb') as f:
                    network_params = pickle.load(f)
                self._rlinf_network = RLinFNetwork(action_dim=32, rngs=nnx.Rngs(self._rng))
                nnx.update(self._rlinf_network, network_params)
                logger.info(f"RLinF network loaded from: {rlinf_network_path}")
            except Exception as e:
                logger.warning(f"Failed to load RLinF network: {e}")
                self._rlinf_network = None

    @override
    def infer(self, obs: dict) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {"state": inputs["state"]}
        result = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)

        if isinstance(result, dict):
            outputs.update(result)
        else:
            outputs["actions"] = result

        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return self.post_process(obs, outputs)

    def post_process(self, obs: dict, outputs: dict) -> dict:
        task_name_requiring_waist = ["sorting_packages", "sorting_packages_continuous"]
        task_name = jax.tree.map(lambda x: x, obs).get("task_name", None)

        if task_name is None:
            return outputs

        print(f"Policy inferring for task: {task_name}, with inference time: {outputs['policy_timing']['infer_ms']:.3f} ms")
        if task_name not in task_name_requiring_waist:
            outputs["actions"] = outputs["actions"][:, :16]
        else:
            raw_state = jax.tree.map(lambda x: x, obs).get("state", None)
            assert raw_state is not None
            outputs["actions"][:, 16:20] = raw_state[16:20]

        strategy = self._get_task_strategy(task_name)

        if strategy == "rlinf":
            logger.info(f"Using RLinF strategy for task: {task_name}")
            return self._rlinf_refine_actions(obs, outputs)
        else:
            logger.info(f"Using simple refinement strategy for task: {task_name}")
            outputs["actions"] = self._gaussian_smooth(outputs["actions"])

            if task_name == "sorting_packages":
                self._refine_sorting_packages(outputs)
            elif task_name == "pouring_workpiece":
                self._refine_pouring_workpiece(outputs)
            elif task_name == "opening_door":
                self._refine_opening_door(outputs)

            return outputs

    def _get_task_strategy(self, task_name: str) -> str:
        return self._task_strategy_map.get(task_name, self._refinement_strategy)

    def set_task_strategy(self, task_name: str, strategy: str):
        if strategy not in ["simple", "rlinf"]:
            raise ValueError(f"Invalid strategy: {strategy}")
        self._task_strategy_map[task_name] = strategy
        logger.info(f"Set strategy for task '{task_name}' to '{strategy}'")

    def set_global_strategy(self, strategy: str):
        if strategy not in ["simple", "rlinf"]:
            raise ValueError(f"Invalid strategy: {strategy}")
        self._refinement_strategy = strategy
        logger.info(f"Set global strategy to '{strategy}'")

    def get_task_strategy_map(self) -> dict:
        return self._task_strategy_map.copy()

    def _gaussian_smooth(self, action):
        kernel = np.exp(-np.arange(-2, 3)**2 / 2)
        kernel /= np.sum(kernel)
        smoothed = np.zeros_like(action)
        for dim in range(action.shape[-1]):
            smoothed[:, dim] = np.convolve(action[:, dim], kernel, mode='same')
        return smoothed

    def _refine_sorting_packages(self, outputs):
        actions = outputs["actions"]
        if actions.shape[-1] > 13:
            for i in range(len(actions)):
                if actions[i][13] < 0.3:
                    actions[i][13] = 0.4

        for i in range(1, len(actions)):
            for dim in range(6, 9):
                if abs(actions[i][dim] - actions[i-1][dim]) > 0.1:
                    actions[i][dim] = actions[i-1][dim] + np.sign(actions[i][dim] - actions[i-1][dim]) * 0.05

    def _refine_pouring_workpiece(self, outputs):
        actions = outputs["actions"]
        if actions.shape[-1] > 6:
            for i in range(len(actions)):
                if actions[i][6] < 0.5:
                    actions[i][6] = 0.6

        for i in range(len(actions)):
            if actions[i][1] > 0.2:
                actions[i][1] = 0.15

    def _refine_opening_door(self, outputs):
        actions = outputs["actions"]
        if actions.shape[-1] > 8:
            for i in range(len(actions)):
                target_wrist = [0.1, -0.2, 0.3]
                for j, t in enumerate(target_wrist):
                    if abs(actions[i][6 + j] - t) > 0.1:
                        actions[i][6 + j] = actions[i][6 + j] * 0.9 + t * 0.1

    def _rlinf_refine_actions(self, obs: dict, outputs: dict) -> dict:
        task_name = obs.get("task_name", None)
        raw_actions = outputs["actions"]

        if self._rlinf_network is None:
            logger.warning("RLinF network not initialized, falling back to simple refinement")
            return self._simple_refinement(outputs, task_name)

        state = obs.get("state", jnp.zeros(32))

        improved_actions = []
        for i in range(raw_actions.shape[0]):
            action = raw_actions[i:i+1]
            improved_action, quality = self._rlinf_network(state[None], action)
            improved_action = self._apply_physical_constraints(improved_action[0], task_name)
            improved_actions.append(improved_action)

        improved_actions = jnp.array(improved_actions)
        final_actions = self._gaussian_smooth(improved_actions)

        outputs["actions"] = final_actions
        return outputs

    def _apply_physical_constraints(self, action: jnp.ndarray, task_name: str) -> jnp.ndarray:
        action = jnp.clip(action, -1, 1)

        if task_name == "sorting_packages":
            if action.shape[-1] > 13:
                action = action.at[13].set(jnp.maximum(action[13], 0.4))
        elif task_name == "pouring_workpiece":
            if action.shape[-1] > 6:
                action = action.at[6].set(jnp.maximum(action[6], 0.5))

        return action

    def _simple_refinement(self, outputs: dict, task_name: str) -> dict:
        actions = outputs["actions"]
        actions = self._gaussian_smooth(actions)

        if task_name == "sorting_packages":
            self._refine_sorting_packages(outputs)
        elif task_name == "pouring_workpiece":
            self._refine_pouring_workpiece(outputs)
        elif task_name == "opening_door":
            self._refine_opening_door(outputs)

        outputs["actions"] = actions
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class RLinFNetwork(nnx.Module):
    """
    真正的 RLinF 网络：用于学习动作精细化的策略网络
    需要通过强化学习训练
    """
    def __init__(self, action_dim: int, rngs: nnx.Rngs):
        self.state_encoder_fc1 = nnx.Linear(action_dim, 128, rngs=rngs)
        self.state_encoder_fc2 = nnx.Linear(128, 256, rngs=rngs)

        self.quality_fc1 = nnx.Linear(256, 64, rngs=rngs)
        self.quality_fc2 = nnx.Linear(64, 32, rngs=rngs)
        self.quality_fc3 = nnx.Linear(32, 1, rngs=rngs)

        self.improvement_fc1 = nnx.Linear(256 + action_dim, 128, rngs=rngs)
        self.improvement_fc2 = nnx.Linear(128, 64, rngs=rngs)
        self.improvement_fc3 = nnx.Linear(64, action_dim, rngs=rngs)

    def __call__(self, state: jnp.ndarray, action: jnp.ndarray):
        state_feat = nnx.swish(self.state_encoder_fc1(state))
        state_feat = nnx.swish(self.state_encoder_fc2(state_feat))

        quality_feat = nnx.swish(self.quality_fc1(state_feat))
        quality_feat = nnx.swish(self.quality_fc2(quality_feat))
        quality = nnx.sigmoid(self.quality_fc3(quality_feat))

        combined_feat = jnp.concatenate([state_feat, action], axis=-1)
        delta_feat = nnx.swish(self.improvement_fc1(combined_feat))
        delta_feat = nnx.swish(self.improvement_fc2(delta_feat))
        delta_action = nnx.tanh(self.improvement_fc3(delta_feat))

        improved_action = action + delta_action
        return improved_action, quality


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy
        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:
        results = self._policy.infer(obs)
        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")
        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1
        np.save(output_path, np.asarray(data))
        return results
