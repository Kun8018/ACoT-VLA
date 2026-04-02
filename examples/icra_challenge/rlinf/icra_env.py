# Copyright 2025 ACoT-VLA Authors.
# ICRA Challenge 任务环境适配器，供 RLinf 框架使用。
# 将 ICRA 挑战赛的三个任务（sorting_packages / pouring_workpiece / opening_door）
# 包装成 RLinf 兼容的 gym.Env 接口，内部使用 LIBERO 仿真环境。

import importlib
import logging
import math
import os
import sys
from typing import Any

import gym
import numpy as np
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

# ── ICRA 任务套件名称 → LIBERO 套件映射 ──────────────────────────────────────
ICRA_TASK_SUITE = "icra_challenge"

# 支持的任务名称
SUPPORTED_TASKS = [
    "sorting_packages",
    "pouring_workpiece",
    "opening_door",
]

# 各任务最大步数
TASK_MAX_STEPS = {
    "sorting_packages": 240,
    "pouring_workpiece": 240,
    "opening_door": 240,
}

# 各任务奖励计算函数（可根据实际环境替换）
DUMMY_DONE_KEYS = {
    "sorting_packages": "task_complete",
    "pouring_workpiece": "task_complete",
    "opening_door": "task_complete",
}


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """四元数转轴角（来自 robosuite）。"""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_icra_image(obs: dict, key: str = "agentview_image") -> np.ndarray:
    """从 LIBERO obs 中提取图像并翻转到标准方向。"""
    img = obs[key]
    return np.ascontiguousarray(img[::-1, ::-1])


def build_robot_state(obs: dict) -> np.ndarray:
    """从 LIBERO obs 构建机器人状态向量（7维: xyz + 轴角 + 夹爪）。"""
    pos = obs["robot0_eef_pos"]                           # shape (3,)
    quat = obs["robot0_eef_quat"]                         # shape (4,)
    axisangle = quat2axisangle(quat)                      # shape (3,)
    gripper = obs["robot0_gripper_qpos"][:1]              # shape (1,)
    return np.concatenate([pos, axisangle, gripper])      # shape (7,)


# ── 主环境类 ──────────────────────────────────────────────────────────────────

class ICRAChallengeEnv(gym.Env):
    """
    ICRA 挑战赛任务的 RLinf 兼容环境封装。

    cfg 字段（来自 Hydra config）:
        task_name      : str   - 任务名称，从 SUPPORTED_TASKS 中选择
        total_num_envs : int   - 并行环境数（由 RLinf 管理）
        max_episode_steps : int
        camera_heights : int   = 256
        camera_widths  : int   = 256
        seed           : int   = 0
        use_rel_reward : bool  = True
        is_eval        : bool  = False
        auto_reset     : bool  = False
        ignore_terminations : bool = False
        group_size     : int   = 1
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, cfg, num_envs: int, seed_offset: int,
                 total_num_processes: int, worker_info: Any):
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.seed = cfg.seed + seed_offset
        self.task_name = cfg.task_name
        self.group_size = cfg.group_size
        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = getattr(cfg, "use_rel_reward", True)
        self.max_episode_steps = cfg.max_episode_steps
        self._elapsed_steps = np.zeros(num_envs, dtype=np.int32)
        self._is_start = True

        assert self.task_name in SUPPORTED_TASKS, (
            f"不支持的任务: {self.task_name}，请从 {SUPPORTED_TASKS} 中选择"
        )

        self._init_libero_env(cfg)
        self._prev_step_reward = np.zeros(num_envs)

    # ── 初始化 LIBERO 环境 ────────────────────────────────────────────────────

    def _init_libero_env(self, cfg):
        """初始化底层 LIBERO 仿真环境（支持 libero / liberoplus / liberopro）。"""
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as e:
            raise ImportError(
                "请安装 LIBERO 环境: pip install libero\n"
                "或参考 https://github.com/Lifelong-Robot-Learning/LIBERO"
            ) from e

        # 映射 ICRA 任务名称到 LIBERO 任务套件
        suite_name = cfg.get("task_suite_name", "libero_spatial")
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[suite_name]()

        # 选取与 task_name 匹配的任务（task language 包含 task_name 关键词）
        self.task_ids = self._find_task_ids()
        logger.info(f"[ICRAChallengeEnv] {self.task_name}: 找到 {len(self.task_ids)} 个匹配任务")

        cam_h = getattr(cfg, "camera_heights", 256)
        cam_w = getattr(cfg, "camera_widths", 256)

        # 为每个并行环境创建 LIBERO 实例
        self._envs = []
        for i in range(self.num_envs):
            task_id = self.task_ids[i % len(self.task_ids)]
            task = self.task_suite.get_task(task_id)
            bddl_file = (
                f"{get_libero_path('bddl_files')}/"
                f"{task.problem_folder}/{task.bddl_file}"
            )
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                camera_heights=cam_h,
                camera_widths=cam_w,
            )
            env.seed(self.seed + i)
            self._envs.append(env)

        # 保存初始状态
        self._init_states = [
            self.task_suite.get_task_init_states(tid)
            for tid in self.task_ids
        ]
        self._episode_task_ids = np.array([
            self.task_ids[i % len(self.task_ids)] for i in range(self.num_envs)
        ])
        self._episode_trial_ids = np.zeros(self.num_envs, dtype=np.int32)

    def _find_task_ids(self) -> list[int]:
        """找到 task_suite 中与 task_name 关键词匹配的任务 ID 列表。"""
        matched = []
        for tid in range(self.task_suite.n_tasks):
            task = self.task_suite.get_task(tid)
            # 任务名称关键词匹配（忽略大小写）
            if self.task_name.replace("_", " ").lower() in task.language.lower():
                matched.append(tid)
        # 如果没有精确匹配，则使用所有任务
        if not matched:
            logger.warning(
                f"[ICRAChallengeEnv] 未找到与 '{self.task_name}' 匹配的任务，"
                "将使用套件中的全部任务。"
            )
            matched = list(range(self.task_suite.n_tasks))
        return matched

    # ── gym.Env 接口 ──────────────────────────────────────────────────────────

    def reset(self):
        """重置所有并行环境，返回初始观察。"""
        obs_list = []
        for i, env in enumerate(self._envs):
            env.reset()
            task_id = self._episode_task_ids[i]
            trial_id = int(self._episode_trial_ids[i])
            init_states = self._init_states[
                self.task_ids.index(task_id)
            ]
            trial_id = trial_id % len(init_states)
            obs = env.set_init_state(init_states[trial_id])
            obs_list.append(obs)
        self._elapsed_steps[:] = 0
        self._prev_step_reward[:] = 0.0
        return self._process_obs(obs_list)

    def step(self, actions: np.ndarray):
        """
        执行动作，返回 (obs, rewards, dones, infos)。
        actions: shape (num_envs, action_dim)
        """
        obs_list, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self._envs, actions)):
            obs, reward, done, info = env.step(action.tolist())
            self._elapsed_steps[i] += 1

            # 超时截断
            truncated = self._elapsed_steps[i] >= self.max_episode_steps
            if truncated and not done:
                done = True
                info["TimeLimit.truncated"] = True

            if self.use_rel_reward:
                rel_reward = reward - self._prev_step_reward[i]
                self._prev_step_reward[i] = reward
                reward = rel_reward

            if self.ignore_terminations:
                done = False

            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            if done and self.auto_reset:
                env.reset()
                task_id = self._episode_task_ids[i]
                trial_id = int(self._episode_trial_ids[i])
                init_states = self._init_states[
                    self.task_ids.index(task_id)
                ]
                trial_id = trial_id % len(init_states)
                obs_list[-1] = env.set_init_state(init_states[trial_id])
                self._elapsed_steps[i] = 0
                self._prev_step_reward[i] = 0.0
                # 切换到下一个 trial
                self._episode_trial_ids[i] = (trial_id + 1) % len(init_states)

        return (
            self._process_obs(obs_list),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    def render(self, mode="rgb_array"):
        imgs = [get_icra_image(env.env._get_observations()) for env in self._envs]
        return np.stack(imgs)

    def close(self):
        for env in self._envs:
            env.close()

    # ── 观察处理 ─────────────────────────────────────────────────────────────

    def _process_obs(self, obs_list: list[dict]) -> dict:
        """
        将 LIBERO obs 列表转换为 RLinf 期望的批量字典格式。
        返回字段与 openpi LiberoInputs 变换兼容：
            obs["agentview_image"]           : (N, H, W, 3) uint8
            obs["robot0_eye_in_hand_image"]  : (N, H, W, 3) uint8
            obs["robot_state"]               : (N, 7) float32
            obs["task_name"]                 : list[str]
        """
        agent_imgs, wrist_imgs, states = [], [], []
        for obs in obs_list:
            agent_imgs.append(get_icra_image(obs, "agentview_image"))
            wrist_imgs.append(get_icra_image(obs, "robot0_eye_in_hand_image"))
            states.append(build_robot_state(obs))

        return {
            "agentview_image": np.stack(agent_imgs),
            "robot0_eye_in_hand_image": np.stack(wrist_imgs),
            "robot_state": np.stack(states, dtype=np.float32),
            "task_name": [self.task_name] * self.num_envs,
        }

    @property
    def observation_space(self):
        h = getattr(self.cfg, "camera_heights", 256)
        w = getattr(self.cfg, "camera_widths", 256)
        return gym.spaces.Dict({
            "agentview_image": gym.spaces.Box(0, 255, (self.num_envs, h, w, 3), np.uint8),
            "robot0_eye_in_hand_image": gym.spaces.Box(0, 255, (self.num_envs, h, w, 3), np.uint8),
            "robot_state": gym.spaces.Box(-np.inf, np.inf, (self.num_envs, 7), np.float32),
        })

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, (self.num_envs, 7), np.float32)
