#!/usr/bin/env python3
"""
ICRA 挑战赛 RLinf RL 训练入口脚本。

使用方式：
    # 安装 RLinf
    pip install rlinf

    # 基础训练（sorting_packages 任务）
    cd /path/to/ACoT-VLA
    ICRA_RLINF_CFG_PATH=examples/icra_challenge/rlinf/config \
    python examples/icra_challenge/rlinf/train.py \
        --config-name icra_challenge_ppo_openpi \
        ++actor.model.model_path=/path/to/checkpoint \
        ++env.train.task_name=sorting_packages \
        ++env.eval.task_name=sorting_packages

    # 切换到成功率低的任务
    python examples/icra_challenge/rlinf/train.py \
        --config-name icra_challenge_ppo_openpi \
        ++actor.model.model_path=/path/to/checkpoint \
        ++env.train.task_name=pouring_workpiece \
        ++env.eval.task_name=pouring_workpiece

    # 多 GPU 训练（修改 cluster.num_nodes 和 component_placement）
    python examples/icra_challenge/rlinf/train.py \
        --config-name icra_challenge_ppo_openpi \
        ++cluster.num_nodes=2

集成工作流：
    1. 用 ACoT-VLA 跑评估，找到成功率低的任务
    2. 用本脚本对这些任务跑 RLinf PPO 训练
    3. 训练完成后 checkpoint 存储在 results/icra_rlinf/
    4. 用 scripts/serve_policy.py 加载 RL 微调后的 checkpoint 进行推理
① 评估找出低成功率任务
   python examples/libero/main.py ...

② 安装 RLinf
   pip install rlinf
   # 或 pip install git+https://github.com/RLinf/RLinf.git

③ 对失败任务做 RL 微调（PPO）
   ICRA_RLINF_CFG_PATH=examples/icra_challenge/rlinf/config \
   python examples/icra_challenge/rlinf/train.py \
       ++actor.model.model_path=/path/to/checkpoint \
       ++env.train.task_name=sorting_packages

④ 加载 RL 微调后的 checkpoint 推理
   python scripts/serve_policy.py \
       --checkpoint results/icra_rlinf/checkpoints/latest
架构说明
RLinf 用 PyTorch 训练（通过其内置的 OpenPi0ForRLActionPrediction 适配器加载 openpi 权重），icra_env.py 将你的 LIBERO 仿真环境包装成 RLinf 所需的 gym.Env 接口。训练时 RLinf 启动 Ray cluster，分布式运行 Rollout worker（生成交互数据）和 Actor worker（更新梯度），PPO kl_beta=0.01 防止偏离预训练策略太远。
"""

import json
import os
import sys

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

# 确保能找到 icra_env 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

mp.set_start_method("spawn", force=True)


def _register_icra_env():
    """将 ICRAChallengeEnv 注册到 RLinf 的环境工厂中。"""
    try:
        from rlinf.envs import register_env
        from icra_challenge.rlinf.icra_env import ICRAChallengeEnv
        register_env("icra_challenge", ICRAChallengeEnv)
    except ImportError:
        # rlinf 未安装时静默跳过（开发模式下有用）
        pass


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="icra_challenge_ppo_openpi",
)
def main(cfg) -> None:
    # 注册 ICRA 环境
    _register_icra_env()

    try:
        from rlinf.config import validate_cfg
    except ImportError as e:
        print(
            "\n[错误] 找不到 rlinf 包，请先安装：\n"
            "  pip install rlinf\n"
            "或从源码安装：\n"
            "  pip install git+https://github.com/RLinf/RLinf.git\n"
        )
        raise e

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    # ── 启动 RLinf 分布式训练 ─────────────────────────────────────────────
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Actor（模型训练节点）
    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=component_placement.get_strategy("actor"),
    )

    # Rollout（推理节点，生成 rollout 数据）
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=component_placement.get_strategy("rollout"),
    )

    # Env（仿真环境节点）
    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=component_placement.get_strategy("env"),
    )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()

    print(
        f"\n训练完成！模型检查点保存在: {cfg.runner.logger.log_path}\n"
        "使用以下命令加载微调后的模型进行推理：\n"
        "  python scripts/serve_policy.py "
        "--env libero "
        f"--checkpoint {cfg.runner.logger.log_path}/checkpoints/latest"
    )


if __name__ == "__main__":
    main()
