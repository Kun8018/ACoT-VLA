#!/usr/bin/env python3
"""
RLT策略集成测试脚本

用于验证Policy类对"rlt"策略的支持是否正常工作
"""

import logging
import tempfile
from pathlib import Path
import pickle
import jax
import jax.numpy as jnp

from openpi.policies.policy import Policy
from openpi.policies.rlt.modeling_rlt_jax import RLTPolicy
from openpi.policies.rlt.configuration_rlt import RLTConfig
from openpi.training.pretrained import get_pretrained_model_config
from openpi.training.config import TrainConfig


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_rlt_policy_creation():
    """测试RLT策略创建和初始化"""
    logger.info("Test 1: Testing RLT policy creation")

    try:
        config = get_pretrained_model_config("acot_icra_simulation_challenge_reasoning_to_action")
        assert config is not None, "Failed to get model config"

        logger.info(f"Model config loaded: {config.model.__class__.__name__}")
        logger.info("✅ Model config test passed")

        # 创建策略实例 (没有实际加载检查点，仅用于测试初始化)
        policy = Policy(
            None,  # 实际使用中这里会是真实模型
            refinement_strategy="rlt",
            rlt_config=RLTConfig(),
        )
        assert policy is not None, "Failed to create Policy instance"

        logger.info("✅ Policy creation test passed")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_rlt_strategy_methods():
    """测试策略方法是否支持rlt选项"""
    logger.info("\nTest 2: Testing strategy methods with RLT")

    try:
        config = get_pretrained_model_config("acot_icra_simulation_challenge_reasoning_to_action")
        policy = Policy(None, refinement_strategy="simple")

        # 测试策略切换
        policy.set_task_strategy("sorting_packages", "rlt")
        assert policy._task_strategy_map["sorting_packages"] == "rlt"
        logger.info("✅ Task strategy switching test passed")

        policy.set_global_strategy("rlt")
        assert policy._refinement_strategy == "rlt"
        logger.info("✅ Global strategy switching test passed")

        # 测试策略获取
        task_strategy = policy._get_task_strategy("sorting_packages")
        assert task_strategy == "rlt"
        logger.info("✅ Task strategy lookup test passed")

        global_strategy = policy._get_task_strategy("pouring_workpiece")  # 没有单独配置
        assert global_strategy == "rlt"
        logger.info("✅ Global strategy fallback test passed")

        logger.info("✅ Strategy methods test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_rlt_network_loading():
    """测试RLT网络加载"""
    logger.info("\nTest 3: Testing RLT network loading")

    try:
        config = get_pretrained_model_config("acot_icra_simulation_challenge_reasoning_to_action")

        # 创建临时目录保存网络
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rlt_file = tmp_path / "rlt_network.pkl"

            # 创建一个简单的网络
            rlt_config = RLTConfig()
            rlt_policy = RLTPolicy(rlt_config, state_dim=7, action_dim=7)
            rlt_policy_state = nnx_utils.state(rlt_policy)

            # 保存网络
            with open(rlt_file, 'wb') as f:
                pickle.dump(rlt_policy_state, f)

            # 加载网络
            policy = Policy(
                None,
                refinement_strategy="rlt",
                rlt_config=rlt_config,
                rlt_network_path=str(rlt_file),
            )

            assert policy._rlt_network is not None, "RLT network not loaded"
            logger.info("✅ RLT network loading test passed")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_rlt_policy_integration():
    """综合测试RLT策略的使用场景"""
    logger.info("\nTest 4: Testing RLT policy integration")

    try:
        # 创建配置
        config = get_pretrained_model_config("acot_icra_simulation_challenge_reasoning_to_action")

        # 创建策略实例
        policy = Policy(
            None,
            refinement_strategy="simple",
            task_strategy_map={
                "sorting_packages": "rlt",
                "opening_door": "rlinf",
                "pouring_workpiece": "simple"
            }
        )

        logger.info("✅ Policy initialization with multi-task strategy map passed")

        logger.info("\n" + "=" * 60)
        logger.info("Strategy mapping test:")
        logger.info("=" * 60)
        logger.info(f"  sorting_packages    : {policy._get_task_strategy('sorting_packages')}")
        logger.info(f"  opening_door        : {policy._get_task_strategy('opening_door')}")
        logger.info(f"  pouring_workpiece   : {policy._get_task_strategy('pouring_workpiece')}")
        logger.info(f"  unknown_task        : {policy._get_task_strategy('unknown_task')}")

        logger.info("\n✅ RLT policy integration test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("RLT Policy Integration Tests")
    logger.info("=" * 60)

    tests = [
        test_rlt_policy_creation,
        test_rlt_strategy_methods,
        test_rlt_network_loading,
        test_rlt_policy_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"Test failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"Test results: {passed}/{total} passed")
    logger.info("=" * 60)

    if passed == total:
        logger.info("✅ All RLT policy integration tests passed!")
    else:
        logger.warning("⚠️  Some tests failed")
