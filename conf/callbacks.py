from functools import partial

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback

# Import psutil after ray so the packaged version is used.
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
)
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID
from rlgym.rocket_league.common_values import BALL_RADIUS, BALL_RESTING_HEIGHT, CEILING_Z

from env.state_mutators.airial_curriculum import AirialTraining

CURRICULUM_STEPS = 100
BALL_HEIGHT_STEP = ((CEILING_Z - BALL_RADIUS) - BALL_RESTING_HEIGHT) / CURRICULUM_STEPS
CAR_HEIGHT_STEP = ((CEILING_Z - BALL_RADIUS) - 25) / CURRICULUM_STEPS
CAR_YEET_STEP = 2000 / CURRICULUM_STEPS


def _remote_fn(env_runner, new_task: int):
    # We recreate the entire env object by changing the env_config on the worker,
    # then calling its `make_env()` method.
    # env_runner.config.environment(env_config={"desc": new_task})
    # env_runner.make_env()
    for env in env_runner.env.envs:
        env.env.state_mutator = AirialTraining(
            max_ball_height=BALL_RESTING_HEIGHT + BALL_HEIGHT_STEP * new_task,
            max_car_height=25 + CAR_HEIGHT_STEP * new_task,
            max_car_yeet=CAR_YEET_STEP * new_task,
        )


def AirialCurriculumCallbackFactory():
    return AirialCurriculumCallback


class AirialCurriculumCallback(RLlibCallback):
    def on_episode_end(
        self,
        *,
        episode: EpisodeType | EpisodeV2,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger | None = None,
        env: gym.Env | None = None,
        env_index: int,
        rl_module: RLModule | None = None,
        worker: EnvRunner | None = None,
        base_env: BaseEnv | None = None,
        policies: dict[PolicyID, Policy] | None = None,
        **kwargs,
    ) -> None:
        # Hate this
        for env in env.envs:
            my_env = env.env
            ball_touches = np.array([car.ball_touches for car in my_env.state.cars.values()])
            metrics_logger.log_value("ball_touched", int(any(ball_touches > 0)), reduce="mean", clear_on_reduce=True)
        return super().on_episode_end(
            episode=episode,
            env_runner=env_runner,
            metrics_logger=metrics_logger,
            env=env,
            env_index=env_index,
            rl_module=rl_module,
            worker=worker,
            base_env=base_env,
            policies=policies,
            **kwargs,
        )

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        current_task = algorithm._counters["current_task"]
        metrics_logger.log_value("current_task", current_task, reduce="max")
        if current_task >= CURRICULUM_STEPS:
            print("At the highest task!!!")
            return

        current_touches = result[ENV_RUNNER_RESULTS]["ball_touched"]
        if current_touches > 0.95:
            new_task = current_task + 1
            print(f"Switching task on all EnvRunners to #{new_task}")
            algorithm.env_runner_group.foreach_env_runner(func=partial(_remote_fn, new_task=new_task))
            algorithm._counters["current_task"] = new_task
