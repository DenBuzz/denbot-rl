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
from rlgym.rocket_league.common_values import (
    BALL_RADIUS,
    BALL_RESTING_HEIGHT,
    CAR_MAX_SPEED,
    CEILING_Z,
)

from env.denbot_reward import DenBotReward
from env.state_mutators.airial import AirialState


class EpisodeData(RLlibCallback):
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
            metrics_logger.log_value("boost_difference", my_env.reward_fn.boost_difference, reduce="mean", clear_on_reduce=True)
            metrics_logger.log_value("max_ball_height", my_env.state_mutator.max_ball_height, reduce="mean", clear_on_reduce=True)
            metrics_logger.log_value("max_car_height", my_env.state_mutator.max_car_height, reduce="mean", clear_on_reduce=True)
            metrics_logger.log_value("max_car_yeet", my_env.state_mutator.max_car_yeet, reduce="mean", clear_on_reduce=True)
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


class AirialCurriculum(RLlibCallback):
    CURRICULUM_STEPS = 100
    BOOST_REWARD_STEPS = 50

    BALL_START_HEIGHT = BALL_RESTING_HEIGHT + 2
    BALL_HEIGHT_STEP = ((CEILING_Z - BALL_RADIUS) - BALL_START_HEIGHT) / CURRICULUM_STEPS

    CAR_START_HEIGHT = 34
    CAR_HEIGHT_STEP = ((CEILING_Z - 4 * BALL_RADIUS) - CAR_START_HEIGHT) / CURRICULUM_STEPS

    CAR_YEET_STEP = (CAR_MAX_SPEED - 500) / CURRICULUM_STEPS

    BOOST_START_REWARD = 1
    BOOST_END_REWARD = 0.1

    def on_algorithm_init(self, *, algorithm: "Algorithm", metrics_logger: MetricsLogger | None = None, **kwargs) -> None:
        new_task = 0

        def _remote_fn(env_runner):
            for env in env_runner.env.envs:
                env.env.set_task(new_task)
                airial_state: AirialState = env.env.state_mutator
                reward_fn: DenBotReward = env.env.reward_fn

                airial_state.max_ball_height = self.BALL_START_HEIGHT + new_task * self.BALL_HEIGHT_STEP
                airial_state.max_car_height = self.CAR_START_HEIGHT + new_task * self.CAR_HEIGHT_STEP
                airial_state.max_car_yeet = 1 + new_task * self.CAR_YEET_STEP
                reward_fn.boost_difference = np.clip(
                    1 - (new_task / self.BOOST_REWARD_STEPS) * (self.BOOST_START_REWARD - self.BOOST_END_REWARD),
                    0.1,
                    1,
                )

        print(f"Switching task on all EnvRunners to #{new_task}")
        algorithm.env_runner_group.foreach_env_runner(func=_remote_fn)
        return super().on_algorithm_init(algorithm=algorithm, metrics_logger=metrics_logger, **kwargs)

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        current_task = algorithm._counters["current_task"]
        result["curriculum_complete"] = 0.0
        metrics_logger.log_value("current_task", current_task, reduce="max")
        if current_task >= self.CURRICULUM_STEPS:
            print("At the highest task!!!")
            result["curriculum_complete"] = 1.0
            return

        current_touches = result[ENV_RUNNER_RESULTS].get("ball_touched", 0)
        if current_touches > 0.95:
            new_task = current_task + 1

            new_boost_reward = np.clip(1 - (new_task / self.BOOST_REWARD_STEPS) * (self.BOOST_START_REWARD - self.BOOST_END_REWARD), 0.1, 1)

            def _remote_fn(env_runner):
                for env in env_runner.env.envs:
                    env.env.set_task(new_task)
                    airial_state: AirialState = env.env.state_mutator
                    reward_fn: DenBotReward = env.env.reward_fn

                    airial_state.max_ball_height = self.BALL_START_HEIGHT + new_task * self.BALL_HEIGHT_STEP
                    airial_state.max_car_height = self.CAR_START_HEIGHT + new_task * self.CAR_HEIGHT_STEP
                    airial_state.max_car_yeet = 1 + new_task * self.CAR_YEET_STEP
                    reward_fn.boost_difference = new_boost_reward

            print(f"Switching task on all EnvRunners to #{new_task}")
            algorithm.env_runner_group.foreach_env_runner(func=_remote_fn)
            algorithm._counters["current_task"] = new_task
