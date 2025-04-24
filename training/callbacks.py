from typing import Any

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EVALUATION_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID

from env.env import RLEnv


class EpisodeData(RLlibCallback):
    def on_episode_end(
        self,
        *,
        episode: EpisodeType | EpisodeV2,
        env_runner: EnvRunner | None = None,
        metrics_logger: MetricsLogger,
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
            my_env: RLEnv = env.env
            env_info = my_env.shared_info

            metrics_logger.log_value(f"{env_info['env']}-env", env_info["task"], reduce="max")
            match env_info["env"]:
                case "airial":
                    ball_touches = np.array([car.ball_touches for car in my_env.state.cars.values()])
                    metrics_logger.log_value("airial_ball_touched", int(any(ball_touches > 0)), reduce="mean", ema_coeff=0.2)
                case "wall_air_dribble":
                    metrics_logger.log_value("wall_air_dribble_goal_scored", int(my_env.state.goal_scored), reduce="mean", ema_coeff=0.2)
                case "field_air_dribble":
                    metrics_logger.log_value("field_air_dribble_goal_scored", int(my_env.state.goal_scored), reduce="mean", ema_coeff=0.2)
                case "shooting":
                    metrics_logger.log_value("shooting_goal_scored", int(my_env.state.goal_scored), reduce="mean", ema_coeff=0.2)
                case "ball_hunt":
                    ball_touches = np.array([car.ball_touches for car in my_env.state.cars.values()])
                    metrics_logger.log_value("ball_hunt_ball_touched", int(any(ball_touches > 0)), reduce="mean", ema_coeff=0.2)

            # ball_touches = np.array([car.ball_touches for car in my_env.state.cars.values()])
            # metrics_logger.log_value("ball_touched", int(any(ball_touches > 0)), reduce="mean", clear_on_reduce=True)
            # metrics_logger.log_value("boost_difference", my_env.reward_fn.boost_difference, reduce="mean", clear_on_reduce=True)
            # metrics_logger.log_value("max_ball_height", my_env.state_mutator.max_ball_height, reduce="mean", clear_on_reduce=True)
            # metrics_logger.log_value("max_car_height", my_env.state_mutator.max_car_height, reduce="mean", clear_on_reduce=True)
            # metrics_logger.log_value("max_car_yeet", my_env.state_mutator.max_car_yeet, reduce="mean", clear_on_reduce=True)
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


class CurriculumCallback(RLlibCallback):
    curriculum_config: dict[str, Any]

    def on_train_result(self, *, algorithm: Algorithm, metrics_logger: MetricsLogger | None = None, result: dict, **kwargs) -> None:
        meta_task = algorithm._counters["meta_task"]
        task_envs = self.curriculum_config["tasks"][meta_task]["envs"]

        env_promotions = []
        env_completions = {}
        for env in task_envs:
            env_curriculum = self.curriculum_config["envs"][env]
            env_completions[env] = self._task_complete(metrics_logger, env, env_curriculum)
            if self._should_promote(metrics_logger, env, env_curriculum):
                env_promotions.append(env)
                metrics_logger.delete((EVALUATION_RESULTS, ENV_RUNNER_RESULTS, env_curriculum["metric"]["key"]))

        if all(env_completions.values()):
            print(f"Meta task with {task_envs} is complete!")
            meta_task += 1
            algorithm._counters["meta_task"] = meta_task

            def _remote_fn(env_runner):
                for env in env_runner.env.envs:
                    my_env: RLEnv = env.env
                    my_env.set_tasks(meta_task, {})

            algorithm.env_runner_group.foreach_env_runner(func=_remote_fn)
            algorithm.eval_env_runner_group.foreach_env_runner(func=_remote_fn)

        if env_promotions:
            print(f"Promoting: {env_promotions}")

            def _remote_fn(env_runner):
                for env in env_runner.env.envs:
                    my_env: RLEnv = env.env
                    for env in env_promotions:
                        my_env.env_tasks[env] += 1

            algorithm.env_runner_group.foreach_env_runner(func=_remote_fn)
            algorithm.eval_env_runner_group.foreach_env_runner(func=_remote_fn)

        return super().on_train_result(algorithm=algorithm, metrics_logger=metrics_logger, result=result, **kwargs)

    def _should_promote(self, metrics_logger: MetricsLogger, env: str, env_curriculum: dict[str, Any]) -> bool:
        env_metric = env_curriculum["metric"]
        metric = metrics_logger.peek((EVALUATION_RESULTS, ENV_RUNNER_RESULTS, env_metric["key"]), default=0)
        if metric is None:
            return False
        if metric >= env_metric["value"]:
            return True
        return False

    def _task_complete(self, metrics_logger: MetricsLogger, env: str, env_curriculum: dict[str, Any]) -> bool:
        if metrics_logger.peek((EVALUATION_RESULTS, ENV_RUNNER_RESULTS, f"{env}-env"), default=0) >= env_curriculum["max"]:
            print(f"{env} complete!")
            return True
        return False
