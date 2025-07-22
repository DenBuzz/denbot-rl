import random

from gymnasium.vector import SyncVectorEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType

from env.curriculum.scenario import Scenario


class CurriculumManager:
    def __init__(self, tasks: list[list[Scenario]]):
        self._tasks: list[list[Scenario]] = tasks
        self._current_task = 0
        print(f"Scenario-based curriculum initialized with:\n{self._tasks}")

    def set_task(self, task):
        self._current_task = task

    def get_current_task_scenarios(self) -> list[Scenario]:
        """Returns the list of scenarios for the current curriculum stage."""
        return self._tasks[self._current_task]

    def sample_scenario(self) -> Scenario:
        """ENV_RUNNER: on_episode_created was called, choose a scenario."""
        scenario = random.choice(self.get_current_task_scenarios())
        self._active_scenario = scenario
        return scenario

    def record_episode(self, env, episode, prev_episode_chunks, metrics_logger: MetricsLogger):
        """ENV_RUNNER: on_episode_end was called, use the current scenario to log score."""
        score = self._active_scenario.evaluate_score(env, episode, prev_episode_chunks)
        metric_name = self.get_scenario_key(self._active_scenario)
        metrics_logger.log_value(metric_name, value=score)

    def should_promote(self, result: dict) -> bool:
        """DRIVER: on_train_result was called, checks if all scenarios
        in the current stage are above their score_threshold."""
        promote = True

        for scenario in self.get_current_task_scenarios():
            if scenario.score_threshold is None:
                continue  # no threshold set, always ready to promote
            score = result["env_runners"].get(self.get_scenario_key(scenario), -float("inf"))
            if score < scenario.score_threshold:
                promote = False
                break

        return promote

    def promote(self) -> None:
        """Advances to the next stage and resets progress for the next set of scenarios."""
        self._current_task += 1
        print(f"TASK PROMOTED! Now on task {self._current_task}.")

    def get_scenario_key(self, scenario: Scenario) -> str:
        return f"task-{self._current_task}-{scenario.name}"


def CurriculumCallback(curriculum_manager: CurriculumManager):
    class BuiltCurriculumCallback(CurriculumCallback_):
        def __init__(self) -> None:
            super().__init__(curriculum_manager=curriculum_manager)

    return BuiltCurriculumCallback


class CurriculumCallback_(RLlibCallback):
    """RLlib callbacks are quirky. Because env_runners exist on their own separate processes,
    the callbacks that they use will be different instances than the one used on the main learner.
    Basically, on_train_result will use a different instance of the curriculum_manager than the
    on_episode_* callbacks. To solve this issue, we need to sync up the repective managers whenever
    there's a change to their state. Specifically, when the task changes.

    To reduce how often we need to manually sync the managers, we should leverage a more global
    state as much as possible. Specifically things like the metrics_logger."""

    def __init__(self, *args, curriculum_manager: CurriculumManager, **kwargs) -> None:
        self.curriculum_manager = curriculum_manager
        super().__init__()

    def on_train_result(
        self, *, algorithm: Algorithm, metrics_logger: MetricsLogger | None = None, result: dict, **kwargs
    ) -> None:
        if self.curriculum_manager.should_promote(result):
            self.curriculum_manager.promote()

            def update_env_task(env_runner: EnvRunner):
                if env_runner.env is None:
                    return
                env_runner.env.call("set_task", self.curriculum_manager._current_task)

            assert algorithm.env_runner_group is not None
            algorithm.env_runner_group.foreach_env_runner(update_env_task)

        result["curriculum_task"] = self.curriculum_manager._current_task

    def on_episode_created(self, *, env: SyncVectorEnv, **kwargs) -> None:
        """ENV_RUNNER: Important to use on_episode_created because it is called before the env's reset!
        This allows us to modify the attributes of the environment before the reset to
        ensure that they will take effect."""
        # Update the manager to reflect the env's current task.
        # The env's task is what get's updated from on_train_result
        self.curriculum_manager.set_task(env.call("current_task")[0])
        scenario = self.curriculum_manager.sample_scenario()
        env.call("load_config", scenario.env_config)

    def on_episode_end(
        self,
        *,
        episode: EpisodeType | EpisodeV2,
        prev_episode_chunks: list[EpisodeType] | None = None,
        metrics_logger: MetricsLogger | None = None,
        env: SyncVectorEnv,
        env_index: int,
        **kwargs,
    ) -> None:
        """ENV_RUNNER: log episode results to the metrics_logger.
        The env_runner's specific curriculum_manager knows what the active scenario is."""
        assert metrics_logger is not None
        self.curriculum_manager.record_episode(env.envs[env_index].env, episode, prev_episode_chunks, metrics_logger)
