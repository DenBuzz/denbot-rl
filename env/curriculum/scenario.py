from abc import ABC
from collections.abc import Callable
from typing import Any

from ray.rllib.evaluation.episode_v2 import EpisodeV2

from env.env import RLEnv

ScoringFunction = Callable[[RLEnv, EpisodeV2, list[EpisodeV2]], float]


class Scenario(ABC):
    """Defines a single, self-contained sub-task within a curriculum."""

    def __init__(
        self,
        name: str,
        env_config: dict[str, Any],
        score_threshold: float | None,
        scoring_function: ScoringFunction,
        min_episodes_for_mastery: int = 100,
    ):
        self.name = name
        self.env_config = env_config
        self.score_threshold = score_threshold
        self.scoring_function = scoring_function

    def evaluate_score(self, env, episode, prev_episode_chunks):
        """Runs the scoring function on a completed episode."""
        return self.scoring_function(env, episode, prev_episode_chunks)

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()
