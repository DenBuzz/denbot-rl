import gymnasium as gym
from typing import Any
from rlgym.rocket_league.api import GameState
import numpy as np

import torch as th


class SeerActionParser:
    """
    World-famous discrete action parser which uses a lookup table to reduce the number of possible actions from 1944 to 90
    """

    def get_action_space(self, agent: str) -> gym.Space:
        return gym.spaces.MultiDiscrete([2, 5, 5, 3, 2, 2, 2])

    def reset(self, agents: list[str], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def parse_actions(
        self, actions: dict[str, th.Tensor], state: GameState, shared_info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            parsed_actions[agent] = np.insert(action, 3, action[1])

        return parsed_actions
