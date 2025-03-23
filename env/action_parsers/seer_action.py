import gymnasium as gym
from typing import Any
from rlgym.rocket_league.api import GameState
import numpy as np

import torch as th


class SeerActionParser:
    """
    World-famous discrete action parser which uses a lookup table to reduce the number of possible actions from 1944 to 90
    """

    throttle = roll = [-1, 0, 1]
    steer_yaw = pitch = [-1, -0.5, 0, 0.5, 1]
    jump = boost = handbreak = [0, 1]

    def get_action_space(self, agent: str) -> gym.Space:
        return gym.spaces.MultiDiscrete([2, 5, 5, 3, 2, 2, 2])

    def reset(self, agents: list[str], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def parse_actions(
        self, actions: dict[str, th.Tensor], state: GameState, shared_info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            parsed_actions[agent] = np.array(
                [
                    self.boost[action[0]],
                    self.steer_yaw[action[1]],
                    self.pitch[action[2]],
                    self.steer_yaw[action[1]],
                    self.roll[action[3]],
                    self.jump[action[4]],
                    self.boost[action[5]],
                    self.handbreak[action[6]],
                ]
            )

        return parsed_actions
