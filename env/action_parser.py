import gymnasium as gym
import numpy as np
import torch as th
from rlgym.rocket_league.api import GameState


class SeerAction:
    throttle = roll = [-1, 0, 1]
    steer_yaw = pitch = [-1, -0.5, 0, 0.5, 1]
    jump = boost = handbreak = [0, 1]

    def __init__(self, repeats: int = 8):
        self.repeats = repeats

    def get_action_space(self, agent: str) -> gym.Space:
        return gym.spaces.MultiDiscrete([2, 5, 5, 3, 2, 2, 2])

    def parse_actions(self, actions: dict[str, th.Tensor], state: GameState) -> dict[str, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            parsed_actions[agent] = (
                np.array(
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
                .reshape(1, -1)
                .repeat(8, axis=0)
            )

        return parsed_actions
