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
        return gym.spaces.MultiDiscrete([3, 5, 5, 3, 2, 2, 2])

    def parse_actions(self, actions: dict[str, th.Tensor], state: GameState) -> dict[str, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            parsed_actions[agent] = (
                np.array(
                    [
                        self.throttle[action[0]],
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


class SeerContinuousAction:
    def __init__(self, repeats: int = 8):
        self.repeats = repeats

    def get_action_space(self, agent: str) -> gym.Space:
        return gym.spaces.Box(-1, 1, shape=(7,))

    def parse_actions(self, actions: dict[str, np.ndarray], state: GameState) -> dict[str, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            # Apply the steer action to yaw as well
            action = np.insert(action, 3, action[1])
            # Convert floats to bools for jump boost and handbreak
            action[-3:] = action[-3:] > 0
            parsed_actions[agent] = action.reshape(1, -1).repeat(8, axis=0)

        return parsed_actions
