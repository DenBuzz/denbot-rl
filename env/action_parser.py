from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space
from ray.rllib.utils.typing import AgentID

from env.env_components import ActionParser


class SeerAction(ActionParser):
    throttle = roll = [-1, 0, 1]
    steer_yaw = pitch = [-1, -0.5, 0, 0.5, 1]
    jump = boost = handbreak = [0, 1]

    def __init__(self, repeats: int = 8):
        self.repeats = repeats

    @property
    def action_space(self) -> dict[AgentID, Space]:
        # assuming 3 agents per team, all seer spaces.
        # Also nutty comprehension
        seer_space = gym.spaces.MultiDiscrete([3, 5, 5, 3, 2, 2, 2])
        return {f"{side}-{i}": seer_space for i in range(3) for side in ["blue", "orange"]}

    def parse_action(self, action_dict: dict[AgentID, np.ndarray]) -> Any:
        parsed_actions = {}
        for agent, action in action_dict.items():
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


class SeerContinuousAction(ActionParser):
    def __init__(self, repeats: int = 8):
        self.repeats = repeats

    @property
    def action_space(self) -> dict[AgentID, Space]:
        # assuming 3 agents per team, all seer spaces.
        # Also nutty comprehension
        seer_space = gym.spaces.Box(-1, 1, shape=(7,))
        return {f"{side}-{i}": seer_space for i in range(3) for side in ["blue", "orange"]}

    def parse_action(self, action_dict: dict[AgentID, np.ndarray]) -> Any:
        parsed_actions = {}
        for agent, action in action_dict.items():
            # Apply the steer action to yaw as well
            action = np.insert(action, 3, action[1])
            # Convert floats to bools for jump boost and handbreak
            action[-3:] = action[-3:] > 0
            parsed_actions[agent] = action.reshape(1, -1).repeat(8, axis=0)
        return parsed_actions
