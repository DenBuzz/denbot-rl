import gymnasium as gym
from typing import Any
from rlgym.rocket_league.api import GameState
import numpy as np


class LookupTableAction:
    """
    World-famous discrete action parser which uses a lookup table to reduce the number of possible actions from 1944 to 90
    """

    def __init__(self):
        self._lookup_table = self.make_lookup_table()

    def get_action_space(self, agent: str) -> gym.Space:
        return gym.spaces.Discrete(len(self._lookup_table))

    def reset(self, agents: list[str], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def parse_actions(
        self, actions: dict[str, np.ndarray], state: GameState, shared_info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            # Action can have shape (Ticks, 1) or (Ticks)
            # print(action)
            # assert len(action.shape) == 1 or (len(action.shape) == 2 and action.shape[1] == 1)
            #
            # if len(action.shape) == 2:
            #     action = action.squeeze(1)

            parsed_actions[agent] = self._lookup_table[action]

        return parsed_actions

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])

        return np.array(actions)
