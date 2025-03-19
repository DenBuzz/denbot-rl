from typing import Any
import gymnasium as gym

import numpy as np

from rlgym.rocket_league.api import GameState


class RepeatAction:
    """
    A simple wrapper to emulate tick skip.

    Repeats every action for a specified number of ticks.
    """

    def __init__(self, parser, repeats=8):
        super().__init__()
        self.parser = parser
        self.repeats = repeats

    def get_action_space(self, agent: str) -> gym.Space:
        return self.parser.get_action_space(agent)

    def reset(
        self,
        agents: list[str],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        self.parser.reset(agents, initial_state, shared_info)

    def parse_actions(
        self, actions: dict[str, Any], state: GameState, shared_info: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        rlgym_actions = self.parser.parse_actions(actions, state, shared_info)
        repeat_actions = {}
        for agent, action in rlgym_actions.items():
            if action.shape == (8,):
                action = np.expand_dims(action, axis=0)
            elif action.shape != (1, 8):
                raise ValueError(f"Expected action to have shape (8,) or (1,8), got {action.shape}")

            repeat_actions[agent] = action.repeat(self.repeats, axis=0)

        return repeat_actions
