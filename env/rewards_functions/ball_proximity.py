import numpy as np
from rlgym.rocket_league.common_values import SIDE_WALL_X, BACK_WALL_Y, CEILING_Z
from typing import Any
import matplotlib.pyplot as plt

from rlgym.rocket_league.api import GameState

MAX_FIELD_DIST = np.sqrt((SIDE_WALL_X * 2) ** 2 + (BACK_WALL_Y * 2) ** 2 + (CEILING_Z) ** 2)


class BallProximityReward:
    """
    A RewardFunction that gives a reward of 1 if the agent touches the ball, 0 otherwise.
    """

    def reset(self, agents: list[str], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: list[str],
        state: GameState,
        is_terminated: dict[str, bool],
        is_truncated: dict[str, bool],
        shared_info: dict[str, Any],
    ) -> dict[str, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: str, state: GameState) -> float:
        agent_dist = np.linalg.norm(state.cars[agent].physics.position - state.ball.position)
        return self.distance_to_reward(agent_dist)

    @staticmethod
    def distance_to_reward(dist):
        return np.exp2(-dist / 2500) - 1


if __name__ == "__main__":
    distances = np.linspace(0, MAX_FIELD_DIST, 1000)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()
    ax.plot(distances, BallProximityReward.distance_to_reward(distances))
    ax.grid()
    plt.show()
