import numpy as np
from rlgym.rocket_league.common_values import ORANGE_TEAM, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z
from typing import Any

from rlgym.rocket_league.api import GameState

MAX_FIELD_DIST = np.sqrt((SIDE_WALL_X * 2) ** 2 + (BACK_WALL_Y * 2) ** 2 + (CEILING_Z) ** 2)


class Speed2Ball:
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
        car = state.cars[agent]
        if car.team_num == ORANGE_TEAM:
            phys = car.inverted_physics
            ball = state.inverted_ball
        else:
            phys = car.physics
            ball = state.ball

        norm = np.linalg.norm(phys.linear_velocity)
        if norm == 0:
            return 0

        vel_u = phys.linear_velocity / norm
        ball_vec = ball.position - phys.position
        ball_vec_u = ball_vec / np.linalg.norm(ball_vec)
        return np.dot(ball_vec_u, vel_u)
