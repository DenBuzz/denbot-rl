from typing import Any

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y, CEILING_Z, ORANGE_TEAM, SIDE_WALL_X

MAX_FIELD_DIST = np.sqrt((SIDE_WALL_X * 2) ** 2 + (BACK_WALL_Y * 2) ** 2 + (CEILING_Z) ** 2)


class FacingBall:
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

        ball_vec = ball.position - phys.position
        ball_vec_u = ball_vec / np.linalg.norm(ball_vec)
        return np.dot(phys.forward, ball_vec_u)
