from typing import Any
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import SIDE_WALL_X, BALL_RADIUS, BACK_WALL_Y, BALL_RESTING_HEIGHT
import numpy as np


class RandomBallLocation:
    """
    A StateMutator that randomizes ball location.
    """

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        x = (self.rng.random() - 0.5) * 2 * (SIDE_WALL_X - 8 * BALL_RADIUS)
        y = (self.rng.random() - 0.5) * 2 * (BACK_WALL_Y - 8 * BALL_RADIUS)
        state.ball.position = np.array([x, y, BALL_RESTING_HEIGHT], dtype=np.float32)
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)


class RandomCarLocation:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        ball_pos = state.ball.position

        for car in state.cars.values():
            x = (self.rng.random() - 0.5) * 2 * (SIDE_WALL_X - 4 * BALL_RADIUS)
            y = (self.rng.random() - 0.5) * 2 * (BACK_WALL_Y - 4 * BALL_RADIUS)
            while np.linalg.norm(ball_pos[:2] - np.array([x, y])) < 2 * BALL_RADIUS:
                x = (self.rng.random() - 0.5) * 2 * (SIDE_WALL_X - 4 * BALL_RADIUS)
                y = (self.rng.random() - 0.5) * 2 * (BACK_WALL_Y - 4 * BALL_RADIUS)

            car.physics.position = np.array([x, y, 100])
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0, 0, 0], dtype=np.float32)
            car.boost_amount = 33.3
