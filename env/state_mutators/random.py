from typing import Any

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y, BALL_RADIUS, BALL_RESTING_HEIGHT, SIDE_WALL_X


class RandomBallLocation:
    """
    A StateMutator that randomizes ball location.
    """

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        state.ball.position = np.array(
            [
                (self.rng.random() - 0.5) * 2 * (SIDE_WALL_X - 40 * BALL_RADIUS),
                (self.rng.random() - 0.5) * 2 * (BACK_WALL_Y - 40 * BALL_RADIUS),
                self.rng.uniform(2 * BALL_RESTING_HEIGHT, 5 * BALL_RESTING_HEIGHT),
            ],
            dtype=np.float32,
        )
        state.ball.linear_velocity = np.array(
            [
                self.rng.normal(scale=200),
                self.rng.normal(scale=200),
                self.rng.normal(loc=400, scale=100),
            ]
        )
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)


class RandomCarLocation:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        ball_pos = state.ball.position

        for car in state.cars.values():
            car.physics.position = np.array(
                [
                    self.rng.normal(ball_pos[0], scale=30),
                    self.rng.normal(ball_pos[0], scale=30),
                    17.1,
                ]
            )
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0, 0, 0], dtype=np.float32)
            car.boost_amount = np.random.random() * 100
