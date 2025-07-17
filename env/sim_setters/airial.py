import numpy as np
import rlgym.rocket_league.common_values as cv
import RocketSim as rsim
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.sim import RocketSimEngine
from scipy.stats import triang

from env.sim_setters.state_mutator import StateMutator


class AirialState(StateMutator):
    CURRICULUM_STEPS = 100

    BALL_START_HEIGHT = cv.BALL_RESTING_HEIGHT + 35
    BALL_HEIGHT_STEP = ((cv.CEILING_Z - cv.BALL_RADIUS) - BALL_START_HEIGHT) / CURRICULUM_STEPS

    CAR_START_HEIGHT = 34
    CAR_HEIGHT_STEP = ((cv.CEILING_Z - 4 * cv.BALL_RADIUS) - CAR_START_HEIGHT) / CURRICULUM_STEPS

    CAR_YEET_STEP = (cv.CAR_MAX_SPEED - 500) / CURRICULUM_STEPS

    def __init__(
        self,
        blue_size: int = 1,
        orange_size: int = 0,
        min_ball_height: float = cv.BALL_RESTING_HEIGHT,
        max_ball_height: float = 10 * cv.BALL_RESTING_HEIGHT,
        min_car_height: float = 34.0,
        max_car_height: float = 10 * 34.0,
        max_car_yeet: float = 1000,
    ) -> None:
        super().__init__()
        self.blue_size = blue_size
        self.orange_size = orange_size
        self.min_ball_height = min_ball_height
        self.max_ball_height = max_ball_height
        self.min_car_height = min_car_height
        self.max_car_height = max_car_height
        self.max_car_yeet = max_car_yeet

    def reset(self, info: dict) -> None:
        task = info.get("task", 0)
        self.max_ball_height = self.BALL_START_HEIGHT + task * self.BALL_HEIGHT_STEP
        self.max_car_height = self.CAR_START_HEIGHT + task * self.CAR_HEIGHT_STEP
        self.max_car_yeet = 1 + task * self.CAR_YEET_STEP

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
        # Apply no mass
        mutator_config = rsim.MutatorConfig()
        mutator_config.ball_mass = 0
        sim._arena.set_mutator_config(mutator_config)
        ball_height = triang.rvs(
            0.75,
            loc=self.min_ball_height,
            scale=self.max_ball_height - self.min_ball_height,
            random_state=self.rng,
        )

        state.ball.position = np.array(
            [
                (self.rng.random() - 0.5) * 2 * (cv.SIDE_WALL_X - 10 * cv.BALL_RADIUS),
                (self.rng.random() - 0.5) * 2 * (cv.BACK_WALL_Y - 10 * cv.BALL_RADIUS),
                ball_height,
            ],
            dtype=np.float32,
        )
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)

        car = self.default_car()
        x_max = cv.SIDE_WALL_X - 10 * cv.BALL_RADIUS
        y_max = cv.BACK_WALL_Y - 10 * cv.BALL_RADIUS
        car.physics.position = np.array(
            [
                self.rng.uniform(-x_max, x_max),
                self.rng.uniform(-y_max, y_max),
                self.rng.uniform(self.min_car_height, self.max_car_height),
            ]
        )
        car.physics.linear_velocity = self.rng.uniform(low=0, high=self.max_car_yeet, size=3)
        car.physics.euler_angles = self.rng.uniform(low=0, high=2 * np.pi, size=3)

        state.cars["blue-0"] = car
