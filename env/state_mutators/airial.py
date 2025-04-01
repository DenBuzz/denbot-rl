import numpy as np
import RocketSim as rsim

# Import psutil after ray so the packaged version is used.
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import (
    BACK_WALL_Y,
    BALL_RADIUS,
    BALL_RESTING_HEIGHT,
    BLUE_TEAM,
    OCTANE,
    ORANGE_TEAM,
    SIDE_WALL_X,
)
from rlgym.rocket_league.sim import RocketSimEngine
from scipy.stats import triang


class AirialState:
    def __init__(
        self,
        blue_size: int = 1,
        orange_size: int = 0,
        min_ball_height: float = BALL_RESTING_HEIGHT,
        max_ball_height: float = 10 * BALL_RESTING_HEIGHT,
        min_car_height: float = 34.0,
        max_car_height: float = 10 * 34.0,
        max_car_yeet: float = 1000,
    ) -> None:
        self.rng = np.random.default_rng()
        self.blue_size = blue_size
        self.orange_size = orange_size
        self.min_ball_height = min_ball_height
        self.max_ball_height = max_ball_height
        self.min_car_height = min_car_height
        self.max_car_height = max_car_height
        self.max_car_yeet = max_car_yeet

    def reset(self, info: dict):
        self._current_task = info["current_task"]

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
                (self.rng.random() - 0.5) * 2 * (SIDE_WALL_X - 10 * BALL_RADIUS),
                (self.rng.random() - 0.5) * 2 * (BACK_WALL_Y - 10 * BALL_RADIUS),
                ball_height,
            ],
            dtype=np.float32,
        )
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)

        assert len(state.cars) == 0  # This mutator doesn't support other team size mutators

        for idx in range(self.blue_size):
            car = self._new_car()
            car.team_num = BLUE_TEAM
            state.cars["blue-{}".format(idx)] = car

        for idx in range(self.orange_size):
            car = self._new_car()
            car.team_num = ORANGE_TEAM
            state.cars["orange-{}".format(idx)] = car

    def _new_car(self) -> Car:
        car = Car()
        car.hitbox_type = OCTANE

        car.physics = PhysicsObject()

        x_max = SIDE_WALL_X - 10 * BALL_RADIUS
        y_max = BACK_WALL_Y - 10 * BALL_RADIUS
        car.physics.position = np.array(
            [
                self.rng.uniform(-x_max, x_max),
                self.rng.uniform(-y_max, y_max),
                self.rng.uniform(self.min_car_height, self.max_car_height),
            ]
        )
        car.physics.linear_velocity = self.rng.uniform(low=0, high=self.max_car_yeet, size=3)
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.euler_angles = self.rng.uniform(low=0, high=2 * np.pi, size=3)

        car.demo_respawn_timer = 0.0
        car.on_ground = False
        car.supersonic_time = 0.0
        car.boost_amount = 0
        car.boost_active_time = 0.0
        car.handbrake = 0.0

        car.has_jumped = False
        car.is_holding_jump = False
        car.is_jumping = False
        car.jump_time = 0.0

        car.has_flipped = False
        car.has_double_jumped = False
        car.air_time_since_jump = 0.0
        car.flip_time = 0.0
        car.flip_torque = np.zeros(3, dtype=np.float32)

        car.is_autoflipping = False
        car.autoflip_timer = 0.0
        car.autoflip_direction = 0.0
        return car
