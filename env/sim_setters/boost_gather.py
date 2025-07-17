import numpy as np

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

from env.sim_setters.state_mutator import StateMutator


class BoostGather(StateMutator):
    def __init__(
        self,
        blue_size: int = 1,
        orange_size: int = 0,
        min_car_height: float = 34.0,
        max_car_height: float = 10 * 34.0,
        max_car_yeet: float = 1000,
    ) -> None:
        super().__init__()
        self.blue_size = blue_size
        self.orange_size = orange_size
        self.min_car_height = min_car_height
        self.max_car_height = max_car_height
        self.max_car_yeet = max_car_yeet

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
        # Apply no mass
        state.ball.position = np.array([0, 0, BALL_RESTING_HEIGHT], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)

        car = self._new_car()

        state.cars["blue-0"] = car

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
