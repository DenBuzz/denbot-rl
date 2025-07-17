import numpy as np
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.api.car import BLUE_TEAM, ORANGE_TEAM
from rlgym.rocket_league.common_values import BACK_WALL_Y, BALL_RADIUS, BALL_RESTING_HEIGHT, OCTANE, SIDE_WALL_X
from rlgym.rocket_league.sim import RocketSimEngine


class Random:
    """
    A StateMutator that randomizes ball location.
    """

    def __init__(
        self,
        blue_size: int = 1,
        orange_size: int = 0,
    ) -> None:
        self.blue_size = blue_size
        self.orange_size = orange_size
        self.rng = np.random.default_rng()

    def reset(self, info): ...

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
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

        for idx in range(self.blue_size):
            car = self._new_car()
            car.team_num = BLUE_TEAM
            state.cars["blue-{}".format(idx)] = car

        for idx in range(self.orange_size):
            car = self._new_car()
            car.team_num = ORANGE_TEAM
            state.cars["orange-{}".format(idx)] = car

        for car in state.cars.values():
            car.physics.position = np.array(
                [
                    self.rng.normal(state.ball.position[0], scale=30),
                    self.rng.normal(state.ball.position[1], scale=30),
                    17.1,
                ]
            )
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0, 0, 0], dtype=np.float32)
            car.boost_amount = np.random.random() * 100

    def _new_car(self) -> Car:
        car = Car()
        car.hitbox_type = OCTANE

        car.physics = PhysicsObject()

        car.physics.position = np.zeros(3)
        car.physics.linear_velocity = np.zeros(3)
        car.physics.angular_velocity = np.zeros(3)
        car.physics.euler_angles = np.zeros(3)

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
