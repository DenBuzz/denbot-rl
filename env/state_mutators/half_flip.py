import numpy as np
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.api.car import BLUE_TEAM
from rlgym.rocket_league.common_values import BACK_WALL_Y, BALL_RADIUS, BALL_RESTING_HEIGHT, OCTANE, SIDE_WALL_X
from rlgym.rocket_league.sim import RocketSimEngine


class HalfFlip:
    """
    A StateMutator that randomizes ball location.
    """

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def reset(self, info): ...

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
        x_lim = SIDE_WALL_X - 30 * BALL_RADIUS
        y_lim = BACK_WALL_Y - 30 * BALL_RADIUS
        state.ball.position = np.array(
            [
                self.rng.uniform(-x_lim, x_lim),
                self.rng.uniform(-y_lim, y_lim),
                self.rng.uniform(BALL_RESTING_HEIGHT, 2 * BALL_RESTING_HEIGHT),
            ]
        )
        state.ball.linear_velocity = np.array(
            [
                self.rng.normal(scale=80),
                self.rng.normal(scale=80),
                self.rng.normal(loc=0, scale=50),
            ]
        )
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        car = self._new_car()
        car.team_num = BLUE_TEAM

        angle = self.rng.uniform(0, 2 * np.pi)
        u_vec = np.array([np.cos(angle), np.sin(angle), 0])
        dist = self.rng.normal(loc=1500, scale=300)
        car.physics.position = state.ball.position + u_vec * dist
        car.physics.position[2] = 17
        car.physics.euler_angles = np.array([0, angle, 0])

        state.cars["blue-0"] = car

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
        car.boost_amount = self.rng.uniform(0, 100)
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
