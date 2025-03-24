import numpy as np
import RocketSim as rsim
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


class AirialCurriculum:
    def __init__(self, blue_size: int = 1, orange_size: int = 0, ball_height: float = BALL_RESTING_HEIGHT) -> None:
        self.rng = np.random.default_rng()
        self.blue_size = blue_size
        self.orange_size = orange_size
        self.ball_height = ball_height
        print(f"Ball height set to {ball_height}")

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
        # Apply no mass
        mutator_config = rsim.MutatorConfig()
        mutator_config.ball_mass = 0
        sim._arena.set_mutator_config(mutator_config)

        # regular state stuff
        state.ball.position = np.array(
            [
                (self.rng.random() - 0.5) * 2 * (SIDE_WALL_X - 10 * BALL_RADIUS),
                (self.rng.random() - 0.5) * 2 * (BACK_WALL_Y - 10 * BALL_RADIUS),
                self.ball_height,
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

        for car in state.cars.values():
            car.physics.position = np.array(
                [
                    (self.rng.random() - 0.5) * 2 * (SIDE_WALL_X - 10 * BALL_RADIUS),
                    (self.rng.random() - 0.5) * 2 * (BACK_WALL_Y - 10 * BALL_RADIUS),
                    17,
                ]
            )
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.array([0, 0, 0], dtype=np.float32)
            car.boost_amount = np.random.uniform(70, 100)

    def _new_car(self) -> Car:
        car = Car()
        car.hitbox_type = OCTANE

        car.physics = PhysicsObject()

        car.demo_respawn_timer = 0.0
        car.on_ground = True
        car.supersonic_time = 0.0
        car.boost_amount = 0.0
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
