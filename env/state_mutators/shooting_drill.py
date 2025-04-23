import numpy as np
import rlgym.rocket_league.common_values as cv
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.sim import RocketSimEngine


class ShootingDrill:
    """
    A StateMutator that randomizes ball location.
    """

    CURRICULUM_STEPS = 50

    Y_START = cv.BACK_WALL_Y - 8 * cv.BALL_RADIUS
    Y_END = 0
    X_START = cv.GOAL_CENTER_TO_POST
    X_END = cv.SIDE_WALL_X - cv.CORNER_CATHETUS_LENGTH

    VZ_END = cv.BALL_MAX_SPEED / 10
    VX_END = VZ_END / 2
    VY_END = VZ_END / 2

    CAR_GAP_START = cv.BALL_RADIUS * 2 * 4
    CAR_GAP_END = cv.BALL_RADIUS * 2 * 12

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def reset(self, info):
        task = info.get("task", 0)
        self.y_max = self.Y_START + (self.Y_END - self.Y_START) * (task / self.CURRICULUM_STEPS)
        self.x_max = self.X_START + (self.X_END - self.X_START) * (task / self.CURRICULUM_STEPS)
        self.vz_max = (self.VZ_END) * (task / self.CURRICULUM_STEPS)
        self.vx_max = (self.VX_END) * (task / self.CURRICULUM_STEPS)
        self.vy_max = (self.VY_END) * (task / self.CURRICULUM_STEPS)
        self.car_gap_max = self.CAR_GAP_START + (self.CAR_GAP_END - self.CAR_GAP_START) * (task / self.CURRICULUM_STEPS)

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
        y = self.rng.uniform(self.y_max, self.Y_START)
        x_max = min(self.x_max, cv.GOAL_CENTER_TO_POST + (cv.BACK_WALL_Y - y))
        state.ball.position = np.array([self.rng.uniform(-x_max, x_max), y, cv.BALL_RESTING_HEIGHT])
        state.ball.linear_velocity = np.array(
            [
                self.rng.uniform(0, self.vx_max),
                self.rng.uniform(0, self.vy_max),
                self.rng.uniform(0, self.vz_max),
            ]
        )
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        car = self._new_car()
        car.team_num = cv.BLUE_TEAM

        car_gap = self.rng.uniform(self.CAR_GAP_START, self.car_gap_max)
        car.physics.position = np.array([state.ball.position[0], state.ball.position[1] - car_gap, 17])
        car.physics.euler_angles = np.array([0, np.pi / 2, 0])
        car.boost_amount = np.random.random() * 100

        state.cars["blue-0"] = car

    def _new_car(self) -> Car:
        car = Car()
        car.hitbox_type = cv.OCTANE

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
