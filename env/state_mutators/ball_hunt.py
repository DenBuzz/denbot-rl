import numpy as np
import rlgym.rocket_league.common_values as cv
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.sim import RocketSimEngine


class BallHunt:
    CURRICULUM_STEPS = 10
    Y_MAX = cv.BACK_WALL_Y - cv.CORNER_CATHETUS_LENGTH
    Y_START = cv.BALL_RADIUS * 6
    X_MAX = cv.SIDE_WALL_X - cv.CORNER_CATHETUS_LENGTH
    X_START = cv.BALL_RADIUS * 6
    SPEED_MAX = cv.BALL_MAX_SPEED / 5

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def reset(self, info: dict):
        task = info.get("task", 0)
        self.y_max = self.Y_MAX * (task / self.CURRICULUM_STEPS)
        self.x_max = self.X_MAX * (task / self.CURRICULUM_STEPS)
        self.angle_max = np.pi * (task / self.CURRICULUM_STEPS)
        self.distance_max = 4 * cv.BALL_RADIUS + 40 * cv.BALL_RADIUS * (task / self.CURRICULUM_STEPS)
        self.speed_max = self.SPEED_MAX * (task / self.CURRICULUM_STEPS)

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
        state.ball.position = np.array(
            [
                self.rng.uniform(-self.x_max, self.x_max),
                self.rng.uniform(-self.y_max, self.y_max),
                self.rng.uniform(cv.BALL_RESTING_HEIGHT, cv.BALL_RESTING_HEIGHT + cv.BALL_RADIUS * 2),
            ],
            dtype=np.float32,
        )
        state.ball.linear_velocity = np.array(
            [
                self.rng.uniform(-self.speed_max, self.speed_max),
                self.rng.uniform(-self.speed_max, self.speed_max),
                self.rng.uniform(-self.speed_max / 2, self.speed_max / 2),
            ]
        )
        state.ball.angular_velocity = np.zeros(3)

        car = self._new_car()
        car.team_num = cv.BLUE_TEAM

        ball_x, ball_y = state.ball.position[:2]
        car_dx = self.rng.uniform(2 * cv.BALL_RADIUS, self.distance_max)
        car_dy = self.rng.uniform(2 * cv.BALL_RADIUS, self.distance_max)
        car_x = np.clip(ball_x + car_dx, -cv.SIDE_WALL_X + cv.CORNER_CATHETUS_LENGTH, cv.SIDE_WALL_X - cv.CORNER_CATHETUS_LENGTH)
        car_y = np.clip(ball_y + car_dy, -cv.BACK_WALL_Y + cv.CORNER_CATHETUS_LENGTH, cv.BACK_WALL_Y - cv.CORNER_CATHETUS_LENGTH)
        car.physics.position = np.array([car_x, car_y, 17])

        vec2ball = state.ball.position - car.physics.position

        angle = np.arctan2(vec2ball[1], vec2ball[0])
        angle = self.rng.uniform(angle - self.angle_max, angle + self.angle_max)

        car.physics.euler_angles = np.array([0, angle, 0])

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
