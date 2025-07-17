import numpy as np
import rlgym.rocket_league.common_values as cv
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.sim import RocketSimEngine

from env.sim_setters.state_mutator import StateMutator


class BallHunt(StateMutator):
    CURRICULUM_STEPS = 10
    Y_MAX = cv.BACK_WALL_Y - cv.CORNER_CATHETUS_LENGTH
    Y_START = cv.BALL_RADIUS * 6
    X_MAX = cv.SIDE_WALL_X - cv.CORNER_CATHETUS_LENGTH
    X_START = cv.BALL_RADIUS * 6
    SPEED_MAX = cv.BALL_MAX_SPEED / 5

    def reset(self, info: dict):
        task = info.get("task", 0)
        self.y_max = self.Y_MAX * (task / self.CURRICULUM_STEPS)
        self.x_max = self.X_MAX * (task / self.CURRICULUM_STEPS)
        self.angle_max = np.pi * (task / self.CURRICULUM_STEPS)
        self.distance_max = 4 * cv.BALL_RADIUS + 40 * cv.BALL_RADIUS * (task / self.CURRICULUM_STEPS)
        self.speed_max = self.SPEED_MAX * (task / self.CURRICULUM_STEPS) + 1

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

        car = self.default_car()
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
        car.boost_amount = self.rng.uniform(0, 100)

        state.cars["blue-0"] = car


class SpeedFlip(StateMutator):
    CURRICULUM_STEPS = 10
    SPEED_MAX = cv.BALL_MAX_SPEED / 20
    SECTOR_MAX = np.pi / 4

    def reset(self, info: dict):
        task = info.get("task", 0)
        self.sector_size = self.SECTOR_MAX * (task / self.CURRICULUM_STEPS)
        self.speed_max = self.SPEED_MAX * (task / self.CURRICULUM_STEPS) + 1

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
        state.ball.position = np.array(
            [
                self.rng.uniform(-cv.SIDE_WALL_X / 4, cv.SIDE_WALL_X / 4),
                self.rng.uniform(-cv.BACK_WALL_Y / 3, cv.BACK_WALL_Y / 3),
                cv.BALL_RESTING_HEIGHT + 2,
            ],
            dtype=np.float32,
        )
        state.ball.linear_velocity = np.array(
            [
                self.rng.uniform(-self.speed_max, self.speed_max),
                self.rng.uniform(-self.speed_max, self.speed_max),
                self.rng.uniform(-10, 10),
            ]
        )
        state.ball.angular_velocity = np.zeros(3)

        car_angle = self.rng.uniform(0, 2 * np.pi)
        car_distance = self.rng.uniform(3278, 4000)
        car_dx = car_distance * np.cos(car_angle)
        car_dy = car_distance * np.sin(car_angle)

        car = self.default_car()
        car.team_num = cv.BLUE_TEAM

        ball_x, ball_y = state.ball.position[:2]

        car_x = np.clip(ball_x + car_dx, -cv.SIDE_WALL_X + cv.CORNER_CATHETUS_LENGTH, cv.SIDE_WALL_X - cv.CORNER_CATHETUS_LENGTH)
        car_y = np.clip(ball_y + car_dy, -cv.BACK_WALL_Y + cv.CORNER_CATHETUS_LENGTH, cv.BACK_WALL_Y - cv.CORNER_CATHETUS_LENGTH)

        car.physics.position = np.array([car_x, car_y, 17])
        yaw = car_angle
        yaw += self.rng.uniform(-self.sector_size, self.sector_size)
        if self.rng.random() > 0.5:
            yaw += np.pi
        car.physics.euler_angles = np.array([0, yaw % (2 * np.pi), 0])
        car.boost_amount = self.rng.uniform(0, 50)

        state.cars["blue-0"] = car
