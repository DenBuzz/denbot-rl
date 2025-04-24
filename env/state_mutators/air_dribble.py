import numpy as np
import rlgym.rocket_league.common_values as cv
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.sim import RocketSimEngine


class WallAirDribble:
    SIDE_WALL_MAX_Y = cv.BACK_WALL_Y - cv.CORNER_CATHETUS_LENGTH

    def __init__(self) -> None:
        self.rng = np.random.default_rng()

    def reset(self, info: dict):
        # task = info.get("task", 0)
        pass

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
        ball_x = cv.SIDE_WALL_X - 4 * (2 * cv.BALL_RADIUS)
        if self.rng.random() > 0.5:
            # reflect sometimes
            ball_x = -ball_x

        ball_y = self.rng.uniform(-self.SIDE_WALL_MAX_Y, self.SIDE_WALL_MAX_Y)
        state.ball.position = np.array(
            [
                ball_x,
                ball_y,
                self.rng.uniform(cv.BALL_RESTING_HEIGHT, cv.BALL_RESTING_HEIGHT + cv.BALL_RADIUS / 2),
            ],
            dtype=np.float32,
        )
        ball_vx = np.sign(ball_x) * self.rng.uniform(500, 1200)
        ball_vy = self.rng.uniform(0, 100)
        state.ball.linear_velocity = np.array([ball_vx, ball_vy, 0])
        state.ball.angular_velocity = np.zeros(3)

        car = self._new_car()
        car.team_num = cv.BLUE_TEAM

        car.physics.position = np.array(
            [
                ball_x - np.sign(ball_x) * (self.rng.uniform(2, 7) * cv.BALL_RADIUS),
                ball_y - self.rng.uniform(0, 50),
                17,
            ]
        )
        car.physics.linear_velocity = state.ball.linear_velocity
        car.physics.euler_angles = np.array([0, np.pi / 2 * (1 - np.sign(ball_x)), 0])
        car.boost_amount = 100

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


class FieldAirDribble(WallAirDribble):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, state: GameState, sim: RocketSimEngine) -> None:
        ball_x = self.rng.uniform(-cv.SIDE_WALL_X + cv.CORNER_CATHETUS_LENGTH, cv.SIDE_WALL_X - cv.CORNER_CATHETUS_LENGTH)
        ball_y = self.rng.uniform(-cv.BACK_WALL_Y + cv.CORNER_CATHETUS_LENGTH, cv.BACK_WALL_Y - cv.CORNER_CATHETUS_LENGTH)

        state.ball.position = np.array(
            [
                ball_x,
                ball_y,
                self.rng.uniform(cv.BALL_RESTING_HEIGHT, cv.BALL_RESTING_HEIGHT + cv.BALL_RADIUS / 2),
            ],
            dtype=np.float32,
        )
        goal_ball_vec = cv.ORANGE_GOAL_BACK[:2] - state.ball.position[:2]

        ball_vx = goal_ball_vec[0] / 8 + self.rng.uniform(-200, 200)
        ball_vy = goal_ball_vec[1] / 8 + self.rng.uniform(-200, 200)
        pop = self.rng.uniform(500, 1200)
        state.ball.linear_velocity = np.array([ball_vx, ball_vy, pop])
        state.ball.angular_velocity = np.zeros(3)

        car = self._new_car()
        car.team_num = cv.BLUE_TEAM

        goal_ball_vec_u = goal_ball_vec / np.linalg.norm(goal_ball_vec)
        car_xy = state.ball.position[:2] - 2 * cv.BALL_RADIUS * goal_ball_vec_u[:2]

        car.physics.position = np.concatenate((car_xy, [17]))
        # car.physics.linear_velocity = state.ball.linear_velocity
        car.physics.euler_angles = np.array([0, np.pi / 2, 0])
        car.boost_amount = 100

        state.cars["blue-0"] = car
