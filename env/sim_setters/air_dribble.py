import numpy as np
import rlgym.rocket_league.common_values as cv
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.sim import RocketSimEngine

from env.sim_setters.rocket_sim_setter import RocketSimSetter


class WallAirDribble(RocketSimSetter):
    SIDE_WALL_MAX_Y = cv.BACK_WALL_Y - cv.CORNER_CATHETUS_LENGTH

    def reset(self, info: dict):
        # task = info.get("task", 0)
        pass

    @property
    def agents(self) -> list[str]:
        return ["blue-0"]

    def apply(self, sim: RocketSimEngine) -> GameState:
        state: GameState = sim.create_base_state()
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

        car = self.default_car()
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
        sim.set_state(state, {})
        return state


class FieldAirDribble(WallAirDribble):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, sim: RocketSimEngine) -> GameState:
        state: GameState = sim.create_base_state()
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

        ball_vx = goal_ball_vec[0] / 10 + self.rng.uniform(-200, 200)
        ball_vy = goal_ball_vec[1] / 10 + self.rng.uniform(-200, 200)
        pop = self.rng.uniform(500, 1200)
        state.ball.linear_velocity = np.array([ball_vx, ball_vy, pop])
        state.ball.angular_velocity = np.zeros(3)

        car = self.default_car()
        car.team_num = cv.BLUE_TEAM

        goal_ball_vec_u = goal_ball_vec / np.linalg.norm(goal_ball_vec)
        car_xy = state.ball.position[:2] - 2 * cv.BALL_RADIUS * goal_ball_vec_u[:2]

        car.physics.position = np.concatenate((car_xy, [17]))
        # car.physics.linear_velocity = state.ball.linear_velocity
        car.physics.euler_angles = np.array([0, np.pi / 2, 0])
        car.boost_amount = 100

        state.cars["blue-0"] = car
        sim.set_state(state, {})
        return state
