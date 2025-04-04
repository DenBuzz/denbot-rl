from collections import defaultdict

import numpy as np
import rlgym.rocket_league.common_values as common_values
from rlgym.rocket_league.api import Car, GameState, PhysicsObject


class DenBotReward:
    def __init__(
        self,
        goal_scored: float = 0,
        boost_difference: float = 0,
        ball_touch: float = 0,
        demo: float = 0,
        distance_player_ball: float = 0,
        distance_ball_goal: float = 0,
        facing_ball: float = 0,
        align_ball_goal: float = 0,
        closest_to_ball: float = 0,
        touched_last: float = 0,
        behind_ball: float = 0,
        velocity_player_to_ball: float = 0,
        velocity: float = 0,
        boost_amount: float = 0,
        forward_velocity: float = 0,
    ) -> None:
        self.goal_scored = goal_scored
        self.boost_difference = boost_difference
        self.ball_touch = ball_touch
        self.demo = demo

        self.distance_player_ball = distance_player_ball
        self.distance_ball_goal = distance_ball_goal
        self.facing_ball = facing_ball
        self.align_ball_goal = align_ball_goal
        self.closest_to_ball = closest_to_ball
        self.touched_last = touched_last
        self.behind_ball = behind_ball
        self.velocity_player_to_ball = velocity_player_to_ball
        self.velocity = velocity
        self.boost_amount = boost_amount
        self.forward_velocity = forward_velocity

        self._agent_boosts = defaultdict(float)

    def reset(self, info: dict):
        self._agent_boosts = defaultdict(float)

    def apply(self, agent: str, state: GameState) -> float:
        car = state.cars[agent]
        if car.team_num == common_values.ORANGE_TEAM:
            car_phys = car.inverted_physics
            ball = state.inverted_ball
        else:
            car_phys = car.physics
            ball = state.ball

        reward_inputs = (agent, car, car_phys, ball, state)

        reward = (
            self._goal_scored(*reward_inputs) * self.goal_scored
            + self._boost_difference(*reward_inputs) * self.boost_difference
            + self._ball_touch(*reward_inputs) * self.ball_touch
            + self._distance_player_ball(*reward_inputs) * self.distance_player_ball
            + self._facing_ball(*reward_inputs) * self.facing_ball
            + self._velocity_player_to_ball(*reward_inputs) * self.velocity_player_to_ball
            + self._velocity(*reward_inputs) * self.velocity
            + self._boost_amount(*reward_inputs) * self.boost_amount
        )

        return reward

    def _goal_scored(
        self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState
    ) -> float:
        if state.scoring_team != car.team_num:
            return -1
        ball_speed_bonus = np.linalg.norm(ball.linear_velocity) / common_values.BALL_MAX_SPEED
        return 1.0 + float(ball_speed_bonus)

    def _boost_difference(
        self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState
    ) -> float:
        reward = np.sqrt(car.boost_amount / 100) - np.sqrt(self._agent_boosts[agent] / 100)
        self._agent_boosts[agent] = car.boost_amount
        return reward

    def _distance_player_ball(
        self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState
    ) -> float:
        agent_dist = np.linalg.norm(car_physics.position - ball.position) - common_values.BALL_RADIUS
        return np.exp2(-agent_dist / common_values.CAR_MAX_SPEED)

    def _ball_touch(
        self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState
    ) -> float:
        return int(car.ball_touches > 0)

    def _facing_ball(
        self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState
    ) -> float:
        ball_vec = ball.position - car_physics.position
        ball_vec_u = ball_vec / np.linalg.norm(ball_vec)
        return np.dot(car_physics.forward, ball_vec_u)

    def _velocity_player_to_ball(
        self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState
    ) -> float:
        norm = np.linalg.norm(car_physics.linear_velocity)
        if norm == 0:
            return 0

        vel_u = car_physics.linear_velocity / norm
        ball_vec = ball.position - car_physics.position
        ball_vec_u = ball_vec / np.linalg.norm(ball_vec)
        return np.dot(ball_vec_u, vel_u)

    def _velocity(
        self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState
    ) -> float:
        norm = float(np.linalg.norm(car_physics.linear_velocity))
        return norm / common_values.CAR_MAX_SPEED

    def _boost_amount(
        self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState
    ) -> float:
        return np.sqrt(car.boost_amount / 100)
