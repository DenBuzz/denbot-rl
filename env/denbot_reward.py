from collections import defaultdict

import numpy as np
import rlgym.rocket_league.common_values as cv
from rlgym.rocket_league.api import Car, GameState, PhysicsObject

BOOST_LOCATIONS = np.array(cv.BOOST_LOCATIONS)
BOOST_PAD_AMOUNTS = 12 * np.ones(BOOST_LOCATIONS.shape[0])
BOOST_PAD_AMOUNTS[[3, 4, 15, 18, 29, 30]] = 100


class DenBotReward:
    def __init__(
        self,
        goal_scored: float = 0,
        boost_collect: float = 0,
        full_boost: float = 0,
        ball_touch: float = 0,
        demo: float = 0,
        distance_player_ball: float = 0,
        offensive_angle: float = 0,
        distance_ball_goal: float = 0,
        facing_ball: float = 0,
        align_ball_goal: float = 0,
        closest_to_ball: float = 0,
        touched_last: float = 0,
        behind_ball: float = 0,
        velocity_player_to_ball: float = 0,
        velocity_ball_goal: float = 0,
        velocity: float = 0,
        boost_amount: float = 0,
        boost_proximity: float = 0,
        forward_velocity: float = 0,
    ) -> None:
        self.goal_scored = goal_scored
        self.boost_collect = boost_collect
        self.full_boost = full_boost
        self.ball_touch = ball_touch
        self.demo = demo

        self.distance_player_ball = distance_player_ball
        self.offensive_angle = offensive_angle
        self.distance_ball_goal = distance_ball_goal
        self.facing_ball = facing_ball
        self.align_ball_goal = align_ball_goal
        self.closest_to_ball = closest_to_ball
        self.touched_last = touched_last
        self.behind_ball = behind_ball
        self.velocity_player_to_ball = velocity_player_to_ball
        self.velocity_ball_goal = velocity_ball_goal
        self.velocity = velocity
        self.boost_amount = boost_amount
        self.boost_proximity = boost_proximity
        self.forward_velocity = forward_velocity

        self._agent_boosts = defaultdict(float)
        self.reward_weights = np.array(
            [
                goal_scored,
                boost_collect,
                full_boost,
                ball_touch,
                demo,
                distance_player_ball,
                offensive_angle,
                distance_ball_goal,
                facing_ball,
                align_ball_goal,
                closest_to_ball,
                touched_last,
                behind_ball,
                velocity_player_to_ball,
                velocity_ball_goal,
                velocity,
                boost_amount,
                boost_proximity,
                forward_velocity,
            ],
            dtype=np.float32,
        )

    def reset(self, info: dict):
        self._agent_boosts = defaultdict(float)
        info["reward_weights"] = self.reward_weights

    def apply(self, agent: str, state: GameState) -> float:
        car = state.cars[agent]
        if car.team_num == cv.ORANGE_TEAM:
            car_phys = car.inverted_physics
            ball = state.inverted_ball
        else:
            car_phys = car.physics
            ball = state.ball

        reward_inputs = (agent, car, car_phys, ball, state)

        reward = (
            self._goal_scored(*reward_inputs) * self.goal_scored
            + self._boost_collect(*reward_inputs) * self.boost_collect
            + self._full_boost(*reward_inputs) * self.full_boost
            + self._ball_touch(*reward_inputs) * self.ball_touch
            + self._distance_player_ball(*reward_inputs) * self.distance_player_ball
            + self._offensive_angle(*reward_inputs) * self.offensive_angle
            + self._facing_ball(*reward_inputs) * self.facing_ball
            + self._velocity_player_to_ball(*reward_inputs) * self.velocity_player_to_ball
            + self._velocity_ball_goal(*reward_inputs) * self.velocity_ball_goal
            + self._velocity(*reward_inputs) * self.velocity
            + self._boost_amount(*reward_inputs) * self.boost_amount
            + self._boost_proximity(*reward_inputs) * self.boost_proximity
        )

        self._agent_boosts[agent] = car.boost_amount
        return reward

    def _goal_scored(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        if not state.goal_scored:
            return 0
        if state.scoring_team != car.team_num:
            return -1
        ball_speed_bonus = np.linalg.norm(ball.linear_velocity) / cv.BALL_MAX_SPEED
        return 1.0 + float(ball_speed_bonus)

    def _boost_collect(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        if car.boost_amount > self._agent_boosts[agent]:
            return np.sqrt(car.boost_amount / 100) - np.sqrt(self._agent_boosts[agent] / 100)
        return 0.0

    def _full_boost(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        return float(car.boost_amount >= 100)

    def _distance_player_ball(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        agent_dist = np.linalg.norm(car_physics.position - ball.position) - cv.BALL_RADIUS
        return np.exp2(-agent_dist / cv.CAR_MAX_SPEED)

    def _offensive_angle(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        goal_vec = ball.position[:2] - cv.ORANGE_GOAL_BACK[:2]
        car_vec = car_physics.position[:2] - ball.position[:2]
        foo = np.dot(car_vec / np.linalg.norm(car_vec), goal_vec / np.linalg.norm(goal_vec))
        return foo

    def _distance_ball_goal(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        back_net = np.array([0, cv.BACK_NET_Y, cv.GOAL_HEIGHT / 2])
        ball_goal_distance = np.linalg.norm(back_net - ball.position)
        return np.exp(ball_goal_distance / cv.BALL_MAX_SPEED)

    def _ball_touch(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        return int(car.ball_touches > 0)

    def _facing_ball(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        ball_vec = ball.position - car_physics.position
        ball_vec_u = ball_vec / np.linalg.norm(ball_vec)
        return np.dot(car_physics.forward, ball_vec_u)

    def _velocity_player_to_ball(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        norm = np.linalg.norm(car_physics.linear_velocity)
        if norm == 0:
            return 0

        vel_u = car_physics.linear_velocity / norm
        ball_vec = ball.position - car_physics.position
        ball_vec_u = ball_vec / np.linalg.norm(ball_vec)
        return np.dot(ball_vec_u, vel_u)

    def _velocity_ball_goal(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        norm = np.linalg.norm(ball.linear_velocity)
        if norm == 0:
            return 0

        vel_u = ball.linear_velocity / norm
        goal_vec = cv.ORANGE_GOAL_BACK - ball.position
        goal_vec_u = goal_vec / np.linalg.norm(goal_vec)
        return np.dot(goal_vec_u, vel_u)

    def _velocity(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        norm = float(np.linalg.norm(car_physics.linear_velocity))
        return norm / cv.CAR_MAX_SPEED

    def _boost_amount(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        return np.sqrt(car.boost_amount / 100)

    def _boost_proximity(self, agent: str, car: Car, car_physics: PhysicsObject, ball: PhysicsObject, state: GameState) -> float:
        timer_mask = state.boost_pad_timers == 0
        distances = np.linalg.norm(BOOST_LOCATIONS[:, :2] - car_physics.position[:2], axis=1)
        rewards = timer_mask * np.sqrt(BOOST_PAD_AMOUNTS / 100) * np.exp(-0.002 * distances)

        return np.sum(rewards)
