import gymnasium as gym

import math
from typing import Any

import numpy as np

from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BACK_WALL_Y, ORANGE_TEAM

from env.obs_builders.encoders import encode_position, fourier_encoder


class ObsBuilder:
    def obs_space(self) -> gym.Space: ...

    def build_obs(self, agents: list[str], state: GameState) -> dict[str, np.ndarray]: ...


class DefaultObs(ObsBuilder):
    """
    The default observation builder.
    """

    def __init__(
        self,
        num_cars=3,
        pos_frequencies=4,
        ang_coef=1 / math.pi,
        lin_vel_coef=1 / 2300,
        ang_vel_coef=1 / math.pi,
        pad_timer_coef=1 / 10,
        boost_coef=1 / 100,
    ):
        super().__init__()
        self.position_frequencies = pos_frequencies
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.PAD_TIMER_COEF = pad_timer_coef
        self.BOOST_COEF = boost_coef
        self.num_cars = num_cars

    def get_obs_space(self, agent: str) -> gym.Space:
        return gym.spaces.Box(
            -100,
            100,
            shape=(
                40 + 9 + 3 * 2 * self.position_frequencies + 29 + 3 * 2 * self.position_frequencies * self.num_cars,
            ),
        )

    def reset(self, agents: list[str], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: list[str], state: GameState) -> dict[str, np.ndarray]:
        obs = {}
        for agent in agents:
            obs[agent] = self._build_agent_obs(agent, state)

        return obs

    def _build_agent_obs(self, agent: str, state: GameState) -> np.ndarray:
        car = state.cars[agent]
        if car.team_num == ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pad_timers
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pad_timers

        obs = [  # Global stuff
            encode_position(ball.position, frequencies=self.position_frequencies),
            ball.linear_velocity * self.LIN_VEL_COEF,
            ball.angular_velocity * self.ANG_VEL_COEF,
            pads * self.PAD_TIMER_COEF,  # 34
            [  # Partially observable variables
                car.is_holding_jump,
                car.handbrake,
                car.has_jumped,
                car.is_jumping,
                car.has_flipped,
                car.is_flipping,
                car.has_double_jumped,
                car.can_flip,
                car.air_time_since_jump,
            ],
        ]

        car_obs = self._generate_car_obs(car, ball, inverted)
        obs.append(car_obs)

        # allies = []
        # enemies = []
        #
        # for other, other_car in state.cars.items():
        #     if other == agent:
        #         continue
        #
        #     car_obs = self._generate_car_obs(other_car, ball, inverted)
        #     if other_car.team_num == car.team_num:
        #         allies.append(car_obs)
        #     else:
        #         enemies.append(car_obs)
        #
        # if self.num_cars is not None:
        #     # Padding for multi game mode
        #     while len(allies) < self.num_cars - 1:
        #         allies.append(np.zeros_like(car_obs))
        #     while len(enemies) < self.num_cars:
        #         enemies.append(np.zeros_like(car_obs))
        #
        # obs.extend(allies)
        # obs.extend(enemies)
        return np.concatenate(obs, dtype=np.float32)

    def _generate_car_obs(self, car: Car, ball: PhysicsObject, inverted: bool) -> np.ndarray:
        if inverted:
            physics = car.inverted_physics
        else:
            physics = car.physics

        norm = np.linalg.norm(physics.linear_velocity)
        ball_vec = ball.position - physics.position
        ball_vec_u = ball_vec / np.linalg.norm(ball_vec)

        if norm == 0:
            vel_ball_dot = 0
        else:
            vel_u = physics.linear_velocity / norm
            vel_ball_dot = np.dot(vel_u, ball_vec_u)

        pointing_ball_dot = np.dot(physics.forward, ball_vec_u)

        xy_angle_offset = np.arccos(np.dot(physics.forward[:2], ball_vec_u[:2]))
        xy_ball_angle = np.sign(np.cross(physics.forward[:2], ball_vec_u[:2])) * xy_angle_offset

        return np.concatenate(
            [
                encode_position(physics.position, frequencies=self.position_frequencies),  # 3*2*freqs
                fourier_encoder(-np.pi, np.pi, xy_ball_angle, frequencies=2, periodic=True),
                physics.forward,
                physics.up,
                physics.linear_velocity * self.LIN_VEL_COEF,
                physics.angular_velocity * self.ANG_VEL_COEF,
                ball_vec / (2 * BACK_WALL_Y),
                ball_vec_u,
                [
                    pointing_ball_dot,
                    vel_ball_dot,
                    car.boost_amount * self.BOOST_COEF,
                    car.demo_respawn_timer,
                    int(car.on_ground),
                    int(car.is_boosting),
                    int(car.is_supersonic),
                ],
            ]
        )
