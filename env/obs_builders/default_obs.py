import gymnasium as gym

import math
from typing import Any

import numpy as np

from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM, SIDE_WALL_X, CEILING_Z, BACK_NET_Y
from env.obs_builders.encoders import fourier_encoder


class DefaultObs:
    """
    The default observation builder.
    """

    def __init__(
        self,
        num_cars=3,
        pos_embedding_size=4,
        ang_coef=1 / math.pi,
        lin_vel_coef=1 / 2300,
        ang_vel_coef=1 / math.pi,
        pad_timer_coef=1 / 10,
        boost_coef=1 / 100,
    ):
        super().__init__()
        self.position_embedding_size = pos_embedding_size
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.PAD_TIMER_COEF = pad_timer_coef
        self.BOOST_COEF = boost_coef
        self.num_cars = num_cars

    def get_obs_space(self, agent: str) -> gym.Space:
        return gym.spaces.Box(-100, 100, shape=(61 + 29 * self.num_cars * 2,))

    def reset(self, agents: list[str], initial_state: GameState, shared_info: dict[str, Any]) -> None:
        pass

    def build_obs(self, agents: list[str], state: GameState, shared_info: dict[str, Any]) -> dict[str, np.ndarray]:
        obs = {}
        for agent in agents:
            obs[agent] = self._build_obs(agent, state, shared_info)

        return obs

    def _build_obs(self, agent: str, state: GameState, shared_info: dict[str, Any]) -> np.ndarray:
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
            self._encode_position(ball.position),
            ball.linear_velocity * self.LIN_VEL_COEF,
            ball.angular_velocity * self.ANG_VEL_COEF,
            pads * self.PAD_TIMER_COEF,
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

        car_obs = self._generate_car_obs(car, inverted)
        obs.append(car_obs)

        allies = []
        enemies = []

        for other, other_car in state.cars.items():
            if other == agent:
                continue

            car_obs = self._generate_car_obs(other_car, inverted)
            if other_car.team_num == car.team_num:
                allies.append(car_obs)
            else:
                enemies.append(car_obs)

        if self.num_cars is not None:
            # Padding for multi game mode
            while len(allies) < self.num_cars - 1:
                allies.append(np.zeros_like(car_obs))
            while len(enemies) < self.num_cars:
                enemies.append(np.zeros_like(car_obs))

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs, dtype=np.float32)

    def _generate_car_obs(self, car: Car, inverted: bool) -> np.ndarray:
        if inverted:
            physics = car.inverted_physics
        else:
            physics = car.physics

        return np.concatenate(
            [
                self._encode_position(physics.position),
                physics.forward,
                physics.up,
                physics.linear_velocity * self.LIN_VEL_COEF,
                physics.angular_velocity * self.ANG_VEL_COEF,
                [
                    car.boost_amount * self.BOOST_COEF,
                    car.demo_respawn_timer,
                    int(car.on_ground),
                    int(car.is_boosting),
                    int(car.is_supersonic),
                ],
            ]
        )

    def _encode_position(self, position: np.ndarray) -> np.ndarray:
        """Encode the shit out of some positions"""
        encoded = [
            fourier_encoder(-SIDE_WALL_X, SIDE_WALL_X, position[0], self.position_embedding_size),
            fourier_encoder(-BACK_NET_Y, BACK_NET_Y, position[1], self.position_embedding_size),
            fourier_encoder(-CEILING_Z, CEILING_Z, position[2], self.position_embedding_size),
        ]
        return np.concatenate(encoded)
