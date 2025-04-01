import math

import gymnasium as gym
import numpy as np
from numpy.linalg import norm
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BACK_WALL_Y, ORANGE_TEAM

from env.encoders import encode_position, fourier_encoder


class DenbotObs:
    """
    The default observation builder.
    """

    def __init__(
        self,
        pos_frequencies=4,
        angle_frequencies=2,
        ang_coef=1 / math.pi,
        lin_vel_coef=1 / 2300,
        ang_vel_coef=1 / math.pi,
        pad_timer_coef=1 / 10,
        boost_coef=1 / 100,
    ):
        super().__init__()
        self.position_frequencies = pos_frequencies
        self.angle_frequencies = angle_frequencies
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.PAD_TIMER_COEF = pad_timer_coef
        self.BOOST_COEF = boost_coef

    def reset(self, info: dict): ...

    def get_obs_space(self, agent: str) -> gym.Space:
        return gym.spaces.Box(
            -100,
            100,
            shape=(
                3 * 2 * self.position_frequencies
                + 3
                + 3
                + 34
                + 9
                + 3 * 2 * self.position_frequencies
                + 4 * 3 * 2
                + 4 * 3
                + 6,
            ),
        )

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

        ball_vec = ball.position - physics.position

        yaw_offset = planar_angle(reference=physics.forward, normal=physics.up, target=ball_vec)
        # Using left for pitch reference makes up positive and down negative
        pitch_offset = planar_angle(reference=physics.forward, normal=physics.left, target=ball_vec)

        vel_ball_xy_offset = planar_angle(
            reference=physics.linear_velocity,
            normal=np.array([0, 0, 1]),
            target=ball_vec,
        )
        vel_ball_z_offset = planar_angle(
            reference=physics.linear_velocity,
            normal=np.cross(physics.linear_velocity, np.array([0, 0, 1])),
            target=ball_vec,
        )

        return np.concatenate(
            [
                encode_position(physics.position, frequencies=self.position_frequencies),  # 3*2*freqs
                fourier_encoder(-np.pi, np.pi, yaw_offset, frequencies=self.angle_frequencies, periodic=True),
                fourier_encoder(-np.pi, np.pi, pitch_offset, frequencies=self.angle_frequencies, periodic=True),
                fourier_encoder(-np.pi, np.pi, vel_ball_xy_offset, frequencies=self.angle_frequencies, periodic=True),
                fourier_encoder(-np.pi, np.pi, vel_ball_z_offset, frequencies=self.angle_frequencies, periodic=True),
                physics.forward,
                physics.up,
                physics.linear_velocity * self.LIN_VEL_COEF,
                physics.angular_velocity * self.ANG_VEL_COEF,
                [
                    norm(ball_vec) / (2 * BACK_WALL_Y),
                    car.boost_amount * self.BOOST_COEF,
                    car.demo_respawn_timer,
                    int(car.on_ground),
                    int(car.is_boosting),
                    int(car.is_supersonic),
                ],
            ]
        )
        ###############
        # Get the normal vector (right) to the forward/up plane
        # Subtract from the vector to the ball the normal component dotted with that same vector
        # That yeilds the vector projected to the forward/up plane


def planar_angle(reference: np.ndarray, normal: np.ndarray, target: np.ndarray):
    """Return the yaw angle between a car's foward and a unit vector"""
    # find an subtract the normal component from the vector
    if (x := norm(normal)) == 0:
        return 0
    else:
        n = normal / x

    t_proj = target - np.dot(target, n) * n
    if (x := norm(t_proj)) == 0:
        return 0
    else:
        t_proj_norm = t_proj / x

    ref_proj = reference - np.dot(reference, n) * n
    if (x := norm(ref_proj)) == 0:
        return 0
    else:
        ref_proj_norm = ref_proj / x

    angle = np.arccos(np.clip(np.dot(ref_proj_norm, t_proj_norm), -1, 1))
    sign = np.sign(np.dot(np.cross(ref_proj_norm, t_proj_norm), n))
    if sign == 0:
        return angle
    return angle * sign
