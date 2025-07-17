import gymnasium as gym
import numpy as np
import rlgym.rocket_league.common_values as cv
from numpy.linalg import norm
from rlgym.rocket_league.api import Car, GameState, PhysicsObject

from env.encoders import binary_encoder, encode_position, fourier_encoder, planar_angle


class Observer1v1:
    """
    The default observation builder.
    """

    def __init__(self):
        pass

    def reset(self, info: dict):
        self.reward_weights = info["reward_weights"]  # 19

    def get_obs_space(self, agent: str) -> gym.Space:
        return gym.spaces.Dict(
            {
                "rewards": gym.spaces.Box(-100, 100, shape=(19,)),
                "pads": gym.spaces.Box(-100, 100, shape=(34,)),
                "ball": gym.spaces.Box(-100, 100, shape=(61 + 16,)),
                "agent": gym.spaces.Box(-100, 100, shape=(16 + 72 + 52 + 102,)),
                "mask": gym.spaces.MultiBinary(n=22),
            }
        )

    def build_obs(self, agents: list[str], state: GameState) -> dict[str, np.ndarray]:
        obs = {}
        for agent in agents:
            obs[agent] = self._build_agent_obs(agent, state)

        return obs

    def _build_agent_obs(self, agent: str, state: GameState) -> dict[str, np.ndarray]:
        car = state.cars[agent]
        if car.team_num == cv.ORANGE_TEAM:
            ball = state.inverted_ball
            pads = state.inverted_boost_pad_timers
            car_phys = car.inverted_physics
        else:
            ball = state.ball
            pads = state.boost_pad_timers
            car_phys = car.physics

        obs = {
            "rewards": self.reward_weights,
            "pads": self._pad_timers(pads),
            "ball": np.concatenate(
                (
                    self._ball_obs(ball),
                    self._relative_ball_obs(ball),
                ),
                dtype=np.float32,
            ),
            "agent": np.concatenate(
                (
                    self._car_obs(car),
                    self._car_physics_obs(car_phys),
                    self._relative_physics_obs(car_phys, ball),
                    self._relative_pads(car_phys),
                ),
                dtype=np.float32,
            ),
            "mask": self._get_mask(car),
        }
        return obs

    def _ball_obs(self, ball: PhysicsObject):
        pos = encode_position(ball.position, frequencies=6)  # high res ball position, 3*2*6=36
        vel = fourier_encoder(-cv.BALL_MAX_SPEED, cv.BALL_MAX_SPEED, ball.linear_velocity, frequencies=4).flatten()  # 3*2*4=24
        speed = (norm(ball.linear_velocity) / cv.BALL_MAX_SPEED,)  # 1
        # angular_vel = fourier_encoder()
        return np.concatenate((pos, vel, speed))  # 61

    def _pad_timers(self, pads) -> np.ndarray:
        return pads / 10  # 34

    def _car_obs(self, car: Car) -> np.ndarray:
        boost = binary_encoder(0, 100, car.boost_amount, num_bins=5)  # 5
        simple = np.array(
            [
                car.demo_respawn_timer,
                car.air_time_since_jump,
                int(car.on_ground),
                int(car.is_supersonic),
                car.handbrake,
                car.has_jumped,
                car.is_jumping,
                car.has_flipped,
                car.is_flipping,
                car.has_double_jumped,
                car.can_flip,
            ]
        )  # 11
        return np.concatenate((boost, simple))  # 16

    def _car_physics_obs(self, physics: PhysicsObject) -> np.ndarray:
        pos = encode_position(physics.position, frequencies=6)  # 3*2*6=36
        vel = fourier_encoder(-cv.CAR_MAX_SPEED, cv.CAR_MAX_SPEED, physics.linear_velocity, frequencies=4).flatten()  # 3*2*4=24
        speed = (norm(physics.linear_velocity) / cv.CAR_MAX_SPEED,)  # 1
        orientation = physics.quaternion  # 4
        angular_vel = fourier_encoder(-cv.CAR_MAX_ANG_VEL, cv.CAR_MAX_ANG_VEL, physics.angular_velocity, frequencies=1).flatten()  # 3*2*1=6
        angular_speed = (norm(angular_vel) / cv.CAR_MAX_ANG_VEL,)  # 1
        return np.concatenate((pos, vel, speed, orientation, angular_vel, angular_speed))  # 72

    def _relative_physics_obs(self, physics: PhysicsObject, ball: PhysicsObject) -> np.ndarray:
        ball_vec = ball.position - physics.position

        yaw_offset = planar_angle(reference=physics.forward, normal=physics.up, target=ball_vec)
        # Using left for pitch reference makes up positive and down negative
        pitch_offset = planar_angle(reference=physics.forward, normal=physics.left, target=ball_vec)

        vel_ball_xy_offset = planar_angle(reference=physics.linear_velocity, normal=np.array([0, 0, 1]), target=ball_vec)
        vel_ball_z_offset = planar_angle(
            reference=physics.linear_velocity, normal=np.cross(physics.linear_velocity, np.array([0, 0, 1])), target=ball_vec
        )
        angles = fourier_encoder(
            -np.pi,
            np.pi,
            np.array([yaw_offset, pitch_offset, vel_ball_xy_offset, vel_ball_z_offset]),
            frequencies=3,
            periodic=True,
        ).flatten()  # 4*2*3=24
        displacement = fourier_encoder(0, 2 * cv.BACK_WALL_Y, ball_vec, frequencies=4, periodic=False).flatten()  # 3*2*4=24
        distance = fourier_encoder(0, 2 * cv.BACK_WALL_Y, float(norm(ball_vec)), frequencies=2, periodic=False).flatten()  # 2*1*2=4
        return np.concatenate((angles, displacement, distance))  # 52

    def _relative_ball_obs(self, ball: PhysicsObject) -> np.ndarray:
        """Relative values for the ball"""
        posts = np.array(
            [
                [-cv.GOAL_CENTER_TO_POST, cv.BACK_WALL_Y, 0],
                [cv.GOAL_CENTER_TO_POST, cv.BACK_WALL_Y, 0],
                [-cv.GOAL_CENTER_TO_POST, -cv.BACK_WALL_Y, 0],
                [cv.GOAL_CENTER_TO_POST, -cv.BACK_WALL_Y, 0],
            ]
        )
        ball2posts = posts - ball.position
        post_angles = []
        for target in ball2posts:
            post_angles.append(planar_angle(ball.linear_velocity, np.array([0, 0, 1]), target=target))
        angles = fourier_encoder(-np.pi, np.pi, np.array(post_angles), frequencies=2, periodic=True).flatten()  # 4*2*2
        return angles  # 16

    def _relative_pads(self, physics: PhysicsObject) -> np.ndarray:
        """idk about this..."""
        pad_vecs = np.array(cv.BOOST_LOCATIONS) - physics.position
        offsets = []
        for target in pad_vecs:
            offsets.append(planar_angle(physics.linear_velocity, np.array([0, 0, 1]), target))
        offsets = fourier_encoder(-np.pi, np.pi, np.array(offsets), frequencies=1, periodic=True).flatten()  # 34 * 2
        distances = norm(pad_vecs, axis=-1) / (2 * cv.BACK_WALL_Y)  # 34
        return np.concatenate((offsets, distances))  # 102

    def _get_mask(self, car: Car):
        if not car.on_ground:
            throttle_mask = np.array([0, 0, 1])
            pitch_mask = np.ones(5)
            roll_mask = np.ones(3)
            hand_break_mask = np.array([0, 1])
        else:
            throttle_mask = np.ones(3)
            pitch_mask = np.array([0, 0, 1, 0, 0])
            roll_mask = np.array([0, 1, 0])
            hand_break_mask = np.ones(2)

        steer_yaw_mask = np.ones(5)
        if not (car.on_ground or car.has_flip):
            jump_mask = np.array([1, 0])
        else:
            jump_mask = np.ones(2)
        if car.boost_amount > 0:
            boost_mask = np.ones(2)
        else:
            boost_mask = np.array([1, 0])

        return np.concatenate((throttle_mask, steer_yaw_mask, pitch_mask, roll_mask, jump_mask, boost_mask, hand_break_mask)).astype("int")
