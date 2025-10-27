from typing import Any

import rlviser_py as rlviser
import RocketSim as rsim
from rlgym.rocket_league.api import Car
from rlgym.rocket_league.common_values import BOOST_LOCATIONS
from rlgym_tools.rocket_league.replays.convert import ParsedReplay


class ReplayRenderer:
    """
    A renderer that uses RLViser to render replays.
    """

    def __init__(self, replay: ParsedReplay):
        rlviser.set_boost_pad_locations(BOOST_LOCATIONS)
        self.replay = replay
        self.packet_id = 0

    def play_replay(self):
        for tick in self.replay.game_df.index:
            print(tick)

    def render_index(self, index: int, delay: bool = True) -> Any:
        game_data = self.replay.game_df.iloc[index]

        boost_pad_states = [bool(timer == 0) for timer in state.boost_pad_timers]

        ball = rsim.BallState()
        ball.pos = rsim.Vec(*state.ball.position)
        ball.vel = rsim.Vec(*state.ball.linear_velocity)
        ball.ang_vel = rsim.Vec(*state.ball.angular_velocity)
        ball.rot_mat = rsim.RotMat(*state.ball.rotation_mtx.transpose().flatten())

        car_data = []
        for car_id in cars_to_update:
            car = state.cars[car_id]
            car_state = self._get_car_state(car)
            car_data.append((int(car_id), car.team_num, rsim.CarConfig(car.hitbox_type), car_state))

        self.packet_id += 1
        # rlviser.render(
        #     tick_count=self.packet_id,
        #     tick_rate=self.tick_rate,
        #     game_mode=rsim.GameMode.SOCCAR,
        #     boost_pad_states=boost_pad_states,
        #     ball=ball,
        #     cars=car_data,
        # )

    def close(self):
        rlviser.quit()

    # I stole this from RocketSimEngine
    def _get_car_state(self, car: Car):
        car_state = rsim.CarState()
        car_state.pos = rsim.Vec(*car.physics.position)
        car_state.vel = rsim.Vec(*car.physics.linear_velocity)
        car_state.ang_vel = rsim.Vec(*car.physics.angular_velocity)
        car_state.rot_mat = rsim.RotMat(*car.physics.rotation_mtx.transpose().flatten())

        car_state.demo_respawn_timer = car.demo_respawn_timer
        car_state.is_on_ground = car.on_ground
        car_state.supersonic_time = car.supersonic_time
        car_state.boost = car.boost_amount
        car_state.time_spent_boosting = car.boost_active_time
        car_state.handbrake_val = car.handbrake

        car_state.has_jumped = car.has_jumped
        car_state.last_controls.jump = car.is_holding_jump
        car_state.is_jumping = car.is_jumping
        car_state.jump_time = car.jump_time

        car_state.has_flipped = car.has_flipped
        car_state.has_double_jumped = car.has_double_jumped
        car_state.air_time_since_jump = car.air_time_since_jump
        car_state.flip_time = car.flip_time
        car_state.flip_rel_torque = rsim.Vec(*car.flip_torque)

        car_state.is_auto_flipping = car.is_autoflipping
        car_state.auto_flip_timer = car.autoflip_timer
        car_state.auto_flip_torque_scale = car.autoflip_direction

        return car_state


# replay = ParsedReplay.load("path_to_replay.replay")
#
# # Get timeline ticks
# timeline = replay.game_df.index
# player_states = {pid: None for pid in replay.player_dfs.keys()}
# player_ages = {pid: 0 for pid in replay.player_dfs.keys()}
#
# for tick in timeline:
#     updated_players = []
#     for pid, df in replay.player_dfs.items():
#         if tick in df.index:
#             # Packet arrived: update state
#             player_states[pid] = df.loc[tick]
#             player_ages[pid] = 0
#             updated_players.append(pid)
#         else:
#             # No new data: age increases
#             player_ages[pid] += 1
#
#     # Ball update
#     if tick in replay.ball_df.index:
#         ball_state = replay.ball_df.loc[tick]
#     # else: keep previous ball_state
#
#     # Send updates to visualization (rlviser)
#     # viewer.update_players({pid: player_states[pid] for pid in updated_players})
#     # viewer.update_ball(ball_state)
#     # Optionally pass player_ages for "stale" data visualization
