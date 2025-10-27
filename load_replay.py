import time

from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym_tools.rocket_league.replays.convert import ReplayFrame, replay_to_rlgym
from rlgym_tools.rocket_league.replays.parsed_replay import ParsedReplay

# Parse replay file
replay = ParsedReplay.load("replays/968df055-66c7-4ffb-b820-08774c7c6acf.replay")

# Convert to ReplayFrame sequence
replay_frames = replay_to_rlgym(replay)

# renderer = ReplayRenderer(replay=replay)
renderer = RLViserRenderer(tick_rate=30)

frame: ReplayFrame = next(replay_frames)

start_time = time.time()
episode_seconds_remaining = frame.episode_seconds_remaining

state = frame.state
renderer.render(state, {})

for frame in replay_frames:
    state = frame.state
    episode_elapsed_time = episode_seconds_remaining - frame.episode_seconds_remaining

    if time.time() - start_time < episode_elapsed_time:
        time.sleep(episode_elapsed_time - (time.time() - start_time))
    renderer.render(state, {})

renderer.close()
