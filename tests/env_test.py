from time import sleep

import pytest

from env import RLEnv
from env.env import create_env


@pytest.fixture
def env_fixture():
    return create_env()


def test_random_episode(env_fixture: RLEnv):
    obs, _ = env_fixture.reset()
    dones = {"__all__": False}
    truncs = {"__all__": False}
    while not dones["__all__"] and not truncs["__all__"]:
        action = {}
        for agent in obs:
            action[agent] = env_fixture.action_spaces[agent].sample()
        obs, _, dones, truncs, _ = env_fixture.step(action)
        env_fixture.render()
        sleep(0.02)

    env_fixture.close()


# def test_episode_render(env_fixture: RLEnv):
#     obs, _ = env_fixture.reset()
#     dones = {"__all__": False}
#     truncs = {"__all__": False}
#     while not dones["__all__"] and not truncs["__all__"]:
#         action = {}
#         for agent in obs:
#             action[agent] = env_fixture.action_spaces[agent].sample()
#         obs, _, dones, truncs, _ = env_fixture.step(action)
#         env_fixture.render()
#         sleep(0.05)
#
#     env_fixture.close()
