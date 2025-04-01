from time import sleep

import pytest

from env import RLEnv
from load_latest import create_env


@pytest.fixture
def env_fixture():
    return create_env("airial")


def test_random_episode(env_fixture: RLEnv):
    """
    Test random episode in the environment.

    This function performs a single episode in the environment by repeatedly sampling random actions for each agent.
    It continues until the episode is done or truncated, rendering the environment at each step and pausing briefly to allow observation of changes.
    Finally, it closes the environment fixture.
    """
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
