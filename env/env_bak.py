from collections import defaultdict
from typing import Any

import numpy as np
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.sim import RocketSimEngine

from env.action_parser import SeerAction
from env.denbot_reward import DenBotReward
from env.observers.denbot_obs import DenbotObs


class RLEnv(MultiAgentEnv):
    """
    The main RLGym class. This class is responsible for managing the environment and the interactions between
    the different components of the environment. It is the main interface for the user to interact with an environment.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.envs = config["envs"]
        self.curriculum = config["curriculum"]
        self.meta_task = 0
        self.env_tasks = defaultdict(int)

        self.obs_builder = DenbotObs()
        self.action_parser = SeerAction(repeats=8)
        self.renderer = RLViserRenderer()

        self.sim = RocketSimEngine()
        self.possible_agents = []
        for i in range(3):
            self.possible_agents.append(f"blue-{i}")
            self.possible_agents.append(f"orange-{i}")

        self.action_spaces = {agent: self.action_parser.get_action_space(agent) for agent in self.possible_agents}
        self.observation_spaces = {agent: self.obs_builder.get_obs_space(agent) for agent in self.possible_agents}

    @property
    def state(self) -> GameState:
        return self.sim.state

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self._load_task()
        self.state_mutator.reset(self.shared_info)
        self.reward_fn.reset(self.shared_info)
        self.termination_cond.reset(self.shared_info)
        self.truncation_cond.reset(self.shared_info)
        self.obs_builder.reset(self.shared_info)

        initial_state = self.sim.create_base_state()
        self.state_mutator.apply(initial_state, self.sim)
        state = self.sim.set_state(initial_state, {})

        agents = self.agents = self.sim.agents
        return self.obs_builder.build_obs(agents, state), {}

    def step(self, action_dict: MultiAgentDict) -> tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        engine_actions = self.action_parser.parse_actions(action_dict, self.state)
        new_state = self.sim.step(engine_actions, {})
        agents = self.agents
        obs = self.obs_builder.build_obs(agents, new_state)
        is_terminated = self.termination_cond.is_done(agents, new_state)
        if all(is_terminated.values()):
            is_terminated["__all__"] = True
        else:
            is_terminated["__all__"] = False
        is_truncated = self.truncation_cond.is_done(agents, new_state)
        if all(is_truncated.values()):
            is_truncated["__all__"] = True
        else:
            is_truncated["__all__"] = False
        rewards = {agent: self.reward_fn.apply(agent, new_state) for agent in agents}
        return obs, rewards, is_terminated, is_truncated, {}

    def _load_task(self) -> None:
        meta_task_config = self.curriculum["tasks"][self.meta_task]
        next_env = np.random.choice(meta_task_config["envs"])
        env_config = self.envs[next_env]

        self.shared_info = {"task": self.env_tasks[next_env], "env": next_env}

        self.state_mutator = env_config["state_mutator"]
        self.termination_cond = env_config["termination_cond"]
        self.truncation_cond = env_config["truncation_cond"]
        self.reward_fn = DenBotReward(**env_config["rewards"])

    def render(self) -> Any:
        self.renderer.render(self.state, {})
        return True

    def set_tasks(self, task: int, tasks: dict[str, int] | None = None) -> None:
        if tasks:
            self.env_tasks = defaultdict(int)
            self.env_tasks.update(tasks)
        self.meta_task = task

    def close(self) -> None:
        self.sim.close()
        if self.renderer is not None:
            self.renderer.close()
