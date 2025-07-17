from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.sim import RocketSimEngine

from env.env_components import ActionParser, ObsBuilder, RewardFunction, SimulationSetter, TerminalCondition


class RLEnv(MultiAgentEnv):
    agents: list[AgentID]
    possible_agents: list[AgentID]

    sim: RocketSimEngine
    state: GameState
    obs_builder: ObsBuilder
    action_parser: ActionParser
    sim_setter: SimulationSetter
    reward_function: RewardFunction
    termination_condition: TerminalCondition
    truncation_condition: TerminalCondition
    renderer: RLViserRenderer

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.load_config(config=config)
        self.possible_agents = []
        for i in range(3):
            self.possible_agents.append(f"blue-{i}")
            self.possible_agents.append(f"orange-{i}")
        self.agents = self.sim_setter.agents
        self.observation_spaces = self.obs_builder.observation_space
        self.action_spaces = self.action_parser.action_space
        self.sim = RocketSimEngine()
        self.renderer = RLViserRenderer()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[MultiAgentDict, MultiAgentDict]:
        super().reset(seed=seed, options=options)
        info = {}
        self.sim_setter.reset(info)
        self.reward_function.reset(info)
        self.obs_builder.reset(info)
        self.action_parser.reset(info)
        self.termination_condition.reset(info)
        self.truncation_condition.reset(info)

        self.state = self.sim_setter.apply(self.sim)
        self.agents = self.sim_setter.agents
        obs = self.obs_builder.build_obs(state=self.state)
        return obs, {}

    def step(self, action_dict: MultiAgentDict) -> tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        engine_actions = self.action_parser.parse_action(action_dict=action_dict)
        self.state = self.sim.step(actions=engine_actions, shared_info={})
        obs = self.obs_builder.build_obs(state=self.state)

        is_terminated = self.termination_condition.is_done(state=self.state)
        is_terminated["__all__"] = all(is_terminated.values())

        is_truncated = self.truncation_condition.is_done(state=self.state)
        is_truncated["__all__"] = all(is_truncated.values())

        rewards = self.reward_function.get_reward(state=self.state)
        return obs, rewards, is_terminated, is_truncated, {}

    def load_config(self, config: dict) -> None:
        self.config.update(config)
        self.obs_builder = self.config["obs_builder"]
        self.action_parser = self.config["action_parser"]
        self.sim_setter = self.config["sim_setter"]
        self.reward_function = self.config["reward_function"]
        self.termination_condition = self.config["termination_condition"]
        self.truncation_condition = self.config["truncation_condition"]

    def close(self):
        self.sim.close()

    def render(self) -> None:
        return self.renderer.render(self.state, {})
