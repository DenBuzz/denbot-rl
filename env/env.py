from typing import Any

from hydra import compose, initialize
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    MutatorSequence,
)

from env.action_parsers import RepeatAction
from env.action_parsers.seer_action import SeerActionParser
from env.denbot_reward import DenBotReward
from env.obs_builders import DefaultObs
from env.state_mutators.random import RandomBallLocation, RandomCarLocation
from env.termination_conditions.ball_touch_termination import BallTouchTermination


class RLEnv(MultiAgentEnv):
    """
    The main RLGym class. This class is responsible for managing the environment and the interactions between
    the different components of the environment. It is the main interface for the user to interact with an environment.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_mutator = MutatorSequence(
            FixedTeamSizeMutator(blue_size=config["blue_size"], orange_size=config["orange_size"]),
            # KickoffMutator(),
            RandomBallLocation(),
            RandomCarLocation(),
        )
        self.obs_builder = DefaultObs(num_cars=config["blue_size"] + config["orange_size"], **config["obs_builder"])
        self.action_parser = RepeatAction(SeerActionParser())
        self.reward_fn = DenBotReward(**config["rewards"])
        self.termination_cond = BallTouchTermination()
        self.truncation_cond = AnyCondition(
            TimeoutCondition(timeout_seconds=config["timeout_seconds"]),
            NoTouchTimeoutCondition(timeout_seconds=config["no_touch_timeout_seconds"]),
        )
        self.renderer = RLViserRenderer()
        self.sim = RocketSimEngine()
        self.possible_agents = []
        for i in range(config["blue_size"]):
            self.possible_agents.append(f"blue-{i}")
        for i in range(config["orange_size"]):
            self.possible_agents.append(f"orange-{i}")

        self.action_spaces = {agent: self.action_parser.get_action_space(agent) for agent in self.possible_agents}
        self.observation_spaces = {agent: self.obs_builder.get_obs_space(agent) for agent in self.possible_agents}

    @property
    def state(self) -> GameState:
        return self.sim.state

    def set_state(self, desired_state: GameState) -> dict[str, Any]:
        state = self.sim.set_state(desired_state, {})
        agents = self.agents
        return self.obs_builder.build_obs(agents, state, {})

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        initial_state = self.sim.create_base_state()
        self.state_mutator.apply(initial_state, {})
        state = self.sim.set_state(initial_state, {})

        agents = self.agents = self.sim.agents
        self.obs_builder.reset(agents, state, {})
        self.action_parser.reset(agents, state, {})
        self.termination_cond.reset(agents, state, {})
        self.truncation_cond.reset(agents, state, {})

        return self.obs_builder.build_obs(agents, state), {}

    def step(
        self, action_dict: MultiAgentDict
    ) -> tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        engine_actions = self.action_parser.parse_actions(action_dict, self.state, {})
        new_state = self.sim.step(engine_actions, {})
        agents = self.agents
        obs = self.obs_builder.build_obs(agents, new_state)
        is_terminated = self.termination_cond.is_done(agents, new_state, {})
        if all(is_terminated.values()):
            is_terminated["__all__"] = True
        else:
            is_terminated["__all__"] = False
        is_truncated = self.truncation_cond.is_done(agents, new_state, {})
        if all(is_truncated.values()):
            is_truncated["__all__"] = True
        else:
            is_truncated["__all__"] = False
        rewards = {agent: self.reward_fn.apply(agent, new_state) for agent in agents}
        return obs, rewards, is_terminated, is_truncated, {}

    def render(self) -> Any:
        self.renderer.render(self.state, {})
        return True

    def close(self) -> None:
        self.sim.close()
        if self.renderer is not None:
            self.renderer.close()


def create_env():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="train")

    return RLEnv(cfg.algorithm.env_config)
