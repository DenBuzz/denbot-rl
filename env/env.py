from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from env.obs_builders import DefaultObs
from env.action_parsers import LookupTableAction, RepeatAction
from env.rewards_functions import BallProximityReward

from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from typing import Any
from hydra import initialize, compose
from ray.rllib.env import MultiAgentEnv


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
            KickoffMutator(),
        )
        self.obs_builder = DefaultObs(num_cars=config["blue_size"] + config["orange_size"])
        self.action_parser = RepeatAction(LookupTableAction())
        self.reward_fn = CombinedReward(
            (GoalReward(), config["goal_reward"]),
            (TouchReward(), config["touch_reward"]),
            (BallProximityReward(), config["ball_proximity_reward"]),
        )
        self.termination_cond = GoalCondition()
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
        self.reward_fn.reset(agents, state, {})

        return self.obs_builder.build_obs(agents, state, {}), {}

    def step(self, action: dict[str, Any]) -> Any:
        engine_actions = self.action_parser.parse_actions(action, self.state, {})
        new_state = self.sim.step(engine_actions, {})
        agents = self.agents
        obs = self.obs_builder.build_obs(agents, new_state, {})
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
        rewards = self.reward_fn.get_rewards(agents, new_state, is_terminated, is_truncated, {})
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
