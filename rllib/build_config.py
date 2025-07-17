from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.utils.typing import AgentID, EpisodeType, PolicyID

from env.env import RLEnv


def one2one(agent_id: AgentID, episode: EpisodeType, **kwargs) -> PolicyID:
    return agent_id


def build_algorithm_config(config: dict):
    """Build up an rllib algorithm config"""

    algo_config: AlgorithmConfig = config.pop("base_config")
    algo_config.environment(env=RLEnv)

    if rl_module_config := config.pop("rl_modules", None):
        mrlm = MultiRLModuleSpec(rl_module_specs={key: RLModuleSpec(module_class=val) for key, val in rl_module_config.items()})
        algo_config.rl_module(rl_module_spec=mrlm)

    for key, val in config.items():
        getattr(algo_config, key)(**val)

    return algo_config
