from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf
from ray.rllib.algorithms import PPOConfig

from env import RLEnv


def mapping_fn(agent_id: str, episode):
    return "denbot"


def build_exp_config(config):
    config = instantiate(config)
    algo_cfg: dict[str, Any] = OmegaConf.to_container(config.algorithm)
    ppo_config = (
        PPOConfig()
        .environment(env=RLEnv, **algo_cfg["environment"])
        .training(**algo_cfg["training"])
        .env_runners(**algo_cfg["env_runners"])
        .learners(**algo_cfg["learners"])
        .multi_agent(policies={"denbot"}, policy_mapping_fn=mapping_fn)
        .rl_module(**algo_cfg["rl_module"])
        .callbacks(**algo_cfg["callbacks"])
    )
    return ppo_config
