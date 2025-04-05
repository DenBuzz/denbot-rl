from typing import Any

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from ray.rllib.algorithms import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback

from env import RLEnv


def load_configs():
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="train")

    return instantiate(cfg)


def mapping_fn(agent_id: str, episode):
    return "denbot"


def multi_callback(callbacks):
    callback_list = [type(c) for c in list(callbacks.values())]

    class Callback(*callback_list, RLlibCallback): ...

    return Callback


def build_exp_config(config):
    config = instantiate(config)
    algo_cfg: dict[str, Any] = OmegaConf.to_container(config.algorithm)
    ppo_config = (
        PPOConfig()
        .environment(env=RLEnv, env_config=config.env_configs[config.curriculum[0]], **algo_cfg["environment"])
        .training(**algo_cfg["training"])
        .env_runners(**algo_cfg["env_runners"])
        .learners(**algo_cfg["learners"])
        .multi_agent(policies={"denbot"}, policy_mapping_fn=mapping_fn)
        .rl_module(**algo_cfg["rl_module"])
        .callbacks(callbacks_class=multi_callback(algo_cfg["callbacks"]))
    )
    return ppo_config
