from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec

from env import RLEnv
from training.callbacks import CurriculumCallback


def load_configs():
    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="train")

    return instantiate(cfg)


def mapping_fn(agent_id: str, episode):
    return "denbot"


def multi_callback(callbacks, curriculum_cfg):
    callback_list = [type(c) for c in list(callbacks.values())]

    # This is kinda whack
    class Callback(*callback_list, CurriculumCallback):
        curriculum_config = curriculum_cfg

    return Callback


def build_exp_config(config):
    config = instantiate(config)
    algo_cfg = config.algorithm
    ppo_config = (
        PPOConfig()
        .environment(env=RLEnv, env_config=config.env_config, **algo_cfg.environment)
        .training(**OmegaConf.to_container(algo_cfg.training))
        .env_runners(**algo_cfg.env_runners)
        .learners(**algo_cfg.learners)
        .multi_agent(policies={"denbot"}, policy_mapping_fn=mapping_fn)
        .rl_module(rl_module_spec=RLModuleSpec(**OmegaConf.to_container(algo_cfg.rl_module.rl_module_spec)))
        .callbacks(callbacks_class=multi_callback(algo_cfg.callbacks, config.env_config.curriculum))
        .evaluation(**algo_cfg.evaluation)
    )
    return ppo_config
