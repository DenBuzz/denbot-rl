import glob
import os
from pathlib import Path
from time import sleep, time

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from ray.rllib.connectors.env_to_module import EnvToModulePipeline
from ray.rllib.connectors.module_to_env import ModuleToEnvPipeline
from ray.rllib.core import (
    COMPONENT_ENV_RUNNER,
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
    COMPONENT_RL_MODULE,
    Columns,
)
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

from env.env import RLEnv
from rllib.build_config import build_algorithm_config, denbot_map


def create_env(exp: str):
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="train", overrides=[f"exp={exp}"])
    config: dict = OmegaConf.to_container(instantiate(cfg))
    exp_config = config["exp"]
    exp_config["algorithm"] = build_algorithm_config(exp_config["algorithm"])

    return RLEnv(exp_config["algorithm"].env_config)


def get_scenarios():
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="train")

    return cfg.scenarios


def get_most_recent_checkpoint() -> Path:
    dir_str = "ray_results/**/checkpoint_*/"
    all_checkpoints = glob.glob(dir_str, recursive=True)
    most_recent_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    print(f"Found: {most_recent_checkpoint}")

    return Path(most_recent_checkpoint).absolute()


def load_components_from_checkpoint(path) -> tuple[RLModule, EnvToModulePipeline, ModuleToEnvPipeline]:
    rl_module = RLModule.from_checkpoint(Path(path, COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER, COMPONENT_RL_MODULE))
    env_to_module = EnvToModulePipeline.from_checkpoint(
        Path(path, COMPONENT_ENV_RUNNER, COMPONENT_ENV_TO_MODULE_CONNECTOR)
    )
    module_to_env = ModuleToEnvPipeline.from_checkpoint(
        Path(
            path,
            COMPONENT_ENV_RUNNER,
            COMPONENT_MODULE_TO_ENV_CONNECTOR,
        )
    )
    return rl_module, env_to_module, module_to_env


def run_episode(
    env: RLEnv, rl_module: RLModule, env_to_module: EnvToModulePipeline, module_to_env: ModuleToEnvPipeline
):
    obs, _ = env.reset()
    start_time = time()
    episode = MultiAgentEpisode(
        observations=[obs],
        agent_to_module_mapping_fn=denbot_map,
        # observation_space=env.observation_space,
        # action_space=env.action_space,
    )

    steps = 0
    while not episode.is_done:
        shared_data = {}
        input_dict = env_to_module(episodes=[episode], rl_module=rl_module, explore=False, shared_data=shared_data)
        rl_module_out = rl_module.forward_inference(input_dict)
        to_env = module_to_env(
            batch=rl_module_out, episodes=[episode], rl_module=rl_module, explore=True, shared_data=shared_data
        )
        action = to_env.pop(Columns.ACTIONS)[0]

        obs, reward, terminated, truncated, _ = env.step(action)
        episode.add_env_step(obs, action, reward, terminateds=terminated, truncateds=truncated)

        env.render()
        sleep(max(0, start_time + steps / 15 - time()))
        steps += 1


if __name__ == "__main__":
    # env = create_env("just_air_dribble")
    # while True:
    #     most_recent_checkpoint = get_most_recent_checkpoint()
    #     rl_module, env_to_module, module_to_env = load_components_from_checkpoint(most_recent_checkpoint)
    #     run_episode(env, rl_module, env_to_module, module_to_env)
    print(get_scenarios())
