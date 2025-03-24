# from pathlib import Path

from ray.rllib.algorithms import AlgorithmConfig

# from ray.rllib.core.rl_module import RLModuleSpec
from conf.callbacks import AirialCurriculumCallback
from env import RLEnv


def mapping_fn(agent_id: str, episode):
    return "denbot"


def build_config(algo_class: type[AlgorithmConfig], config):
    algo_config = (
        algo_class()
        .environment(
            env=RLEnv,
            env_config=config.env_config,
        )
        .training(
            entropy_coeff=[(0, 0.02), (5e7, 0.0001)],
            **config.training,
        )
        .env_runners(**config.env_runners)
        .learners(**config.learners)
        .multi_agent(policies={"denbot"}, policy_mapping_fn=mapping_fn)
        .rl_module(
            # rl_module_spec=RLModuleSpec(
            #     load_state_path=Path(
            #         "ray_results/denbot_1on0/2025-03-23-21-30-18-4f1f4_00000/checkpoint_000011/learner_group/learner/rl_module/denbot"
            #     ).absolute()
            # ),
            model_config={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "LeakyReLU",
                # "value-fcnet_hiddens": [256],
                # "value-fcnet_activation": "LeakyReLU",
                # "policy-fcnet_hiddens": [256],
                # "policy-fcnet_activation": "LeakyReLU",
            },
        )
        .callbacks(AirialCurriculumCallback)
    )
    return algo_config
