from ray.rllib.algorithms import AlgorithmConfig

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
        .training(**config.training)
        .env_runners(**config.env_runners)
        .learners(**config.learners)
        .multi_agent(policies={"denbot"}, policy_mapping_fn=mapping_fn)
        .rl_module(
            model_config={
                "fcnet_hiddens": [256, 256, 256],
                "fcnet_activation": "LeakyReLU",
                # "value-fcnet_hiddens": [256],
                # "value-fcnet_activation": "LeakyReLU",
                # "policy-fcnet_hiddens": [256],
                # "policy-fcnet_activation": "LeakyReLU",
            }
        )
    )
    return algo_config
