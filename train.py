from omegaconf import DictConfig
import hydra
from ray.rllib.algorithms import PPOConfig
from ray import tune
from conf.algorithm import build_config


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(hydra_cfg: DictConfig):
    algo_config = build_config(PPOConfig, hydra_cfg.algorithm)
    tuner = tune.Tuner(
        algo_config.algo_class,
        param_space=algo_config,
        run_config=tune.RunConfig(
            name="denbot_1on0",
            storage_path="~/src/denbot-rl/ray_results",
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=100,
                checkpoint_at_end=True,
            ),
        ),
    )

    return tuner.fit()


if __name__ == "__main__":
    main()
