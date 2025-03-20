from omegaconf import DictConfig
import hydra
from ray.rllib.algorithms import PPOConfig
from ray.tune import Tuner, RunConfig, TuneConfig, CheckpointConfig
from ray.tune.stopper import MaximumIterationStopper
from conf.algorithm import build_config


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(hydra_cfg: DictConfig):
    algo_config = build_config(PPOConfig, hydra_cfg.algorithm)
    tuner = Tuner(
        algo_config.algo_class,
        param_space=algo_config,
        run_config=RunConfig(
            name="denbot_1on0",
            storage_path="~/src/denbot-rl/ray_results",
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=100,
                checkpoint_at_end=True,
            ),
            stop=MaximumIterationStopper(max_iter=10_000),
        ),
        tune_config=TuneConfig(num_samples=1),
    )

    return tuner.fit()


if __name__ == "__main__":
    main()
