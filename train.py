from datetime import datetime as dt

import hydra

# import ray
from omegaconf import DictConfig
from ray.rllib.algorithms import PPOConfig
from ray.tune import CheckpointConfig, RunConfig, TuneConfig, Tuner
from ray.tune.experiment.trial import Trial
from ray.tune.stopper import MaximumIterationStopper

from conf.algorithm import build_config


def dirname_fn(trial: Trial):
    return f"{dt.now().strftime('%Y-%m-%d-%H-%M-%S')}-{trial.trial_id}"


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(hydra_cfg: DictConfig):
    algo_config = build_config(PPOConfig, hydra_cfg.exp)
    tuner = Tuner(
        algo_config.algo_class,
        param_space=algo_config,
        run_config=RunConfig(
            name="denbot_1on0",
            storage_path="~/src/denbot-rl/ray_results",
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=20,
                checkpoint_at_end=True,
            ),
            stop=MaximumIterationStopper(max_iter=20_000),
        ),
        tune_config=TuneConfig(num_samples=1, trial_dirname_creator=dirname_fn),
    )

    # ray.init(local_mode=True)
    return tuner.fit()


if __name__ == "__main__":
    main()
