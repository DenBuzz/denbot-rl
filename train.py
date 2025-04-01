from datetime import datetime as dt

import hydra
import ray
from omegaconf import DictConfig
from ray.tune import CheckpointConfig, RunConfig, TuneConfig, Tuner
from ray.tune.experiment.trial import Trial

from conf.build_config import build_exp_config
from training.stoppers import CurriculumStopper


def dirname_fn(trial: Trial):
    return f"{dt.now().strftime('%Y-%m-%d-%H-%M-%S')}-{trial.trial_id}"


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(hydra_cfg: DictConfig):
    algo_config = build_exp_config(hydra_cfg.exp)
    tuner = Tuner(
        algo_config.algo_class,
        param_space=algo_config,
        run_config=RunConfig(
            name="denbot_1on0",
            storage_path="~/src/denbot-rl/ray_results",
            checkpoint_config=CheckpointConfig(
                num_to_keep=5,
                checkpoint_frequency=50,
                checkpoint_at_end=True,
            ),
            stop=CurriculumStopper(),
        ),
        tune_config=TuneConfig(num_samples=1, trial_dirname_creator=dirname_fn),
    )

    ray.init(**hydra_cfg.ray_init)
    return tuner.fit()


if __name__ == "__main__":
    main()
