from datetime import datetime as dt

import hydra
import ray
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray.tune import CheckpointConfig, RunConfig, Tuner
from ray.tune.experiment.trial import Trial

from rllib.build_config import build_algorithm_config


def dirname_fn(trial: Trial):
    return f"{dt.now().strftime('%Y-%m-%d-%H-%M-%S')}-{trial.trial_id}"


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):
    config: dict = OmegaConf.to_container(instantiate(cfg))
    exp_config = config["exp"]
    exp_config["algorithm"] = build_algorithm_config(exp_config["algorithm"])

    checkpoint_config = CheckpointConfig(**exp_config["run_config"].pop("checkpoint_config", {}))
    exp_config["run_config"] = RunConfig(**exp_config["run_config"], checkpoint_config=checkpoint_config)

    ray.init(**exp_config["ray_init"])
    tuner = Tuner(
        exp_config["algorithm"]["algo_class"],
        param_space=exp_config["algorithm"],
        run_config=exp_config["run_config"],
    )
    tuner.fit()


if __name__ == "__main__":
    main()
