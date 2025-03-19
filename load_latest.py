import glob
from ray.rllib.core.rl_module import RLModule
import os
from pathlib import Path
from env.env import create_env
from time import sleep
import torch


def get_latest_module() -> RLModule:
    dir_str = "ray_results/**/rl_module/"
    all_checkpoints = glob.glob(dir_str, recursive=True)
    most_recent_checkpoint = max(all_checkpoints, key=os.path.getmtime)

    return RLModule.from_checkpoint(path=Path(most_recent_checkpoint).absolute())


def run_episode(module: RLModule):
    env = create_env()
    obs, _ = env.reset()
    obs = {"denbot": {"obs": torch.tensor(obs["blue-0"])}}
    dones = {"__all__": False}
    truncs = {"__all__": False}
    while not dones["__all__"] and not truncs["__all__"]:
        action = torch.distributions.Categorical(
            logits=module.forward_inference(obs)["denbot"]["action_dist_inputs"]
        ).sample()
        obs, _, dones, truncs, _ = env.step({"blue-0": action})
        obs = {"denbot": {"obs": torch.tensor(obs["blue-0"])}}
        env.render()
        sleep(0.01)

    env.close()


if __name__ == "__main__":
    rl_module = get_latest_module()
    run_episode(rl_module)
