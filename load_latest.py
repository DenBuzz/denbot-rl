import glob
from ray.rllib.core.rl_module import RLModule
import os
from pathlib import Path
from env.env import RLEnv, create_env
from time import sleep, time
import torch


def get_latest_module() -> RLModule:
    dir_str = "ray_results/**/rl_module/"
    all_checkpoints = glob.glob(dir_str, recursive=True)
    most_recent_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    print(f"loading: {most_recent_checkpoint}")

    return RLModule.from_checkpoint(path=Path(most_recent_checkpoint).absolute())


def sample_action(action_dist_inputs, space):
    action = []
    i = 0
    for n_logits in space.nvec:
        action.append(torch.distributions.Categorical(logits=action_dist_inputs[i : i + n_logits]).sample())
        i += n_logits
    return action


def run_episode(env: RLEnv, module: RLModule):
    obs, _ = env.reset()
    start_time = time()

    obs = {"denbot": {"obs": torch.tensor(obs["blue-0"])}}
    dones = {"__all__": False}
    truncs = {"__all__": False}
    steps = 0
    while not dones["__all__"] and not truncs["__all__"]:
        action = sample_action(
            action_dist_inputs=module.forward_inference(obs)["denbot"]["action_dist_inputs"],
            space=env.action_spaces["blue-0"],
        )
        obs, _, dones, truncs, _ = env.step({"blue-0": action})
        obs = {"denbot": {"obs": torch.tensor(obs["blue-0"])}}
        env.render()
        sleep(max(0, start_time + steps / 15 - time()))
        steps += 1


if __name__ == "__main__":
    env = create_env()
    while True:
        rl_module = get_latest_module()
        run_episode(env, rl_module)
