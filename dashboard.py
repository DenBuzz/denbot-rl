import glob
import os
from pathlib import Path

import streamlit as st

from conf.build_config import load_configs
from env.env import RLEnv
from load_latest import load_components_from_checkpoint, run_episode


def get_trials() -> list[str]:
    trial_str = "ray_results/*/*/"
    all_trials = glob.glob(trial_str, recursive=True)
    trials = sorted(all_trials, key=os.path.getmtime, reverse=True)

    return trials


def get_checkpoints(trial_path: str) -> list[str]:
    checkpoint_str = f"{trial_path}/checkpoint_*/"
    all_checkpoints = glob.glob(checkpoint_str, recursive=True)
    checkpoints = sorted(all_checkpoints, key=os.path.getmtime, reverse=True)

    return checkpoints


def play_episode(env_config, checkpoint_path):
    rl_module, env_to_module, module_to_env = load_components_from_checkpoint(Path(checkpoint_path).absolute())
    env = RLEnv(config=env_config)
    run_episode(env, rl_module, env_to_module, module_to_env)
    # env.close()


st.title("DenBot Dashboard")

configs = load_configs()
env_configs = configs.exp.env_configs

trial = st.selectbox("Select a trial", get_trials())
checkpoint = st.selectbox("Select a checkpoint", get_checkpoints(trial))
exp = st.selectbox("Select an experiment", env_configs.keys())

st.button("Play Episode", on_click=play_episode, args=[env_configs[exp], checkpoint])
