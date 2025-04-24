import glob
import os
from pathlib import Path

import dash
import torch
from dash import dcc, html
from dash.dependencies import Input, Output, State

from conf.build_config import load_configs
from env.env import RLEnv
from load_latest import load_components_from_checkpoint, run_episode

torch.classes.__path__ = []

# Initialize the Dash app
app = dash.Dash(__name__)

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

# Load configurations
configs = load_configs()
env_config = configs.exp.env_config

# Define the app layout
app.layout = html.Div([
    html.H1("DenBot Dashboard"),

    dcc.Dropdown(
        id='trial-dropdown',
        options=[{'label': Path(t).name, 'value': t} for t in get_trials()],
        value=get_trials()[0] if get_trials() else None,
        multi=False
    ),

    dcc.Dropdown(
        id='checkpoint-dropdown',
        value=None,
        multi=False
    ),

    dcc.Dropdown(
        id='env-dropdown',
        options=[{'label': e, 'value': e} for e in env_config.envs.keys()],
        value=list(env_config.envs.keys())[0],
        multi=False
    ),

    dcc.Checklist(
        id='loop-checkbox',
        options=[{'label': 'Loop', 'value': 'loop'}],
        value=[]
    ),

    html.Button('Play Episode', id='play-button'),

    html.Div(id='output-container')
])

# Callback to update checkpoints dropdown based on selected trial
@app.callback(
    Output('checkpoint-dropdown', 'options'),
    [Input('trial-dropdown', 'value')]
)
def update_checkpoints(selected_trial):
    if selected_trial:
        checkpoints = get_checkpoints(selected_trial)
        return [{'label': Path(c).name, 'value': c} for c in checkpoints]
    return []

# Callback to play episode
@app.callback(
    Output('output-container', 'children'),
    [Input('play-button', 'n_clicks')],
    [State('trial-dropdown', 'value'),
     State('checkpoint-dropdown', 'value'),
     State('env-dropdown', 'value'),
     State('loop-checkbox', 'value')]
)
def play_episode(n_clicks, selected_trial, selected_checkpoint, selected_env, loop_value):
    if n_clicks is not None and n_clicks > 0:
        env_config.curriculum["tasks"] = [{"envs": [selected_env]}]
        loop = 'loop' in loop_value
        rl_module, env_to_module, module_to_env = load_components_from_checkpoint(Path(selected_checkpoint).absolute())
        env = RLEnv(config=env_config)
        run_episode(env, rl_module, env_to_module, module_to_env)
        while loop:
            run_episode(env, rl_module, env_to_module, module_to_env)
            loop = 'loop' in loop_value  # Check the loop state after each episode
    return f"Played episode with {selected_env} environment."

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
