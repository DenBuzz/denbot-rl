import itertools
import os
from collections.abc import Iterable
from pathlib import Path

import rlviser_py
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Log,  # <-- Import Log
    Static,
    Tree,
)
from textual.widgets.tree import TreeNode
from textual.worker import Worker, get_current_worker  # <-- Import Worker utilities

from env.env import RLEnv
from load_latest import load_components_from_checkpoint, run_episode

# --- Configuration ---
RAY_RESULTS_PATH = "./ray_results"


# --- Helper Functions ---
def get_most_recent_checkpoint() -> Path | None:
    """Finds the most recently modified RLlib checkpoint directory."""
    p = Path(RAY_RESULTS_PATH)
    if not p.exists():
        return None
    all_checkpoints = list(p.rglob("checkpoint_*"))
    if not all_checkpoints:
        return None
    most_recent_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    return most_recent_checkpoint.absolute()


def get_scenarios() -> DictConfig:
    """Loads scenario configurations using Hydra."""
    try:
        with initialize(version_base=None, config_path="conf"):
            cfg = compose(config_name="train")
        return cfg.scenarios
    except Exception as e:
        # Return a dummy config if hydra fails, so the app doesn't crash
        print(f"Hydra initialization failed: {e}. Using dummy scenarios.")
        return DictConfig({"error": {"message": "Could not load 'conf/train.yaml'"}})


# --- Custom Widgets & Screens ---


# === Checkpoint Selection Components ===
class CheckpointSelector(DirectoryTree):
    """A DirectoryTree widget specialized for selecting RLlib checkpoint directories."""

    class CheckpointSelected(Message):
        def __init__(self, path: Path) -> None:
            super().__init__()
            self.path: Path = path

    def on_tree_node_selected(self, event: DirectoryTree.NodeSelected) -> None:
        event.stop()
        selected_path = event.node.data.path
        if selected_path.is_dir() and selected_path.name.startswith("checkpoint_"):
            self.post_message(self.CheckpointSelected(selected_path))

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        return [p for p in paths if p.is_dir()]


class CheckpointSelectScreen(Screen):
    """A modal screen for selecting a checkpoint."""

    BINDINGS = [("escape", "app.pop_screen", "Go Back")]

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Select a Checkpoint", id="select_dialog_title"),
            CheckpointSelector(RAY_RESULTS_PATH, id="checkpoint_selector_tree"),
            id="select_dialog",
        )

    def on_checkpoint_selector_checkpoint_selected(self, message: CheckpointSelector.CheckpointSelected) -> None:
        self.dismiss(message.path)


class CheckpointControls(Horizontal):
    """A widget group for checkpoint selection controls."""

    # This widget no longer holds the state, it just displays it.
    # The main app will be the source of truth.
    def compose(self) -> ComposeResult:
        yield Button("Get Latest", id="get_latest_btn", variant="primary")
        yield Button("Select...", id="select_checkpoint_btn")
        yield Static("No checkpoint selected.", id="checkpoint_path_label")


# === Scenario Selection Components ===
class ScenarioSelectScreen(Screen):
    """A modal screen for selecting a scenario."""

    BINDINGS = [("escape", "app.pop_screen", "Go Back")]

    def __init__(self, scenarios: DictConfig):
        super().__init__()
        self.scenarios = scenarios

    def compose(self) -> ComposeResult:
        with Horizontal(id="scenario_select_container"):
            with Vertical(classes="select_pane"):
                yield Label("Available Scenarios")
                yield ListView(
                    *[ListItem(Label(name), name=name) for name in self.scenarios.keys()], id="scenario_list"
                )
            with Vertical(classes="preview_pane"):
                yield Label("Configuration Preview")
                yield Tree("Config", id="scenario_preview")

    def _build_config_tree(self, data: dict | DictConfig, node: TreeNode) -> None:
        """Recursively build a Tree from a dictionary-like object."""
        node.expand()
        for key, value in data.items():
            if isinstance(value, dict | DictConfig):
                child_node = node.add(f"[b]{key}[/b]")
                self._build_config_tree(value, child_node)
            elif isinstance(value, list | ListConfig):
                child_node = node.add(f"[b]{key}[/b] ({len(value)} items)")
                child_node.expand()
                for index, item in enumerate(value):
                    if isinstance(item, dict | DictConfig):
                        self._build_config_tree(item, child_node.add(f"Item {index}"))
                    else:
                        child_node.add_leaf(str(item))
            else:
                node.add_leaf(f"[i]{key}[/i]: {value}")

    def _update_preview(self, scenario_name: str | None) -> None:
        """Clear and rebuild the configuration tree."""
        if scenario_name is None:
            return
        tree = self.query_one("#scenario_preview", Tree)
        config_data = self.scenarios.get(scenario_name, {})
        tree.clear()
        tree.root.set_label(f"Scenario: {scenario_name}")
        self._build_config_tree(config_data, tree.root)

    def on_mount(self) -> None:
        """Initially highlight the first item and populate the tree."""
        list_view = self.query_one(ListView)
        list_view.index = 0
        if list_view.highlighted_child:
            self._update_preview(list_view.highlighted_child.name)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Update the preview pane when a new scenario is highlighted."""
        if event.item:
            self._update_preview(event.item.name)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Dismiss the screen, returning the selected scenario name."""
        self.dismiss(event.item.name)


class ScenarioControls(Horizontal):
    """A widget group for scenario selection controls."""

    # This widget also no longer holds state.
    def compose(self) -> ComposeResult:
        yield Button("Select...", id="select_scenario_btn")
        yield Static("No scenario selected.", id="scenario_name_label")


# --- NEW: Inference Components ---


class InferenceControls(Static):
    """A widget group for starting and stopping inference."""

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Button("Start Inference", id="start_btn", variant="success", disabled=True)
            yield Button("Stop Inference", id="stop_btn", variant="error", disabled=True)
        yield Static("Status: Idle", id="inference_status_label")

    def on_mount(self) -> None:
        """Disable selection buttons when inference is running."""
        self.watch(self.app, "is_running_inference", self._toggle_buttons)

    def _toggle_buttons(self, is_running: bool) -> None:
        """Toggle button states based on inference status."""
        self.query_one("#start_btn", Button).disabled = is_running
        self.query_one("#stop_btn", Button).disabled = not is_running

        # Also disable selection buttons
        self.app.query_one("#get_latest_btn", Button).disabled = is_running
        self.app.query_one("#select_checkpoint_btn", Button).disabled = is_running
        self.app.query_one("#select_scenario_btn", Button).disabled = is_running


class InferenceLog(Message):
    """A message to send log updates from the worker to the main thread."""

    def __init__(self, line: str) -> None:
        super().__init__()
        self.line = line


# --- Main Application ---
class InferenceApp(App):
    """The main application for running inference."""

    CSS_PATH = "inference.css"
    TITLE = "DenBot Inference Tool"
    BINDINGS = [("q", "quit", "Quit")]

    # --- NEW: Reactive properties to hold the application state ---
    checkpoint_path = reactive[Path | None](None)
    scenario_name = reactive[str | None](None)
    is_running_inference = reactive(False)

    # This automatically becomes True when both a checkpoint and scenario are selected
    can_start_inference = reactive(False, init=False)

    def compute_can_start_inference(self) -> bool:
        """Determine if the start button should be enabled."""
        return self.checkpoint_path is not None and self.scenario_name is not None

    def watch_checkpoint_path(self, new_path: Path | None) -> None:
        """Called when the checkpoint_path reactive changes."""
        label = self.query_one("#checkpoint_path_label", Static)
        if new_path:
            label.update(f"Selected: .../{new_path.parent.name}/{new_path.name}")
        else:
            label.update("No checkpoint selected.")

    def watch_scenario_name(self, new_name: str | None) -> None:
        """Called when the scenario_name reactive changes."""
        label = self.query_one("#scenario_name_label", Static)
        if new_name:
            label.update(f"Selected: {new_name}")
        else:
            label.update("No scenario selected.")

    def watch_can_start_inference(self, can_start: bool) -> None:
        """Enable/disable the start button based on state."""
        # Only enable start button if not already running
        self.query_one("#start_btn", Button).disabled = not can_start or self.is_running_inference

    # --- End of new reactive properties ---

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="main_container"):
            yield Label("1. Checkpoint Selection", classes="section_label")
            yield CheckpointControls(id="checkpoint_controls")
            yield Label("2. Scenario Selection", classes="section_label")
            yield ScenarioControls(id="scenario_controls")

            # --- NEW: Inference Control and Log widgets ---
            yield Label("3. Run Inference", classes="section_label")
            yield InferenceControls(id="inference_controls")
            yield Log(id="inference_log", max_lines=200)
            # --- End of new widgets ---

        yield Footer()

    def on_mount(self) -> None:
        """Set theme on startup."""
        self.theme = "catppuccin-mocha"
        self.query_one("#inference_log", Log).write_line("Welcome! Select a checkpoint and scenario to begin.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle all button presses."""
        match event.button.id:
            case "get_latest_btn":
                latest_path = get_most_recent_checkpoint()
                if latest_path:
                    self.checkpoint_path = latest_path
                else:
                    self.query_one("#checkpoint_path_label", Static).update("No checkpoints found!")
            case "select_checkpoint_btn":
                self.push_screen(CheckpointSelectScreen(), self.update_checkpoint_path)
            case "select_scenario_btn":
                scenarios = get_scenarios()
                self.push_screen(ScenarioSelectScreen(scenarios), self.update_scenario_name)
            case "start_btn":
                self.run_inference_worker()
            case "stop_btn":
                self.stop_inference_worker()

    def update_checkpoint_path(self, path: Path) -> None:
        """Callback for when a checkpoint is selected."""
        self.checkpoint_path = path

    def update_scenario_name(self, name: str) -> None:
        """Callback for when a scenario is selected."""
        self.scenario_name = name

    # --- NEW: Worker Management ---
    def run_inference_worker(self) -> None:
        """Starts the background worker for inference."""
        log = self.query_one("#inference_log", Log)
        log.clear()
        log.write_line("Starting inference...")
        self.is_running_inference = True
        self.worker = self.run_inference()  # Start the worker

    def stop_inference_worker(self) -> None:
        """Stops the background worker."""
        if self.worker:
            log = self.query_one("#inference_log", Log)
            log.write_line("Stopping inference worker...")
            self.worker.cancel()

    @work(exclusive=True, thread=True, group="inference")
    def run_inference(self) -> None:
        """
        This is the background worker method.
        It simulates running inference episodes in a loop.
        Replace the `time.sleep` with your actual agent/environment loop.
        """
        worker = get_current_worker()
        status_label = self.query_one("#inference_status_label", Static)

        # Safely get state from the main thread
        checkpoint = self.checkpoint_path
        scenario = self.scenario_name

        self.post_message(InferenceLog(f"Worker started with checkpoint: {checkpoint.name}"))
        self.post_message(InferenceLog(f"Running scenario: {scenario}"))

        # This is where your agent and environment would be initialized.
        # e.g., agent = YourAgent.from_checkpoint(checkpoint)
        # e.g., env = YourRocketLeagueEnv(scenario_config=...)
        components = load_components_from_checkpoint(checkpoint)
        scenario_config = get_scenarios()[scenario]
        env = RLEnv(instantiate(scenario_config.env_config))
        rlviser_py.launch()

        for episode_num in itertools.count(1):
            if worker.is_cancelled:
                break

            self.call_from_thread(status_label.update, f"Status: Running Episode {episode_num}")
            self.post_message(InferenceLog(f"--- Episode {episode_num} Started ---"))

            run_episode(env, *components)

            if not worker.is_cancelled:
                self.post_message(InferenceLog(f"--- Episode {episode_num} Finished ---\n"))

        # Worker cleanup
        rlviser_py.quit()
        self.is_running_inference = False
        self.post_message(InferenceLog("Worker has been cancelled."))
        self.call_from_thread(status_label.update, "Status: Idle")

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Called when the worker state changes."""
        if event.state == "CANCELLED" or event.state == "SUCCESS":
            self.is_running_inference = False
            self.worker = None  # Clear the worker reference

    def on_inference_log(self, message: InferenceLog) -> None:
        """Receives log messages from the worker and writes them to the Log widget."""
        self.query_one("#inference_log", Log).write_line(message.line)

    # --- End of Worker Management ---


if __name__ == "__main__":
    app = InferenceApp()
    app.run()
