import glob
import json
import os
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import CenterMiddle, HorizontalGroup
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, DirectoryTree, Footer, Header, Label, Tree
from textual.widgets.tree import TreeNode


def get_most_recent_checkpoint() -> Path:
    dir_str = "ray_results/**/checkpoint_*/"
    all_checkpoints = glob.glob(dir_str, recursive=True)
    most_recent_checkpoint = max(all_checkpoints, key=os.path.getmtime)
    print(f"Found: {most_recent_checkpoint}")

    return Path(most_recent_checkpoint).absolute()


class Checkpoint(HorizontalGroup):
    checkpoint = reactive(str)

    def compose(self) -> ComposeResult:
        yield Button("Get Latest", id="latest")
        yield Button("Select", id="select")
        yield Label(f"{get_most_recent_checkpoint()}")


class CheckpointSelector(DirectoryTree):
    class CheckpointSelected(Message):
        def __init__(self, path: Path) -> None:
            super().__init__()
            self.path: Path = path

    def _on_tree_node_selected(self, event: DirectoryTree.NodeSelected) -> None:
        event.stop()
        selected_path = event.node.data.path
        if selected_path.is_dir() and selected_path.name.startswith("checkpoint_"):
            self.post_message(self.CheckpointSelected(selected_path))


class InferenceApp(App):
    BINDINGS = [
        ("a", "add", "Add node"),
        ("c", "clear", "Clear"),
        ("t", "toggle_root", "Toggle root"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()
        yield CenterMiddle(
            Checkpoint(),
        )
        # yield CheckpointSelector("ray_results")

    @classmethod
    def add_json(cls, node: TreeNode, json_data: object) -> None:
        """Adds JSON data to a node.

        Args:
            node (TreeNode): A Tree node.
            json_data (object): An object decoded from JSON.
        """

        from rich.highlighter import ReprHighlighter

        highlighter = ReprHighlighter()

        def add_node(name: str, node: TreeNode, data: object) -> None:
            """Adds a node to the tree.

            Args:
                name (str): Name of the node.
                node (TreeNode): Parent node.
                data (object): Data associated with the node.
            """
            if isinstance(data, dict):
                node.set_label(Text(f"{{}} {name}"))
                for key, value in data.items():
                    new_node = node.add("")
                    add_node(key, new_node, value)
            elif isinstance(data, list):
                node.set_label(Text(f"[] {name}"))
                for index, value in enumerate(data):
                    new_node = node.add("")
                    add_node(str(index), new_node, value)
            else:
                node.allow_expand = False
                if name:
                    label = Text.assemble(Text.from_markup(f"[b]{name}[/b]="), highlighter(repr(data)))
                else:
                    label = Text(repr(data))
                node.set_label(label)

        add_node("JSON", node, json_data)

    def on_mount(self) -> None:
        """Load some JSON when the app starts."""
        self.title = "DenBot Inference Tool"

        most_recent_checkpoint = get_most_recent_checkpoint()
        with open(most_recent_checkpoint / "../params.json") as data_file:
            self.json_data = json.load(data_file)

    def action_add(self) -> None:
        """Add a node to the tree."""
        tree = self.query_one(Tree)
        json_node = tree.root.add("JSON")
        self.add_json(json_node, self.json_data)
        tree.root.expand()

    def action_clear(self) -> None:
        """Clear the tree (remove all nodes)."""
        tree = self.query_one(Tree)
        tree.clear()

    def action_toggle_root(self) -> None:
        """Toggle the root node."""
        tree = self.query_one(Tree)
        tree.show_root = not tree.show_root


if __name__ == "__main__":
    app = InferenceApp()
    app.run()
