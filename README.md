# DenBot-RL

A Reinforcement Learning project for Rocket League.

-----

## 🚀 Setup

This project uses a `Makefile` to simplify the setup process.

### 1\. Install Dependencies

First, ensure you have initialized the git submodule for the project's dependency:

```bash
git submodule update --init --recursive
```

Next, run the main setup command:

```bash
make
```

This command automatically handles the `rlviser` dependency for you:

  * 💻 **On Linux (x86\_64)**: It downloads a pre-built `rlviser` binary, so no other tools are needed.
  * 🍎 **On other systems (like macOS or Windows)**: It builds `rlviser` from the source code in the submodule.

> **Important**: If you are **not** on a Linux x86\_64 system, you must have **[Rust and Cargo](https://www.rust-lang.org/tools/install)** installed for the build process to succeed.

### 2\. Initialize Python Environment

Once `make` is complete, set up and activate the Python environment using `uv`:

```bash
uv sync
source .venv/bin/activate
```

-----

## ⚡️ Run

### Start Training

You can start a training run with a specific experiment configuration:

```bash
python train.py exp=air_dribble
```

### Watch Agent with RLViser

To visualize a trained agent, you'll first need a saved checkpoint from a training run. Then, run:

```bash
python load_latest.py
```

-----

## 🧼 Clean Up

To remove the `rlviser` executable and all build artifacts created during the setup, run:

```bash
make clean
```
