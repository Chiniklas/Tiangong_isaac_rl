# Tiangong Isaac RL

Forked from the TienKung-Lab locomotion template and extended for Inspire Hand dexterous manipulation.

## Table of Contents
- [Hardware Setup](#hardware-setup)
- [Upstream Locomotion Workflow](#upstream-locomotion-workflow)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Visualize Motion](#visualize-motion)
    - [Visualize Motion with Sensors](#visualize-motion-with-sensors)
    - [Train](#train)
    - [Play](#play)
    - [Sim2Sim (MuJoCo)](#sim2sim-mujoco)
    - [TensorBoard](#tensorboard)
  - [Code Formatting](#code-formatting)
  - [Troubleshooting](#troubleshooting)
  - [References](#references)
- [Inspire Hand Dexterous Manipulation](#inspire-hand-dexterous-manipulation)
  - [Environment Setup](#environment-setup)
  - [Train Agents](#train-agents)
  - [Reward Shaping](#reward-shaping)
  - [Policy Inference and Visualization](#policy-inference-and-visualization)
  - [Sim2Real](#sim2real)
  - [Data Recording](#data-recording)
  - [Policy Distillation](#policy-distillation)
- [Project TODOs](#project-todos)

## Hardware Setup
- Tiangong2 pro version humanoid
- two inspire-hands

## Upstream Locomotion Workflow
### Installation
TienKung-Lab is built with IsaacSim 4.5.0 and IsaacLab 2.1.0. **However**, since the tutorial for IsaacSim 4.5.0 has been depricated, our Tiangong_isaac_rl repo is built upon IsaacSim 5.0.0 AND Isaaclab 2.1.0. The installation is tested ok.

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory)

- Using a python interpreter that has Isaac Lab installed, install the library

```bash
cd TienKung-Lab
pip install -e .
```
- Install the rsl-rl library

```bash
cd TienKung-Lab/rsl_rl
pip install -e .
```

- Verify that the extension is correctly installed by running the following command:

```bash
python legged_lab/scripts/train.py --task=walk  --logger=tensorboard --headless --num_envs=64
```

### Usage

#### Visualize Motion

Visualize the motion by updating the simulation with data from tienkung/datasets/motion_visualization.

```bash
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1
python legged_lab/scripts/play_amp_animation.py --task=run --num_envs=1
```

#### Visualize Motion with Sensors

Visualize the motion with sensors by updating the simulation with data from tienkung/datasets/motion_visualization.

```bash
python legged_lab/scripts/play_amp_animation.py --task=walk_with_sensor --num_envs=1
python legged_lab/scripts/play_amp_animation.py --task=run_with_sensor --num_envs=1
```

#### Train

Train the policy using AMP expert data from tienkung/datasets/motion_amp_expert.

```bash
python legged_lab/scripts/train.py --task=walk --headless --logger=tensorboard --num_envs=4096
python legged_lab/scripts/train.py --task=run --headless --logger=tensorboard --num_envs=4096
```

#### Play

Run the trained policy.

```bash
python legged_lab/scripts/play.py --task=walk --num_envs=1
python legged_lab/scripts/play.py --task=run --num_envs=1
```

#### Sim2Sim (MuJoCo)

Evaluate the trained policy in MuJoCo to perform cross-simulation validation.

Exported_policy/ contains pretrained policies provided by the project. When using the play script, trained policy is exported automatically and saved to path like logs/run/[timestamp]/exported/policy.pt.
```bash
python legged_lab/scripts/sim2sim.py --task walk --policy Exported_policy/walk.pt --duration 10
python legged_lab/scripts/sim2sim.py --task run --policy Exported_policy/run.pt --duration 10
```

#### TensorBoard
```bash
tensorboard --logdir=logs/walk
tensorboard --logdir=logs/run
```

### Code Formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

### Troubleshooting

#### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/legged_lab",
        "<path-to-IsaacLab>/source/isaaclab_tasks",
        "<path-to-IsaacLab>/source/isaaclab_mimic",
        "<path-to-IsaacLab>/source/extensions",
        "<path-to-IsaacLab>/source/isaaclab_assets",
        "<path-to-IsaacLab>/source/isaaclab_rl",
        "<path-to-IsaacLab>/source/isaaclab",
    ]
}
```

### References

TODO

## Inspire Hand Dexterous Manipulation
### Environment Setup 
- **Step 1 – Point at the asset:** Verify the Inspire Hand USD path in `legged_lab/assets/handright9253/inspirehand.py`. If you cloned the repo elsewhere, update `INSPIRE_HAND_USD` so it points at your local USD.
- **Step 2 – Install locally:** Install the package in editable mode (`pip install -e .`) inside an Isaac Lab-enabled environment so `legged_lab` and `rsl_rl` are importable.
- **Step 3 – Warm up Isaac:** Boot Isaac Sim/Isaac Lab once to let it build caches, then register the environment by importing `legged_lab.envs` (this executes the `task_registry.register("inspirehand_grasp", ...)` call).
- **Step 4 – Smoke test:** Run `python -m legged_lab.scripts.inspire_hand.tests.test_env_registration` to confirm the `inspirehand_grasp` task builds successfully before launching training jobs.

### Train Agents

### Reward Shaping

### Policy Inference and Visualization

### Sim2Real

### Data Recording

### Policy Distillation


## Project TODOs
- [ ] transfer RL pipeline from Raisim to Isaacsim
- [ ] train a working RL agent based on one object and one grasp primitive
- [ ] add point cloud sensors for real time perception
- [ ] run RL agents on real hardware
