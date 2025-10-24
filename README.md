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
  - [Inspire Hand Tasks and Sanity Check Scripts](#inspire-hand-tasks-and-sanity-check-scripts)
  - [Reward Shaping](#reward-shaping)
  - [Train Agents](#train-agents)
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

Our Inspire Hand stack follows the same layered architecture across GraspXL and UniGraspTransformer tasks:

1. **Spawn configuration (`spawn_cfg.py`)** collects table, grasp object, and Inspire Hand parameters. These configs optionally load YAML overrides that point to the converted USD meshes, affordance SDFs, and metadata generated by `legged_lab/scripts/tools/convert_dataset_to_usd.py` and `legged_lab/scripts/tools/generate_sdf.py`.
2. **Scene configuration (`*_scene_cfg.py`)** wraps the spawn cfg with Isaac Lab’s `SceneCfg`, instantiating rigid bodies, materials, and collision settings while attaching the SDF tensors to the environment.
3. **Environment configuration (`*_env_cfg.py`)** extends `GraspXLEnvCfg` to select the scene, reward scales, reset tolerances, and domain randomisation toggles. UniGraspTransformer reuses the same base but swaps the reward configuration and spawn bundle.
4. **Environment implementation (`*_env.py`)** subclasses `BaseEnv`, converts spawn metadata into runtime tensors, initialises fingertip caches, samples SDF grids, and exposes utilities like palm motion helpers.
5. **Agent configuration (`*_cfg.py` → `*AgentCfg`)** packages RSL-RL runner defaults (steps per env, logger, experiment names) for training scripts.

After installation (`pip install -e .` inside an Isaac Lab-enabled interpreter) run the tooling once to populate `dataset/grasp_usd/` and then import `legged_lab.envs` to register all tasks with the global registry.

### Inspire Hand Tasks and Sanity Check Scripts
- **Registered tasks:** `inspirehand_grasp`, `graspxl_rl`, `unigrasptransformer`.
- **Quick checks (`graspxl_rl`):**
  - `python3 legged_lab/scripts/graspxl_rl/tests/test_spawn_scene.py`
  - `python3 legged_lab/scripts/graspxl_rl/tests/test_visualize_objective_tracking.py`
- **Quick checks (`unigrasptransformer`):**
  - `python3 legged_lab/scripts/unigrasptransformer/tests/test_spawn_scene.py`
  - `python3 legged_lab/scripts/unigrasptransformer/tests/test_visualize_objective_tracking.py`
- Shared utilities (keyboard controller, mesh conversion, SDF generation) live under `legged_lab/scripts/tools/`.

### Reward Shaping

- **GraspXL (default):** combines exponential reach success, discrete lift/hold bonuses, velocity penalties on palm and object drift, and affordance SDF shaping. See `legged_lab/mdp/rewards_graspxl.py` for raw terms and `legged_lab/envs/graspxl_rl/grasp_helpers.py` for aggregation.
- **UniGraspTransformer:** mirrors the reference project’s two-phase reward—aligning the hand with a PCA-derived pose, rewarding exploratory contacts, then transitioning to grasp/lift maintenance with action penalties. Implemented in `legged_lab/mdp/rewards_unigrasptransformer.py` and consumed via `legged_lab/envs/unigrasptransformer/grasp_helpers.py`.
- **Our extensions:** _placeholder for project-specific shaping experiments._

### Train Agents

- Launch headless training with the standard RSL-RL entrypoints, for example:

  ```bash
  python3 legged_lab/scripts/train.py --task=inspirehand_grasp --headless --num_envs=512 --logger=tensorboard
  ```

- Monitor metrics (`tensorboard --logdir=logs/<task_name>`) and track reward components such as `reward/reach`, `reward/lift`, `reward/hold`, and task-specific diagnostics (`grasp/aff_sdf_mean`, `reward/init`, `reward/grasp`).

### Policy Inference and Visualization

### Sim2Real

### Data Recording

### Policy Distillation


## Project TODOs
- [ ] transfer RL pipeline from Raisim to Isaacsim
- [ ] train a working RL agent based on one object and one grasp primitive
- [ ] add point cloud sensors for real time perception
- [ ] run RL agents on real hardware
