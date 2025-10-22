#!/usr/bin/env python3
"""Preview a configured GraspXL scene with the Inspire Hand.

The script launches the Isaac Lab simulator, constructs the scene using the
standard GraspXL configuration pipeline (spawn cfg → env cfg → env), and steps
the simulation for a short window so you can verify the selected object setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _ensure_isaaclab_on_path():
    """Ensure the local Isaac Lab source tree is importable."""

    import os
    import sys

    if "isaaclab" in sys.modules:
        return

    source_hint = os.environ.get("ISAACLAB_SOURCE")
    candidates = []
    if source_hint:
        candidates.append(Path(source_hint))
        candidates.append(Path(source_hint) / "isaaclab")
    home_root = Path.home() / "IsaacLab" / "source"
    candidates.append(home_root)
    candidates.append(home_root / "isaaclab")

    for path in candidates:
        if path.exists() and path.as_posix() not in sys.path:
            sys.path.append(path.as_posix())

    try:
        import toml  # noqa: F401
    except ModuleNotFoundError:
        try:
            import tomllib as toml  # type: ignore  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "Missing 'toml' dependency required by Isaac Lab. Install it via 'pip install toml'."
            ) from exc


_ensure_isaaclab_on_path()

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="Simulation steps to run.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments to spawn.",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="Optional YAML file describing a fixed object spawn configuration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Could not import isaaclab.app.AppLauncher. Run this script inside the Isaac Lab kit shell (./isaaclab.sh --run)."
        ) from exc

    app = AppLauncher(headless=args.headless)
    simulation_app = app.app

    # The following imports rely on Omniverse plugins, so defer until
    # after the AppLauncher has brought up the Kit app.
    import torch

    from legged_lab.envs.graspxl_rl.graspxl_env import GraspXLEnv
    from legged_lab.envs.graspxl_rl.graspxl_cfg import GraspXLEnvCfg, GraspXLGraspSceneCfg
    from legged_lab.envs.graspxl_rl.spawn_cfg import GraspXLSpawnCfg

    if args.config_yaml is not None:
        spawn_cfg = GraspXLSpawnCfg(config_path=args.config_yaml.expanduser().resolve().as_posix())
    else:
        spawn_cfg = GraspXLSpawnCfg()

    scene_cfg = GraspXLGraspSceneCfg(spawn=spawn_cfg, num_envs=args.num_envs)
    env_cfg = GraspXLEnvCfg(scene=scene_cfg)

    env = GraspXLEnv(env_cfg, headless=args.headless)

    spawned_name = getattr(env._current_object, "object_id", None)
    if spawned_name is None:
        spawned_name = spawn_cfg.grasp_object.object_id or "unknown"

    print(f"[INFO] Spawned GraspXL grasp object: {spawned_name}")

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    try:
        while True:
            env.step(actions)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down...")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
