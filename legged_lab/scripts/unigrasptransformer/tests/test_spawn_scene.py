#!/usr/bin/env python3
"""Spawn a UniGraspTransformer scene and print a concise status summary.

All hyper-parameters come from legged_lab/envs/unigrasptransformer/cfg/spawn_cfg.yaml
via the UnigraspTransformerGraspEnv config. If configuration is inconsistent
(e.g., mesh requested but no USD path), raises an error. No overlays or dataset
picking here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


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
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument(
        "--steps",
        type=int,
        default=-1,
        help="Simulation steps to run (-1 keeps the sim running until interrupted).",
    )
    parser.add_argument(
        "--hand-usd",
        type=Path,
        default=None,
        help="Optional USD asset to use for the hand instead of the default from spawn_cfg.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Could not import isaaclab.app.AppLauncher. Run inside the Isaac Lab kit shell (./isaaclab.sh --run)."
        ) from exc

    app = AppLauncher(headless=args.headless)
    simulation_app = app.app

    from legged_lab.envs.unigrasptransformer.unigrasptransformer_env import UniGraspTransformerEnv
    from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import UnigraspTransformerGraspEnv

    # Load config (spawn cfg auto-loads spawn_cfg.yaml in dex_grasp_cfg)
    grasp_cfg = UnigraspTransformerGraspEnv()
    # Optional override for hand USD
    if args.hand_usd:
        if hasattr(grasp_cfg.scene, "hand_spawn"):
            grasp_cfg.scene.hand_spawn["asset_path"] = Path(args.hand_usd).expanduser().as_posix()
        else:
            print("[WARN] hand_spawn not available; hand USD override skipped")

    # Validate object config if mesh is requested and random pick failed upstream
    obj_spawn = grasp_cfg.scene.object_spawn
    if obj_spawn.get("enable", False) and obj_spawn.get("spawn_mesh", False) and not obj_spawn.get("object_path"):
        raise RuntimeError(
            "Config error: object.spawn_mesh=true but object.object_path is not set to a USD file or object directory (and no dataset object was auto-picked)"
        )

    # Build env directly (env will build InteractiveScene internally)
    env = UniGraspTransformerEnv(grasp_cfg, headless=args.headless)

    # Print concise status
    tbl = getattr(grasp_cfg.scene, "table", None) is not None or obj_spawn.get("enable", False)
    obj_status = "disabled"
    obj_spawned = obj_spawn.get("enable", False)
    if obj_spawn.get("enable", False):
        if obj_spawn.get("spawn_mesh", False):
            obj_status = f"USD: {obj_spawn.get('object_path') or '<missing>'}"
        else:
            obj_status = f"cuboid size={obj_spawn.get('size')}"

    hand_spawn = getattr(grasp_cfg.scene, "hand_spawn", {})
    print("[INFO] Scene Summary:")
    print(f"  - table: {'enabled' if tbl else 'disabled'}")
    print(f"  - object: {obj_status} (spawned={'yes' if obj_spawned else 'no'})")
    print(f"  - hand: pos={hand_spawn.get('pos')}, rot_xyzw={hand_spawn.get('rot_xyzw')}, asset={hand_spawn.get('asset_path')}")

    # Step a few frames to display scene
    try:
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        step_i = 0
        run_forever = args.steps < 0
        while run_forever or step_i < args.steps:
            env.step(actions)
            step_i += 1
    except KeyboardInterrupt:
        pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
