#!/usr/bin/env python3
"""Smoke test that instantiates UniGraspTransformerEnv and steps the sim."""

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
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Simulation steps to run (-1 keeps the sim running until interrupted).",
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

    # Local imports after the app is created.
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import UnigraspTransformerGraspEnv
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_env import UniGraspTransformerEnv

    # instantiate grasp env
    grasp_cfg = UnigraspTransformerGraspEnv()
    grasp_cfg.scene.num_envs = args.num_envs

    # instantiate unigrasptransformer env
    env = UniGraspTransformerEnv(cfg=grasp_cfg, headless=args.headless)
    scene_cfg = env.scene.cfg
    scene = env.scene

    # Visualize the freshly built scene (render only when not headless).
    if not args.headless:
        env.sim.render()
    
    # for reports
    robot = scene["robot"]
    obj = scene["object"]
    tbl = getattr(scene, "table", None)

    print("[INFO] Scene constructed.")
    # print(f"  - robot prim: {scene_cfg.robot.prim_path}")
    # if tbl is not None:
    #     print(f"  - table prim: {scene_cfg.table.prim_path}")
    # print(f"  - object prim: {scene_cfg.object.prim_path}")
    # print(f"  - num_envs: {scene_cfg.num_envs}")
    # print(f"  - robot DOF count: {robot.num_dof}")
    # print(f"  - object exists: {obj is not None}")


    # implement controllers
    # try:
    #     actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    #     step_i = 0
    #     run_forever = args.steps < 0
    #     while run_forever or step_i < args.steps:
    #         env.step(actions)
    #         step_i += 1
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     if hasattr(env, "close"):
    #         env.close()
    #     simulation_app.close()


if __name__ == "__main__":
    main()
