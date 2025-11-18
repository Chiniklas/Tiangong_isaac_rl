#!/usr/bin/env python3
"""Spawn a UniGraspTransformer scene and print a concise status summary.

All hyper-parameters come from legged_lab/envs/unigrasptransformer/config.yaml
via the SpawnCfg. If configuration is inconsistent (e.g., mesh requested but
no USD path), raises an error. No overlays or dataset picking here.
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
        help="Optional USD asset to use for the hand instead of the default URDF.",
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
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_cfg import (
        UniGraspTransformerEnvCfg,
        UniGraspTransformerGraspSceneCfg,
    )
    from legged_lab.envs.unigrasptransformer.spawn_cfg import UniGraspTransformerSpawnCfg

    # Load config (spawn cfg auto-loads config.yaml in __post_init__)
    spawn_cfg = UniGraspTransformerSpawnCfg()
    if args.hand_usd:
        spawn_cfg.hand.asset_path = Path(args.hand_usd).expanduser().as_posix()

    # Validate object config if mesh is requested and random pick failed upstream
    if spawn_cfg.grasp_object.enable and spawn_cfg.grasp_object.spawn_mesh and not spawn_cfg.grasp_object.static_usd:
        raise RuntimeError("Config error: object.spawn_mesh=true but object.static_usd is not set in config.yaml (and no dataset object was auto-picked)")

    # Build scene and spawn env
    scene_cfg = UniGraspTransformerGraspSceneCfg(spawn=spawn_cfg, num_envs=args.num_envs)
    env_cfg = UniGraspTransformerEnvCfg(scene=scene_cfg)
    env = UniGraspTransformerEnv(env_cfg, headless=args.headless)

    # Print concise status
    tbl = scene_cfg.table is not None
    obj_cfg = spawn_cfg.grasp_object
    obj_spawned = scene_cfg.grasp_object is not None
    if not obj_cfg.enable:
        obj_status = "disabled"
    elif obj_cfg.spawn_mesh:
        obj_status = f"USD: {obj_cfg.static_usd or '<missing>'}"
    else:
        obj_status = f"cuboid size={obj_cfg.size}"

    print("[INFO] Scene Summary:")
    print(f"  - table: {'enabled' if tbl else 'disabled'}")
    print(f"  - object: {obj_status} (spawned={'yes' if obj_spawned else 'no'})")
    print(f"  - hand: pos={spawn_cfg.hand.pos}, rot_xyzw={spawn_cfg.hand.orientation_xyzw}, asset={spawn_cfg.hand.asset_type}")
    # Echo object overlay/mesh flags from YAML so users can verify they were loaded
    print("[INFO] Object Flags (from YAML):")
    print(f"  - enable={obj_cfg.enable}, spawn_mesh={obj_cfg.spawn_mesh}")
    print(f"  - show_point_cloud={getattr(obj_cfg, 'show_point_cloud', None)}")
    print(f"  - show_pca_axes={getattr(obj_cfg, 'show_pca_axes', None)}")
    if getattr(obj_cfg, 'object_id', None):
        print(f"  - object_id={obj_cfg.object_id}")

    # Post-validate object state
    if obj_cfg.enable and obj_cfg.spawn_mesh and not obj_cfg.static_usd:
        raise RuntimeError("Scene built with object.spawn_mesh=true but no object.static_usd configured")

    # Create simple overlays if enabled in YAML and data is available
    try:
        if obj_cfg.enable:
            import numpy as _np
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            # Point cloud
            if getattr(obj_cfg, "show_point_cloud", False) and getattr(obj_cfg, "pc_fps", None):
                try:
                    pc = _np.load(obj_cfg.pc_fps).astype(_np.float32)
                    for i in range(env.num_envs):
                        debug_root = f"/World/envs/env_{i}/Object/Debug"
                        UsdGeom.Xform.Define(stage, debug_root)
                        pc_prim = UsdGeom.Points.Define(stage, f"{debug_root}/ObjectPC")
                        pc_prim.CreateWidthsAttr([0.01])
                        pc_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.85, 0.95)])
                        pc_prim.GetPointsAttr().Set([
                            Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pc
                        ])
                except Exception as _e:
                    print(f"[WARN] Failed to create point cloud overlay: {_e}")

            # PCA axes
            if getattr(obj_cfg, "show_pca_axes", False) and getattr(obj_cfg, "pca_axes", None):
                try:
                    axes = _np.load(obj_cfg.pca_axes).astype(_np.float32)
                    colors = [(1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0)]
                    scale = 0.2
                    for i in range(env.num_envs):
                        debug_root = f"/World/envs/env_{i}/Object/Debug"
                        for a in range(3):
                            curve = UsdGeom.BasisCurves.Define(stage, f"{debug_root}/PCA_Axis_{a}")
                            curve.CreateTypeAttr("linear")
                            curve.CreateCurveVertexCountsAttr([2])
                            curve.CreateWidthsAttr([0.02])
                            curve.GetDisplayColorAttr().Set([Gf.Vec3f(*colors[a])])
                            a0 = (0.0, 0.0, 0.0)
                            a1 = (
                                float(scale * axes[a, 0]),
                                float(scale * axes[a, 1]),
                                float(scale * axes[a, 2]),
                            )
                            curve.GetPointsAttr().Set([Gf.Vec3f(*a0), Gf.Vec3f(*a1)])
                except Exception as _e:
                    print(f"[WARN] Failed to create PCA axes overlay: {_e}")
    except Exception as _e:
        print(f"[WARN] Overlay init skipped: {_e}")

    # Step a few frames to display scene/overlays
    try:
        import torch
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
