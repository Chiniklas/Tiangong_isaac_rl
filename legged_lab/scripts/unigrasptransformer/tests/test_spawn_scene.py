#!/usr/bin/env python3
"""Spawn a UniGraspTransformer scene and print a concise status summary.

All hyper-parameters come from legged_lab/envs/unigrasptransformer/config.yaml
via the SpawnCfg. If configuration is inconsistent (e.g., mesh requested but
no USD path), raises an error. No overlays or dataset picking here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

CONTACT_BODY_KEYWORDS = ("ffdistal", "mfdistal", "rfdistal", "lfdistal", "thdistal")

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


class ContactForceViewer:
    """omni.ui panel that streams selected contact magnitudes."""

    def __init__(self, env, env_index: int = 0, body_keywords: Sequence[str] = CONTACT_BODY_KEYWORDS):
        import omni.ui as ui

        self._ui = ui
        self._env = env
        self._env_index = env_index
        self._sensor = env.contact_sensor
        self._body_indices, self._body_labels = self._resolve_bodies(body_keywords)
        if not self._body_indices:
            raise RuntimeError("ContactForceViewer could not resolve any bodies to track.")

        self._window = ui.Window(
            title="Contact Forces",
            width=320,
            height=0,
            visible=True,
            dockPreference=ui.DockPreference.RIGHT_BOTTOM,
        )
        self._value_models = []
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                ui.Label("Per-link contact force (N)", alignment=ui.Alignment.CENTER)
                for label in self._body_labels:
                    with ui.HStack(spacing=10):
                        ui.Label(label, width=140)
                        drag = ui.FloatDrag(min=0.0, max=200.0, step=0.01)
                        drag.enabled = False
                        self._value_models.append(drag.model)

    def _resolve_bodies(self, keywords: Sequence[str]) -> tuple[list[int], list[str]]:
        names = self._sensor.body_names
        indices: list[int] = []
        labels: list[str] = []
        lowered = [k.lower() for k in keywords]
        for idx, name in enumerate(names):
            lname = name.lower()
            if any(key in lname for key in lowered):
                indices.append(idx)
                labels.append(name)
        if not indices and names:
            limit = min(6, len(names))
            indices = list(range(limit))
            labels = names[:limit]
        return indices, labels

    def update(self):
        data = self._sensor.data.net_forces_w
        if data is None or data.shape[0] <= self._env_index:
            return
        forces = data[self._env_index]
        if forces is None:
            return
        magnitudes = torch.linalg.norm(forces[self._body_indices].to("cpu"), dim=-1).tolist()
        for model, value in zip(self._value_models, magnitudes):
            model.set_value(float(value))


def _maybe_create_contact_viewer(env, headless: bool):
    if headless:
        return None
    try:
        return ContactForceViewer(env)
    except Exception as exc:
        print(f"[WARN] Contact viewer unavailable: {exc}")
        return None


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
    if spawn_cfg.grasp_object.enable and spawn_cfg.grasp_object.spawn_mesh and not spawn_cfg.grasp_object.object_path:
        raise RuntimeError("Config error: object.spawn_mesh=true but object.object_path is not set to a USD file or object directory (and no dataset object was auto-picked)")

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
        obj_status = f"USD: {obj_cfg.object_path or '<missing>'}"
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
    if obj_cfg.enable and obj_cfg.spawn_mesh and not obj_cfg.object_path:
        raise RuntimeError("Scene built with object.spawn_mesh=true but no object.object_path configured")

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

    contact_viewer = _maybe_create_contact_viewer(env, args.headless)

    # Step a few frames to display scene/overlays
    try:
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        step_i = 0
        run_forever = args.steps < 0
        while run_forever or step_i < args.steps:
            env.step(actions)
            if contact_viewer:
                contact_viewer.update()
            step_i += 1
    except KeyboardInterrupt:
        pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
