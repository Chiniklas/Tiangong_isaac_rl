#!/usr/bin/env python3
"""Preview a configured GraspXL scene and visualise palm/object diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path


def _ensure_isaaclab_on_path():
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


_ensure_isaaclab_on_path()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--steps", type=int, default=480, help="Simulation steps to run.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to spawn.")
    parser.add_argument("--config-yaml", type=Path, default=None, help="Spawn configuration YAML.")
    parser.add_argument(
        "--max-affordance-points",
        type=int,
        default=3000,
        help="Maximum number of affordance surface points to render.",
    )
    return parser.parse_args()


def _setup_debug_prims():
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/Debug")

    def _mk_sphere(path: str, radius: float, color):
        xform = UsdGeom.Xform.Define(stage, f"/World/Debug/{path}")
        sphere = UsdGeom.Sphere.Define(stage, f"/World/Debug/{path}/Geom")
        sphere.CreateRadiusAttr(radius)
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        return UsdGeom.XformCommonAPI(xform)

    palm_api = _mk_sphere("PalmCenter", 0.015, (0.1, 0.6, 1.0))
    obj_api = _mk_sphere("ObjectCenter", 0.015, (1.0, 0.55, 0.0))

    def _mk_line(path: str, color):
        curve = UsdGeom.BasisCurves.Define(stage, f"/World/Debug/{path}")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.1)])
        curve.CreateWidthsAttr([0.005])
        curve.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        return curve

    palm_heading = _mk_line("PalmHeading", (0.2, 0.9, 1.0))
    obj_heading = _mk_line("ObjectHeading", (1.0, 0.8, 0.2))

    affordance_points = UsdGeom.Points.Define(stage, "/World/Debug/AffordancePoints")
    affordance_points.CreateWidthsAttr([0.004])
    affordance_points.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.9, 0.2)])

    return palm_api, obj_api, palm_heading, obj_heading, affordance_points


def main():
    args = parse_args()

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Could not import isaaclab.app.AppLauncher. Run inside the Isaac Lab kit shell."
        ) from exc

    app = AppLauncher(headless=args.headless)
    simulation_app = app.app

    import torch
    from isaaclab.utils.math import quat_apply
    from legged_lab.envs.graspxl_rl.graspxl_env import GraspXLEnv
    from legged_lab.envs.graspxl_rl.graspxl_cfg import GraspXLEnvCfg, GraspXLGraspSceneCfg
    from legged_lab.envs.graspxl_rl.spawn_cfg import GraspXLSpawnCfg
    from legged_lab.envs.graspxl_rl.grasp_helpers import apply_palm_motion
    from keyboard_controller import KeyboardController

    if args.config_yaml is not None:
        spawn_cfg = GraspXLSpawnCfg(config_path=args.config_yaml.expanduser().resolve().as_posix())
    else:
        spawn_cfg = GraspXLSpawnCfg()

    scene_cfg = GraspXLGraspSceneCfg(spawn=spawn_cfg, num_envs=args.num_envs)
    env_cfg = GraspXLEnvCfg(scene=scene_cfg)

    env = GraspXLEnv(env_cfg, headless=args.headless)

    palm_api, obj_api, palm_heading_curve, obj_heading_curve, afford_pts_prim = _setup_debug_prims()

    controller = None
    if not args.headless:
        controller = KeyboardController()

    grid = getattr(env, "_aff_sdf_grid", None)
    if grid is not None:
        idx = (grid <= 0.001).nonzero(as_tuple=False)
        if idx.shape[0] > args.max_affordance_points:
            idx = idx[torch.randperm(idx.shape[0])[: args.max_affordance_points]]
        mins = env._aff_sdf_min
        maxs = env._aff_sdf_max
        res = torch.tensor(grid.shape, device=grid.device, dtype=torch.float32)
        norm = idx.float() / torch.clamp(res - 1.0, min=1.0)
        local_pts = mins + norm * (maxs - mins)
        obj_pos = env.obj.data.root_pos_w[0]
        obj_quat = env.obj.data.root_quat_w[0]
        world_pts = quat_apply(obj_quat, local_pts)
        world_pts = (world_pts + obj_pos.unsqueeze(0)).detach().cpu().numpy()
        from pxr import Gf

        afford_pts_prim.GetPointsAttr().Set([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in world_pts])
        afford_pts_prim.GetWidthsAttr().Set([0.003] * len(world_pts))

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    palm_trans = torch.zeros(env.num_envs, 3, device=env.device)
    palm_rot = torch.zeros_like(palm_trans)

    try:
        while True:
            if controller is not None:
                dx, dy, dz = controller.translation
            else:
                dx = dy = dz = 0.0
            palm_trans[:, 0] = dx
            palm_trans[:, 1] = dy
            palm_trans[:, 2] = dz
            apply_palm_motion(env, palm_trans, palm_rot)

            env.step(actions)

            palm_pos = env.hand.data.root_pos_w[0].detach().cpu().numpy()
            obj_pos = env.obj.data.root_pos_w[0].detach().cpu().numpy()

            palm_api.SetTranslate(palm_pos.tolist())
            obj_api.SetTranslate(obj_pos.tolist())

            palm_dir = quat_apply(env.hand.data.root_quat_w[0], torch.tensor([0.0, 0.0, 1.0], device=env.device)).detach().cpu().numpy()
            obj_dir = quat_apply(env.obj.data.root_quat_w[0], torch.tensor([0.0, 0.0, 1.0], device=env.device)).detach().cpu().numpy()

            palm_line = [palm_pos, palm_pos + 0.12 * palm_dir]
            obj_line = [obj_pos, obj_pos + 0.12 * obj_dir]
            from pxr import Gf

            palm_heading_curve.GetPointsAttr().Set(
                [Gf.Vec3f(float(palm_line[0][0]), float(palm_line[0][1]), float(palm_line[0][2])),
                 Gf.Vec3f(float(palm_line[1][0]), float(palm_line[1][1]), float(palm_line[1][2]))]
            )
            obj_heading_curve.GetPointsAttr().Set(
                [Gf.Vec3f(float(obj_line[0][0]), float(obj_line[0][1]), float(obj_line[0][2])),
                 Gf.Vec3f(float(obj_line[1][0]), float(obj_line[1][1]), float(obj_line[1][2]))]
            )
    except KeyboardInterrupt:
        print("\n[INFO] Visualization interrupted by user. Shutting down...")
    finally:
        if controller is not None:
            controller.shutdown()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
