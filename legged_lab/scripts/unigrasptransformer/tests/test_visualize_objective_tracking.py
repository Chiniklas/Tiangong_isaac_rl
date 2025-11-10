#!/usr/bin/env python3
"""Preview a configured UniGraspTransformer scene and visualise palm/object diagnostics."""

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
    parser.add_argument(
        "--palm-axis",
        type=str,
        default="-x",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        help="Local axis to visualize as the palm direction (applied by hand root quat).",
    )
    parser.add_argument(
        "--ugtf-root",
        type=Path,
        default=Path("/home/chizhang/projects/Tiangong_isaac_rl/dataset/unigrasptransformer_asset/meshdatav3_scaled"),
        help="Root of UniDexGrasp++ meshes (meshdatav3_scaled).",
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default="subset_core10",
        help="Subset directory name created by the conversion tool (under --ugtf-root)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="core",
        help="Category inside the subset to sample from (e.g., core/sem/ddg/mujoco)",
    )
    parser.add_argument(
        "--ugtf-object",
        type=str,
        default=None,
        help="Optional object relative path within the subset (e.g., 'core/bottle-xxxx').",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the state GUI to display obs/actions/rewards live.",
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
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_env import UniGraspTransformerEnv
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_cfg import (
        UniGraspTransformerEnvCfg,
        UniGraspTransformerGraspSceneCfg,
    )
    from legged_lab.envs.unigrasptransformer.spawn_cfg import UniGraspTransformerSpawnCfg
    from legged_lab.assets.inspirehand.object_library import GraspObjectInfo
    from legged_lab.envs.unigrasptransformer.grasp_helpers import apply_palm_motion
    from legged_lab.scripts.tools.keyboard_controller import KeyboardController
    from legged_lab.scripts.tools.state_gui import launch_state_gui

    spawn_cfg = UniGraspTransformerSpawnCfg(
        config_path=(args.config_yaml.expanduser().resolve().as_posix() if args.config_yaml else None)
    )

    # If a UniGraspTransformer subset is present, randomly pick an object and load its pre-converted USD via metadata.json
    assets_root = args.ugtf_root.expanduser().resolve()
    subset_dir = assets_root / args.subset_name / args.category
    if subset_dir.exists() and subset_dir.is_dir():
        import json as _json
        import random as _random

        if args.ugtf_object is not None:
            chosen_dir = (assets_root / args.ugtf_object).resolve()
            if not chosen_dir.exists():
                raise FileNotFoundError(f"UGTF subset object dir not found: {chosen_dir}")
        else:
            obj_dirs = [p for p in subset_dir.iterdir() if p.is_dir()]
            if not obj_dirs:
                raise RuntimeError(f"No objects under subset: {subset_dir}")
            chosen_dir = _random.choice(obj_dirs)

        meta_path = chosen_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing metadata.json next to subset object: {meta_path}. Re-run the converter without --usd-output."
            )
        metadata = _json.loads(meta_path.read_text())
        static_usd = metadata.get("static_usd")
        if not static_usd:
            raise RuntimeError(f"metadata.json for {chosen_dir.name} lacks 'static_usd'")
        static_usd_path = Path(static_usd).expanduser().resolve()
        if not static_usd_path.exists():
            raise FileNotFoundError(f"static_usd path does not exist: {static_usd_path}")

        # Override spawn to use the pre-converted USD (no SDFs)
        spawn_cfg.use_object_library = False
        spawn_cfg.grasp_object.static_usd = static_usd_path.as_posix()
        spawn_cfg.grasp_object.affordance_sdf = None
        spawn_cfg.grasp_object.non_affordance_sdf = None
        spawn_cfg.grasp_object.affordance_sdf_data = None
        spawn_cfg.grasp_object.non_affordance_sdf_data = None

        override = GraspObjectInfo(
            object_id=chosen_dir.name,
            category=chosen_dir.parent.name,
            root_dir=chosen_dir,
            urdf=None,
            fixed_base_urdf=None,
            affordance_mesh=None,
            non_affordance_mesh=None,
            lowest_point=None,
            static_usd=static_usd_path,
            affordance_usd=None,
            non_affordance_usd=None,
            affordance_sdf=None,
            non_affordance_sdf=None,
        )
        spawn_cfg._override_object_info = override

    scene_cfg = UniGraspTransformerGraspSceneCfg(spawn=spawn_cfg, num_envs=args.num_envs)
    env_cfg = UniGraspTransformerEnvCfg(scene=scene_cfg)
    # Disable episode time limit for long interactive visualization
    env_cfg.scene.max_episode_length_s = 1e9

    env = UniGraspTransformerEnv(env_cfg, headless=args.headless)

    palm_api, obj_api, palm_heading_curve, obj_heading_curve, afford_pts_prim = _setup_debug_prims()
    # Visualize PCA axes if available
    try:
        import numpy as _np
        from pxr import Gf
        pca_path = getattr(env.cfg.scene.spawn.grasp_object, "pca_axes", None)
        if pca_path:
            axes = _np.load(pca_path).astype(_np.float32)  # 3x3 rows
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            from pxr import UsdGeom
            UsdGeom.Xform.Define(stage, "/World/Debug/PCA")
            colors = [(1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.2, 0.2, 1.0)]
            pca_curves = []
            for i in range(3):
                curve = UsdGeom.BasisCurves.Define(stage, f"/World/Debug/PCA/Axis{i}")
                curve.CreateTypeAttr("linear")
                curve.CreateCurveVertexCountsAttr([2])
                curve.CreateWidthsAttr([0.006])
                curve.GetDisplayColorAttr().Set([Gf.Vec3f(*colors[i])])
                pca_curves.append(curve)
    except Exception:
        pca_curves = []

    controller = None
    if not args.headless:
        controller = KeyboardController()
    gui = launch_state_gui("UGTF Debug") if args.gui else None

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
        step_i = 0
        while True:
            if controller is not None:
                dx, dy, dz = controller.translation
                rx, ry, rz = controller.rotation
            else:
                dx = dy = dz = 0.0
                rx = ry = rz = 0.0
            palm_trans[:, 0] = dx
            palm_trans[:, 1] = dy
            palm_trans[:, 2] = dz
            palm_rot[:, 0] = rx
            palm_rot[:, 1] = ry
            palm_rot[:, 2] = rz
            apply_palm_motion(env, palm_trans, palm_rot)

            actor_obs, reward_buf, reset_buf, extras = env.step(actions)

            # Print reward terms for env 0
            try:
                logs = extras.get("log", {})
                r_total = float(reward_buf[0].detach().cpu().item())
                r_init = logs.get("reward/init")
                r_grasp = logs.get("reward/grasp")
                r_act = logs.get("reward/action_penalty")
                r_init0 = float(r_init[0].item()) if r_init is not None else 0.0
                r_grasp0 = float(r_grasp[0].item()) if r_grasp is not None else 0.0
                r_act0 = float(r_act[0].item()) if r_act is not None else 0.0
                print(f"[reward] step={step_i:05d} total={r_total:+.3f} init={r_init0:+.3f} grasp={r_grasp0:+.3f} act_pen={r_act0:+.3f}")
                if gui is not None:
                    gui.update(
                        step=step_i,
                        obs=actor_obs,
                        actions=actions,
                        reward_total=reward_buf,
                        reward_logs=logs,
                    )
            except Exception:
                pass
            step_i += 1

            palm_pos = env.hand.data.root_pos_w[0].detach().cpu().numpy()
            obj_pos = env.obj.data.root_pos_w[0].detach().cpu().numpy()

            palm_api.SetTranslate(palm_pos.tolist())
            obj_api.SetTranslate(obj_pos.tolist())

            axis_map = {
                "x": (1.0, 0.0, 0.0),
                "y": (0.0, 1.0, 0.0),
                "z": (0.0, 0.0, 1.0),
                "-x": (-1.0, 0.0, 0.0),
                "-y": (0.0, -1.0, 0.0),
                "-z": (0.0, 0.0, -1.0),
            }
            local_axis = torch.tensor(axis_map[args.palm_axis], device=env.device, dtype=torch.float32)
            palm_dir = quat_apply(env.hand.data.root_quat_w[0], local_axis).detach().cpu().numpy()
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

            # Update PCA axis lines at the object center using object rotation
            if pca_curves:
                obj_quat = env.obj.data.root_quat_w[0]
                # rotate local pca axes by object quat and draw 0.15 m long
                from isaaclab.utils.math import quat_apply
                origin = env.obj.data.root_pos_w[0]
                scale = 0.15
                for i in range(3):
                    axis_local = torch.tensor(axes[i], device=env.device, dtype=torch.float32)
                    axis_world = quat_apply(obj_quat, axis_local)
                    a0 = origin.detach().cpu().numpy()
                    a1 = (origin + scale * axis_world).detach().cpu().numpy()
                    pca_curves[i].GetPointsAttr().Set(
                        [Gf.Vec3f(float(a0[0]), float(a0[1]), float(a0[2])),
                         Gf.Vec3f(float(a1[0]), float(a1[1]), float(a1[2]))]
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
