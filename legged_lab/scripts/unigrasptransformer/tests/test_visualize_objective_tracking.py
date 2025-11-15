#!/usr/bin/env python3
"""Extended UniGraspTransformer spawn preview with keyboard control and reward diagnostics.

This script builds the exact same scene as ``test_spawn_scene.py`` and then runs the
simulation indefinitely (or for a fixed number of steps) while exposing palm translation/
rotation control via the keyboard. Per-step reward components are printed to the console,
mirroring what the training loop would log.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from legged_lab.scripts.unigrasptransformer.tests import test_spawn_scene as spawn_scene


spawn_scene._ensure_isaaclab_on_path()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument(
        "--steps",
        type=int,
        default=-1,
        help="Simulation steps to run (-1 keeps sim running until interrupted).",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="Optional spawn override (object_cfg.yaml style). Defaults to config from spawn cfg.",
    )
    parser.add_argument(
        "--disable-keyboard",
        action="store_true",
        help="Disable the interactive keyboard controller even when rendering.",
    )
    parser.add_argument(
        "--palm-axis",
        type=str,
        default="-x",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        help="Local Inspire Hand axis to visualize as the palm direction.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the lightweight state GUI with observations/actions/rewards.",
    )
    return parser.parse_args()


def _print_scene_summary(scene_cfg, spawn_cfg) -> None:
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
    print("[INFO] Object Flags (from YAML):")
    print(f"  - enable={obj_cfg.enable}, spawn_mesh={obj_cfg.spawn_mesh}")
    print(f"  - show_point_cloud={getattr(obj_cfg, 'show_point_cloud', None)}")
    print(f"  - show_pca_axes={getattr(obj_cfg, 'show_pca_axes', None)}")
    if getattr(obj_cfg, 'object_id', None):
        print(f"  - object_id={obj_cfg.object_id}")


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

    return palm_api, obj_api, palm_heading, obj_heading


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
    from legged_lab.envs.unigrasptransformer.grasp_helpers import apply_palm_motion
    from legged_lab.scripts.tools.keyboard_controller import KeyboardController
    from legged_lab.scripts.tools.state_gui import launch_state_gui

    cfg_path: Optional[str] = None
    if args.config_yaml is not None:
        cfg_path = args.config_yaml.expanduser().resolve().as_posix()

    spawn_cfg = UniGraspTransformerSpawnCfg(config_path=cfg_path)

    if spawn_cfg.grasp_object.enable and spawn_cfg.grasp_object.spawn_mesh and not spawn_cfg.grasp_object.static_usd:
        raise RuntimeError(
            "Config error: object.spawn_mesh=true but object.static_usd is not set in config.yaml (and no dataset object was auto-picked)"
        )

    scene_cfg = UniGraspTransformerGraspSceneCfg(spawn=spawn_cfg, num_envs=args.num_envs)
    env_cfg = UniGraspTransformerEnvCfg(scene=scene_cfg)
    env_cfg.scene.max_episode_length_s = 1e9
    env = UniGraspTransformerEnv(env_cfg, headless=args.headless)

    _print_scene_summary(scene_cfg, spawn_cfg)
    palm_api, obj_api, palm_heading_curve, obj_heading_curve = _setup_debug_prims()

    controller = None
    if not args.headless and not args.disable_keyboard:
        controller = KeyboardController()
    gui = launch_state_gui("UGTF Debug") if args.gui else None

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    palm_trans = torch.zeros(env.num_envs, 3, device=env.device)
    palm_rot = torch.zeros_like(palm_trans)

    axis_map = {
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
        "-x": (-1.0, 0.0, 0.0),
        "-y": (0.0, -1.0, 0.0),
        "-z": (0.0, 0.0, -1.0),
    }
    local_axis = torch.tensor(axis_map[args.palm_axis], device=env.device, dtype=torch.float32)

    try:
        step_i = 0
        run_forever = args.steps < 0
        while run_forever or step_i < args.steps:
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

            logs = extras.get("log", {})
            r_total = float(reward_buf[0].detach().cpu().item())
            r_init = logs.get("reward/init")
            r_grasp = logs.get("reward/grasp")
            r_act = logs.get("reward/action_penalty")
            r_init0 = float(r_init[0].item()) if r_init is not None else 0.0
            r_grasp0 = float(r_grasp[0].item()) if r_grasp is not None else 0.0
            r_act0 = float(r_act[0].item()) if r_act is not None else 0.0
            print(
                f"[reward] step={step_i:05d} total={r_total:+.3f} init={r_init0:+.3f} "
                f"grasp={r_grasp0:+.3f} act_pen={r_act0:+.3f}"
            )
            if gui is not None:
                gui.update(step=step_i, obs=actor_obs, actions=actions, reward_total=reward_buf, reward_logs=logs)

            step_i += 1

            palm_pos = env.hand.data.root_pos_w[0].detach().cpu().numpy()
            obj_pos = env.obj.data.root_pos_w[0].detach().cpu().numpy()

            palm_api.SetTranslate(palm_pos.tolist())
            obj_api.SetTranslate(obj_pos.tolist())

            palm_dir = quat_apply(env.hand.data.root_quat_w[0], local_axis).detach().cpu().numpy()
            obj_dir = quat_apply(
                env.obj.data.root_quat_w[0], torch.tensor([0.0, 0.0, 1.0], device=env.device)
            ).detach().cpu().numpy()

            palm_line = [palm_pos, palm_pos + 0.12 * palm_dir]
            obj_line = [obj_pos, obj_pos + 0.12 * obj_dir]
            from pxr import Gf

            palm_heading_curve.GetPointsAttr().Set(
                [
                    Gf.Vec3f(float(palm_line[0][0]), float(palm_line[0][1]), float(palm_line[0][2])),
                    Gf.Vec3f(float(palm_line[1][0]), float(palm_line[1][1]), float(palm_line[1][2])),
                ]
            )
            obj_heading_curve.GetPointsAttr().Set(
                [
                    Gf.Vec3f(float(obj_line[0][0]), float(obj_line[0][1]), float(obj_line[0][2])),
                    Gf.Vec3f(float(obj_line[1][0]), float(obj_line[1][1]), float(obj_line[1][2])),
                ]
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
