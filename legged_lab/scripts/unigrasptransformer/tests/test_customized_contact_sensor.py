#!/usr/bin/env python3
"""Extended UniGraspTransformer spawn preview with keyboard control and contact wrench logging.

This script mirrors ``test_spawn_scene.py`` but logs per-finger contact forces. It subscribes to
PhysX contact reports to accumulate forces per fingertip (torques are set to zero as PhysX reports
only the normal/tangential force at the contact point)."""

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
        obj_status = f"USD: {obj_cfg.object_path or '<missing>'}"
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


def _tensor_scalar(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "view"):
            value = value.view(-1)[0]
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except Exception:
        return None


def _log_contact_wrenches(env, step_idx: int) -> None:
    """Log per-finger contact wrench (Fx,Fy,Fz,Tx,Ty,Tz) for env 0 using the contact sensor.

    Note: the built-in contact sensor only reports normal forces, not tangential forces or torques.
    We log torques as zero placeholders.
    """
    try:
        forces = getattr(env, "_contact_force_accum", None)
        if not forces and getattr(env, "_use_contact_sensor_fallback", False):
            # fallback to contact sensor net forces
            cs = getattr(env, "contact_sensor", None)
            if cs is not None and cs.data.net_forces_w is not None:
                forces = {}
                env_idx = 0
                body_names = getattr(cs, "body_names", [])
                tip_indices = getattr(env, "_tip_body_ids", None)
                if tip_indices is None or len(tip_indices) == 0:
                    tip_indices = list(range(min(5, cs.data.net_forces_w.shape[1])))
                for idx in tip_indices:
                    fname = body_names[idx] if idx < len(body_names) else f"body_{idx}"
                    forces[fname] = cs.data.net_forces_w[env_idx, idx].to("cpu")
        if not forces:
            print(f"[contact] step={step_idx:05d} (no contact forces accumulated)")
            return
        lines = [f"[contact] step={step_idx:05d}"]
        for name, vec in forces.items():
            fx, fy, fz = vec.tolist()
            lines.append(f"  {name}: Fx={fx:+.3f} Fy={fy:+.3f} Fz={fz:+.3f} Tx=+0.000 Ty=+0.000 Tz=+0.000")
        print("\n".join(lines))
    except Exception as exc:
        print(f"[contact] step={step_idx:05d} (failed to read contact forces: {exc})")


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

    if spawn_cfg.grasp_object.enable and spawn_cfg.grasp_object.spawn_mesh and not spawn_cfg.grasp_object.object_path:
        raise RuntimeError(
            "Config error: object.spawn_mesh=true but object.object_path is not set to a USD file or object directory (and no dataset object was auto-picked)"
        )

    scene_cfg = UniGraspTransformerGraspSceneCfg(spawn=spawn_cfg, num_envs=args.num_envs)
    env_cfg = UniGraspTransformerEnvCfg(scene=scene_cfg)
    env_cfg.scene.max_episode_length_s = 1e9
    env = UniGraspTransformerEnv(env_cfg, headless=args.headless)

    _print_scene_summary(scene_cfg, spawn_cfg)

    # Subscribe to PhysX contact reports to accumulate per-body forces (normal + tangential).
    # Torque is logged as zero since contact reports expose forces at contact points only.
    try:
        import omni.physx as physx
        import torch

        # Build a set of fingertip prim path suffixes to match against actor paths.
        tip_indices = getattr(env, "_tip_body_ids", [])
        tip_names = []
        if tip_indices:
            body_names = getattr(env.hand.data, "body_names", [])
            for idx in tip_indices:
                if idx < len(body_names):
                    tip_names.append(body_names[idx])
        tip_suffixes = [name.split(":")[-1] if ":" in name else name for name in tip_names]
        # Fallback: treat all hand bodies as potential tips if mapping is missing.
        if not tip_suffixes:
            tip_suffixes = [name.split(":")[-1] for name in getattr(env.hand.data, "body_names", [])]

        env._contact_force_accum = {}

        def _on_contact_report(events):
            if not tip_suffixes:
                return
            accum = env._contact_force_accum
            accum.clear()
            for event in events:
                for contact in event.contacts:
                    # Each contact has actors with prim paths; match fingertip suffix.
                    for actor_path, force in (
                        (contact.actor0, contact.impulse0),
                        (contact.actor1, contact.impulse1),
                    ):
                        if actor_path is None or force is None:
                            continue
                        name = actor_path.split("/")[-1]
                        short = name.split(":")[-1]
                        if short not in tip_suffixes:
                            continue
                        # impulse is a vec3; approximate force by impulse/dt=1 step
                        try:
                            f = torch.tensor(force, device=env.device, dtype=torch.float32)
                        except Exception:
                            continue
                        accum[short] = accum.get(short, torch.zeros(3, device=env.device, dtype=torch.float32)) + f

        physx.get_physx_interface().subscribe_contact_report_events(_on_contact_report)
    except Exception as exc:
        print(f"[WARN] PhysX contact subscription failed: {exc}")
        env._contact_force_accum = {}
        env._use_contact_sensor_fallback = True

    controller = None
    if not args.headless and not args.disable_keyboard:
        controller = KeyboardController()
    gui = launch_state_gui("UGTF Debug") if args.gui else None
    contact_viewer = spawn_scene._maybe_create_contact_viewer(env, args.headless)

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    palm_trans = torch.zeros(env.num_envs, 3, device=env.device)
    palm_rot = torch.zeros_like(palm_trans)

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

            _log_contact_wrenches(env, step_i)
            if gui is not None:
                gui.update(step=step_i, obs=actor_obs, actions=actions, reward_total=reward_buf, reward_logs={})
            if contact_viewer:
                contact_viewer.update()

            step_i += 1

    except KeyboardInterrupt:
        print("\n[INFO] Visualization interrupted by user. Shutting down...")
    finally:
        if controller is not None:
            controller.shutdown()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()


# TODO:
# 1- implement a joint position controller to test the delta_init_qpos_value reward
# 2- test delta_target_hand_pca
