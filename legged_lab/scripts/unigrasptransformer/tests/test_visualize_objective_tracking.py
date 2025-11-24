#!/usr/bin/env python3
"""Extended UniGraspTransformer spawn preview with keyboard control and reward diagnostics.

This script builds the exact same scene as ``test_spawn_scene.py`` and then runs the
simulation indefinitely (or for a fixed number of steps) while exposing palm translation/
rotation control via the keyboard. Per-step reward components are printed to the console,
mirroring what the training loop would log.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def _format_reward_log(step_idx: int, logs) -> list[str]:
    if not logs:
        return [f"[reward] step={step_idx:05d} (no reward terms reported)"]

    def _get(key: str) -> Optional[float]:
        return _tensor_scalar(logs.get(key))

    base_parts = [f"[reward] step={step_idx:05d}"]
    total = _get("reward/total")
    if total is not None:
        base_parts.append(f"total={total:+.3f}")
    action_penalty = _get("reward/action_penalty")
    if action_penalty is not None:
        base_parts.append(f"action_penalty={action_penalty:+.3f}")
    hold_flag = _get("reward/hold_flag")
    if hold_flag is not None:
        base_parts.append(f"hold_flag={hold_flag:+.3f}")
    base_line = " ".join(base_parts)

    phase_summary: dict[str, float] = {}
    phase_terms: dict[str, list[tuple[str, float]]] = defaultdict(list)

    for key, value in logs.items():
        if not key.startswith("reward/"):
            continue
        parts = key.split("/")
        if len(parts) == 2:
            phase = parts[1]
            if phase in {"total", "action_penalty", "hold_flag"}:
                continue
            scalar = _tensor_scalar(value)
            if scalar is not None:
                phase_summary[phase] = scalar
        elif len(parts) > 2:
            phase = parts[1]
            term = "/".join(parts[2:])
            scalar = _tensor_scalar(value)
            if scalar is not None:
                phase_terms[phase].append((term, scalar))

    lines = [base_line]
    # Prefer printing grasp before init, then any others alphabetically.
    preferred_order = ["grasp", "init"]
    remaining = [p for p in sorted(phase_summary.keys()) if p not in preferred_order]
    for phase in preferred_order + remaining:
        if phase not in phase_summary and phase not in phase_terms:
            continue
        summary_val = phase_summary.get(phase)
        # If no explicit summary, try to infer from a 'total' term.
        if summary_val is None:
            for term, scalar in phase_terms.get(phase, []):
                if term == "total":
                    summary_val = scalar
                    break
        if summary_val is not None:
            lines.append(f"  {phase}: {summary_val:+.3f}")
        terms = phase_terms.get(phase, [])
        # Move 'total' to the end for readability.
        terms_sorted = sorted([t for t in terms if t[0] != "total"], key=lambda item: item[0])
        if any(t[0] == "total" for t in terms):
            terms_sorted.append(("total", next(v for k, v in terms if k == "total")))
        for term, scalar in terms_sorted:
            lines.append(f"    {term}: {scalar:+.3f}")
    if len(lines) == 1:
        lines[0] += " (no reward terms reported)"
    return lines


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

            logs = extras.get("log", {})
            for line in _format_reward_log(step_i, logs):
                print(line)
            if gui is not None:
                gui.update(step=step_i, obs=actor_obs, actions=actions, reward_total=reward_buf, reward_logs=logs)
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
