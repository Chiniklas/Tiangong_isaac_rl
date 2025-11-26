#!/usr/bin/env python3
"""Quick test: apply a small wrench to the hand root and observe palm motion."""

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
    parser.add_argument("--steps", type=int, default=1200, help="Number of sim steps to run.")
    parser.add_argument("--force", type=float, default=5.0, help="Force magnitude (N) along +X in world.")
    return parser.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless)
    simulation_app = app.app

    import torch
    from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import UnigraspTransformerGraspEnv
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_env import UniGraspTransformerEnv

    cfg = UnigraspTransformerGraspEnv()
    cfg.scene.num_envs = 1
    env = UniGraspTransformerEnv(cfg=cfg, headless=args.headless)

    # Reset and hold default finger pose
    env.scene.reset(torch.arange(env.num_envs, device=env.device))

    # Build a small wrench on the root (body id 0), world frame
    forces = torch.zeros((env.num_envs, 1, 3), device=env.device)
    torques = torch.zeros((env.num_envs, 1, 3), device=env.device)
    forces[:, 0, 0] = args.force  # +X

    # Track palm pose
    for step in range(args.steps):
        env.robot.set_external_force_and_torque(forces=forces, torques=torques, body_ids=[0], is_global=True)
        env.robot.set_joint_position_target(env.robot.data.default_joint_pos)
        env.scene.write_data_to_sim()
        env.sim.step(render=not args.headless)
        env.scene.update(dt=env.physics_dt)

        root_state = env.robot.data.root_state_w[0]
        pos = root_state[:3].cpu().numpy()
        quat = root_state[3:7].cpu().numpy()
        if step % 10 == 0:
            print(f"[step {step}] palm pos: {pos}, quat: {quat}")

    if hasattr(env, "close"):
        env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
