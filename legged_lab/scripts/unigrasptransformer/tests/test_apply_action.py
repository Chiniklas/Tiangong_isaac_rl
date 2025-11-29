#!/usr/bin/env python3
"""Smoke test: feed a normalized action through the env step and watch state updates."""

from __future__ import annotations

import argparse
from pathlib import Path




# apply action via keyboard




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
    parser.add_argument("--steps", type=int, default=200, help="Number of env steps to run.")
    parser.add_argument("--wrist_fx", type=float, default=0.3, help="Normalized +X wrench component ([-1,1]).")
    parser.add_argument("--finger", type=float, default=0.3, help="Normalized finger action value ([-1,1]).")
    return parser.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher

    # launch app
    app = AppLauncher(headless=args.headless)
    simulation_app = app.app

    import torch
    from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import UnigraspTransformerGraspEnv
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_env import UniGraspTransformerEnv

    # init env
    cfg = UnigraspTransformerGraspEnv()
    cfg.scene.num_envs = 1
    env = UniGraspTransformerEnv(cfg=cfg, headless=args.headless)

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset(env_ids)

    # build scene?

    # apply action via keyboard
    # Build a constant normalized action: small wrist +X force and uniform finger closure.
    action = torch.zeros((env.num_envs, 24), device=env.device)
    action[:, 0] = args.wrist_fx  # fx
    action[:, 6:] = args.finger   # 18 finger joints

    ## step
    for step_idx in range(args.steps):
        obs, rew, reset, extras = env.step(action)
        if step_idx % 10 == 0:
            root = env.robot.data.root_state_w[0]
            obj = env.object.data.root_state_w[0]
            # print observation
            print(f"[{step_idx:04d}] reward={rew[0].item():.4f}, reset={bool(reset[0])}")
            print(f"    palm pos={root[:3].cpu().numpy()}, obj pos={obj[:3].cpu().numpy()}")

    if hasattr(env, "close"):
        env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
