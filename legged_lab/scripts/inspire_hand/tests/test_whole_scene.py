#!/usr/bin/env python3
"""Smoke test that spawns the Inspire Hand grasp environment and steps it briefly.

This provides a quick all-in-one check that the USD asset, environment
configuration, and reward plumbing load together without errors.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # 1) Launch Omniverse Kit first
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=args.headless).app  # keep launcher alive until shutdown

    # 2) Now import env/cfg (safe after Kit is up)
    from legged_lab.envs.inspirehand.grasp_cfg import InspireHandGraspEnvCfg
    from legged_lab.envs.inspirehand.grasp_env import InspireHandGraspEnv

    # 3) Small env
    cfg = InspireHandGraspEnvCfg()
    # override a few knobs so we can see several replicas in a tight scene
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 1.5

    # headless flag determines whether Omni renders a window
    env = InspireHandGraspEnv(cfg, headless=args.headless)

    # 4) Step a few frames
    import torch
    # zero torques keep the fingertips relaxed; we only check sim stability here
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(60):
        env.step(actions)

    print("\n[SUCCESS] grasp_env spawn test finished.")
    app.close()

if __name__ == "__main__":
    main()
