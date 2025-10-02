#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # 1) Launch Omniverse Kit first
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=args.headless).app

    # 2) Now import env/cfg (safe after Kit is up)
    from legged_lab.envs.inspirehand.grasp_cfg import InspireHandGraspEnvCfg
    from legged_lab.envs.inspirehand.grasp_env import InspireHandGraspEnv

    # 3) Small env
    cfg = InspireHandGraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 1.5

    env = InspireHandGraspEnv(cfg, headless=args.headless)

    # 4) Step a few frames
    import torch
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(60):
        env.step(actions)

    print("\n[SUCCESS] grasp_env spawn test finished.")
    app.close()

if __name__ == "__main__":
    main()
