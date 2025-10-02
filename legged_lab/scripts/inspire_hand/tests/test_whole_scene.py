#!/usr/bin/env python3
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=args.headless).app

    from legged_lab.envs.inspirehand.grasp_cfg import InspireHandGraspEnvCfg
    from legged_lab.envs.inspirehand.grasp_env import InspireHandGraspEnv

    cfg = InspireHandGraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 1.5

    env = InspireHandGraspEnv(cfg, headless=args.headless)

    # Confirm entities
    try:
        hand = env.scene["robot"]
        table = env.scene["table"]
        obj = env.scene["object"]
        print("[CHECK] robot/table/object present: OK")
        print("[INFO] table z (first env):", float(table.data.root_pos_w[0, 2]))
        print("[INFO] object pos (first env):", obj.data.root_pos_w[0].tolist())
    except KeyError as e:
        print("[ERROR] missing entity:", e)
        app.close()
        return

    # Step a bit
    import torch
    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(60):
        env.step(actions)

    print("\n[SUCCESS] grasp_env spawn test finished.")
    app.close()

if __name__ == "__main__":
    main()
