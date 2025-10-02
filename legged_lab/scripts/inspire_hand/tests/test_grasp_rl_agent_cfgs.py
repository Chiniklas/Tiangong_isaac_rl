# this script is for testing grasp_cfg.py and grasp_env.py

# Minimal runtime test for InspireHandGraspEnv + cfg

import argparse
from isaaclab.app import AppLauncher

# CLI passthrough so you can add --headless later if you extend
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args)
simulation_app = app.app

import torch
from legged_lab.envs.inspirehand.grasp_cfg import InspireHandGraspEnvCfg
from legged_lab.envs.inspirehand.grasp_env import InspireHandGraspEnv


def main():
    cfg = InspireHandGraspEnvCfg()
    env = InspireHandGraspEnv(cfg, headless=True)

    print("[INFO] Env online.")
    print(f"  num_envs   : {env.num_envs}")
    print(f"  num_actions: {env.num_actions}")

    # Pull observations via BaseEnv API
    obs, extras = env.get_observations()
    print(f"  obs shape  : {tuple(obs.shape)}")

    # Step a few times with random actions
    for i in range(3):
        a = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, rew, done, ext = env.step(a)
        print(f"  step {i}: reward mean={rew.mean().item():.4f}, done={done.any().item()}")

    print("[SUCCESS] InspireHandGraspEnv build/step test passed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
