import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the customized InspireHand RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import your env
import torch
from legged_lab.envs.inspirehand.grasp_env_cfg import InspireHandEnv, InspireHandEnvCfg

def main():
    env_cfg = InspireHandEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = InspireHandEnv(cfg=env_cfg)
    # env.reset()

    step = 0
    num_envs = env.num_envs
    act_dim = env.cfg.action_space
    device = env.device  # or env.cfg.sim.device

    while simulation_app.is_running():
        with torch.inference_mode():
            if step % 300 == 0:
                env.reset()
                print("[INFO] reset")

            # sample random actions of correct shape for DirectRLEnv
            # shape: (num_envs, act_dim), dtype float32, on the same device
            acts = torch.randn(num_envs, act_dim, device=device)

            # step the environment
            obs, rew, term, trunc, info = env.step(acts)

            # obs should be (num_envs, num_joints) in your minimal setup
            print("obs =", obs[0, :].cpu().numpy())
            step += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
