"""Minimal runtime test that exercises InspireHandGraspEnv and its configs.

This mirrors the training stack: boot Isaac, build the environment from
``InspireHandGraspEnvCfg``, query observations, and step a few frames with
zero actions to confirm reward/termination tensors behave.
"""

import argparse
from dataclasses import asdict
from isaaclab.app import AppLauncher

# CLI passthrough so you can add --headless later if you extend
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args)
simulation_app = app.app

import torch
from legged_lab.envs.inspirehand.graspxl_cfg import (
    InspireHandGraspAgentCfg,
    InspireHandGraspEnvCfg,
)
from legged_lab.envs.inspirehand.graspxl_env import InspireHandGraspEnv


def main():
    cfg = InspireHandGraspEnvCfg()
    agent_cfg = InspireHandGraspAgentCfg()
    # Force headless True so smoke tests run in CI without a viewer
    env = InspireHandGraspEnv(cfg, headless=True)

    print("[INFO] Env online.")
    print(f"  num_envs   : {env.num_envs}")
    print(f"  num_actions: {env.num_actions}")
    print("  action layout:")
    print(f"    joint DOFs : {env._joint_action_dim}")
    print(f"    palm trans : {env._palm_trans_action_dim}")
    print(f"    palm rot   : {env._palm_rot_action_dim}")

    # Pull observations via BaseEnv API
    obs, extras = env.get_observations()
    critic_obs = extras.get("observations", {}).get("critic", None)
    print("  state space:")
    print(f"    actor obs shape : {tuple(obs.shape)}")
    if critic_obs is not None:
        print(f"    critic obs shape: {tuple(critic_obs.shape)}")
    else:
        print("    critic obs shape: <unavailable>")
    print(f"    actor history   : {cfg.robot.actor_obs_history_length}")
    print(f"    critic history  : {cfg.robot.critic_obs_history_length}")

    # network layout
    policy_cfg = agent_cfg.policy
    algo_cfg = agent_cfg.algorithm
    print("  network structure:")
    print(f"    actor hidden dims : {list(policy_cfg.actor_hidden_dims)}")
    print(f"    critic hidden dims: {list(policy_cfg.critic_hidden_dims)}")
    print(f"    activation        : {policy_cfg.activation}")
    print("  training hyperparameters:")
    print(f"    steps per env     : {agent_cfg.num_steps_per_env}")
    print(f"    mini batches      : {algo_cfg.num_mini_batches}")
    print(f"    learning epochs   : {algo_cfg.num_learning_epochs}")
    print(f"    entropy coef      : {algo_cfg.entropy_coef}")

    # reward structure
    reward_cfg = cfg.reward_scales
    reward_dict = asdict(reward_cfg)
    print("  reward structure:")
    for key, value in reward_dict.items():
        print(f"    {key}: {value}")

    # Step a few times with random actions
    for i in range(3):
        # stick to zero torques so we only validate rollout plumbing
        a = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, rew, done, ext = env.step(a)
        print(f"  step {i}: reward mean={rew.mean().item():.4f}, done={done.any().item()}")

    print("[SUCCESS] InspireHandGraspEnv build/step test passed.")


if __name__ == "__main__":
    main()
    # keep the launcher tear-down explicit so GPU contexts release cleanly
    simulation_app.close()
