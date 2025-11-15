from __future__ import annotations

from typing import Dict, Tuple

import torch
from isaaclab.utils.math import quat_mul

from legged_lab.mdp import rewards_unigrasptransformer as reward_terms


def warp_hand_to_default(env, env_ids) -> None:
    """Teleport the Inspire Hand root to env-configured default pose.

    Uses ``env._default_hand_state`` which the environment sets during __init__.
    The position in ``_default_hand_state`` is treated as local to the env origin
    (consistent with other spawn helpers), so we add the per-env origin here.
    """

    if isinstance(env_ids, torch.Tensor):
        indices = env_ids.to(torch.long)
    elif isinstance(env_ids, slice) or env_ids is None:
        indices = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        indices = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if indices.numel() == 0:
        return

    root_state = env.robot.data.root_state_w.clone()
    origins = env.scene.env_origins.to(device=env.device, dtype=root_state.dtype)

    default_state = getattr(env, "_default_hand_state", None)
    if default_state is None:
        # Fallback to existing root state if not provided
        default_state = root_state

    # Positions are relative to env origin; orientations are absolute (XYZW)
    base_pos_local = default_state[indices, :3]
    base_rot_xyzw = default_state[indices, 3:7]

    root_state[indices, :3] = origins[indices] + base_pos_local
    root_state[indices, 3:7] = base_rot_xyzw
    env.robot.write_root_pose_to_sim(root_state[indices, :7], env_ids=indices)
    zero_vel = torch.zeros((indices.numel(), 6), device=env.device, dtype=root_state.dtype)
    env.robot.write_root_velocity_to_sim(zero_vel, env_ids=indices)


def apply_palm_motion(env, palm_trans: torch.Tensor, palm_rot: torch.Tensor) -> None:
    """Apply palm translation / rotation actions to the robot root."""

    if palm_trans.numel() == 0 and palm_rot.numel() == 0:
        return

    root_state = env.robot.data.root_state_w.clone()

    if palm_trans.numel() != 0:
        clamped_trans = torch.clip(palm_trans, -1.0, 1.0) * env._palm_trans_action_scale
        root_state[:, :3] += clamped_trans

    if palm_rot.numel() != 0:
        clamped_rot = torch.clip(palm_rot, -1.0, 1.0) * env._palm_rot_action_scale
        angles = torch.linalg.norm(clamped_rot, dim=-1, keepdim=True)
        axis = torch.where(
            angles > 1e-6,
            clamped_rot / torch.clamp(angles, min=1e-6),
            torch.zeros_like(clamped_rot),
        )
        half_angles = 0.5 * angles
        delta_quat = torch.zeros((root_state.shape[0], 4), device=env.device, dtype=root_state.dtype)
        delta_quat[:, 0] = torch.cos(half_angles.squeeze(-1))
        delta_quat[:, 1:] = axis * torch.sin(half_angles.squeeze(-1)).unsqueeze(-1)

        current_quat_xyzw = root_state[:, 3:7]
        current_quat = torch.cat((current_quat_xyzw[:, 3:4], current_quat_xyzw[:, :3]), dim=-1)
        updated_quat = quat_mul(delta_quat, current_quat)
        updated_quat_xyzw = torch.cat((updated_quat[:, 1:], updated_quat[:, 0:1]), dim=-1)
        root_state[:, 3:7] = updated_quat_xyzw

    if env.table is not None:
        min_height = env.table.data.root_pos_w[:, 2] + 0.02
        root_offset = getattr(env, "_hand_root_offset", None)
        if root_offset is not None:
            min_height = min_height - root_offset[2]
        root_state[:, 2].clamp_(min=min_height)

    env.robot.write_root_pose_to_sim(root_state[:, :7])
    env.robot.write_root_velocity_to_sim(env._zero_root_vel)

def compute_reward(env) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if env.obj is None:
        zero = torch.zeros(env.num_envs, device=env.device, dtype=env.robot.data.root_pos_w.dtype)
        logs: Dict[str, torch.Tensor] = {
            "reward/init": zero.detach().cpu(),
            "reward/grasp": zero.detach().cpu(),
            "reward/action_penalty": zero.detach().cpu(),
        }
        return zero, logs

    reward, term_logs = reward_terms.compute_hand_reward(env)
    logs: Dict[str, torch.Tensor] = {key: value.detach().cpu() for key, value in term_logs.items()}

    hand_pos = env.hand.data.root_pos_w
    obj_pos = env.obj.data.root_pos_w
    for idx, axis in enumerate("xyz"):
        logs[f"debug/hand_pos_env0_{axis}"] = hand_pos[0, idx].detach().cpu()
        logs[f"debug/object_pos_env0_{axis}"] = obj_pos[0, idx].detach().cpu()

    logs["reward/total"] = reward.detach().cpu()
    return reward, logs


__all__ = [
    "warp_hand_to_default",
    "apply_palm_motion",
    "compute_reward",
]
