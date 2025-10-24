from __future__ import annotations

from typing import Dict, Tuple

import torch
from isaaclab.utils.math import quat_mul

from legged_lab.mdp import rewards_unigrasptransformer as reward_terms


def warp_hand_to_default(env, env_ids) -> None:
    """Teleport the Inspire Hand root to the cached default pose."""

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
    base_pos = torch.tensor([0.0, 0.0, 0.75], device=env.device, dtype=root_state.dtype)
    base_rot = torch.tensor([0.0, 0.70710678, 0.0, 0.70710678], device=env.device, dtype=root_state.dtype)
    root_state[indices, :3] = origins[indices] + base_pos
    root_state[indices, 3:7] = base_rot.unsqueeze(0).expand(indices.numel(), -1)
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


def sample_sdf_grid(
    grid: torch.Tensor,
    min_bounds: torch.Tensor,
    max_bounds: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Sample a dense SDF grid via trilinear interpolation."""

    if grid is None:
        return torch.zeros(points.shape[:-1], device=points.device, dtype=points.dtype)

    res = torch.tensor(grid.shape, device=points.device, dtype=points.dtype)
    res_minus_one = res - 1.0

    norm = (points - min_bounds) / (max_bounds - min_bounds)
    norm = torch.clamp(norm, 0.0, 1.0) * res_minus_one

    idx0 = torch.floor(norm).long()
    idx1 = torch.minimum(idx0 + 1, res_minus_one.long())
    frac = norm - idx0.float()

    c000 = grid[idx0[..., 0], idx0[..., 1], idx0[..., 2]]
    c100 = grid[idx1[..., 0], idx0[..., 1], idx0[..., 2]]
    c010 = grid[idx0[..., 0], idx1[..., 1], idx0[..., 2]]
    c110 = grid[idx1[..., 0], idx1[..., 1], idx0[..., 2]]
    c001 = grid[idx0[..., 0], idx0[..., 1], idx1[..., 2]]
    c101 = grid[idx1[..., 0], idx0[..., 1], idx1[..., 2]]
    c011 = grid[idx0[..., 0], idx1[..., 1], idx1[..., 2]]
    c111 = grid[idx1[..., 0], idx1[..., 1], idx1[..., 2]]

    fx, fy, fz = frac[..., 0], frac[..., 1], frac[..., 2]
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


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
    "sample_sdf_grid",
    "compute_reward",
]
