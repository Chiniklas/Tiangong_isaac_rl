from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from isaaclab.utils.math import quat_mul

from legged_lab.assets.inspirehand.object_library import GraspObjectInfo


def sample_object_info(env) -> GraspObjectInfo:
    """Return a random grasp object metadata entry."""

    return env._object_rng.choice(env._object_infos)


def apply_object_info(env, info: GraspObjectInfo, *, initial: bool = False) -> None:
    """Configure the current grasp object and update cached properties."""

    env._current_object = info
    load_object_sdfs(env, info)

    lowest = info.lowest_point if info.lowest_point is not None else None
    if lowest is None and env._non_sdf_min is not None:
        lowest = float(env._non_sdf_min[2].detach().cpu())
    if lowest is None and env._aff_sdf_min is not None:
        lowest = float(env._aff_sdf_min[2].detach().cpu())
    if lowest is None:
        lowest = 0.0
    env._current_lowest = float(lowest if lowest <= 0.0 else -lowest)

    base_x, base_y, _ = env._default_object_pos
    # Always reset to the default pose so preview runs remain deterministic.
    dx = dy = 0.0
    yaw = 0.0

    table_surface = env._default_table_surface
    z_pos = table_surface - env._current_lowest + env._object_clearance

    dtype = env._zero_root_vel.dtype
    pos_offset = torch.tensor([base_x + dx, base_y + dy, z_pos], device=env.device, dtype=dtype)
    origins = env.scene.env_origins.to(device=env.device, dtype=dtype)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    quat_xyzw = torch.tensor([0.0, 0.0, sy, cy], device=env.device, dtype=dtype)

    pose = torch.zeros((env.num_envs, 7), device=env.device, dtype=dtype)
    pose[:, :3] = origins + pos_offset
    pose[:, 3:] = quat_xyzw
    env.obj.write_root_pose_to_sim(pose)

    vel = torch.zeros((env.num_envs, 6), device=env.device, dtype=dtype)
    env.obj.write_root_velocity_to_sim(vel)


def load_object_sdfs(env, info: GraspObjectInfo) -> None:
    """Load affordance / non-affordance SDF grids for the current object."""

    def _load(path: Optional[Path]):
        if path is None or not path.exists():
            return None, None, None
        data = np.load(path)
        grid = torch.from_numpy(data["sdf"]).to(env.device)
        min_bounds = torch.from_numpy(data["min_bounds"]).to(env.device).float()
        max_bounds = torch.from_numpy(data["max_bounds"]).to(env.device).float()
        return grid, min_bounds, max_bounds

    env._aff_sdf_grid, env._aff_sdf_min, env._aff_sdf_max = _load(info.affordance_sdf)
    env._non_sdf_grid, env._non_sdf_min, env._non_sdf_max = _load(info.non_affordance_sdf)
    env._latest_aff_sdf = None
    env._latest_non_sdf = None


def warp_hand_to_default(env, env_ids) -> None:
    """Teleport the Inspire Hand root to the cached default pose."""

    if not hasattr(env, "_default_hand_state"):
        return

    if isinstance(env_ids, torch.Tensor):
        indices = env_ids.to(torch.long)
    elif isinstance(env_ids, slice) or env_ids is None:
        indices = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        indices = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if indices.numel() == 0:
        return

    root_state = env.robot.data.root_state_w.clone()
    default_state = env._default_hand_state.clone()
    origins = env.scene.env_origins.to(device=env.device, dtype=default_state.dtype)
    default_state[:, :3] = origins + default_state[:, :3]
    root_state[indices] = default_state[indices]
    env.robot.write_root_pose_to_sim(root_state[indices, :7], env_ids=indices)
    env.robot.write_root_velocity_to_sim(env._zero_root_vel[indices], env_ids=indices)


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
