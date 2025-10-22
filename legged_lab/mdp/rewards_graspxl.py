# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# GraspXL reward primitives:
# --------------------------
# This module mirrors the structure of :mod:`legged_lab.mdp.rewards` while
# providing building blocks tailored to the GraspXL manipulation environments.
# Each function returns an unscaled tensor so task code can combine and weight
# the components as needed.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul


if TYPE_CHECKING:  # pragma: no cover
    from legged_lab.envs.graspxl_rl.graspxl_env import GraspXLEnv


EnvLike = "GraspXLEnv"  # string form avoids typing dependency


def _zeros_like_env(env: EnvLike, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Utility returning a zeros tensor shaped like ``env.num_envs`` x 3."""

    ref_tensor = env.robot.data.root_pos_w
    if dtype is None:
        dtype = ref_tensor.dtype
    return torch.zeros(env.num_envs, 3, device=env.device, dtype=dtype)


def palm_linear_velocity(env: EnvLike) -> torch.Tensor:
    """Palm linear velocity in world frame with safe fallbacks."""

    hand_data = env.hand.data
    lin_vel = getattr(hand_data, "root_lin_vel_w", None)
    if lin_vel is None:
        lin_vel = getattr(hand_data, "root_com_lin_vel_w", None)
    if lin_vel is None:
        lin_vel = _zeros_like_env(env, dtype=hand_data.root_pos_w.dtype)
    return lin_vel


def palm_angular_velocity(env: EnvLike) -> torch.Tensor:
    """Palm angular velocity in world frame with safe fallbacks."""

    hand_data = env.hand.data
    ang_vel = getattr(hand_data, "root_ang_vel_w", None)
    if ang_vel is None:
        ang_vel = getattr(hand_data, "root_com_ang_vel_w", None)
    if ang_vel is None:
        ang_vel = _zeros_like_env(env, dtype=hand_data.root_pos_w.dtype)
    return ang_vel


def object_linear_velocity(env: EnvLike) -> torch.Tensor:
    """Object linear velocity; zeros if the object is absent."""

    if env.obj is None:
        return _zeros_like_env(env)

    obj_data = env.obj.data
    lin_vel = getattr(obj_data, "root_lin_vel_w", None)
    if lin_vel is None:
        body_lin_vel = getattr(obj_data, "body_lin_vel_w", None)
        if body_lin_vel is not None and body_lin_vel.ndim >= 3 and body_lin_vel.shape[1] > 0:
            lin_vel = body_lin_vel[:, 0, :]
    if lin_vel is None:
        lin_vel = _zeros_like_env(env, dtype=obj_data.root_pos_w.dtype)
    return lin_vel


def object_angular_velocity(env: EnvLike) -> torch.Tensor:
    """Object angular velocity; zeros if the object is absent."""

    if env.obj is None:
        return _zeros_like_env(env)

    obj_data = env.obj.data
    ang_vel = getattr(obj_data, "root_ang_vel_w", None)
    if ang_vel is None:
        body_ang_vel = getattr(obj_data, "body_ang_vel_w", None)
        if body_ang_vel is not None and body_ang_vel.ndim >= 3 and body_ang_vel.shape[1] > 0:
            ang_vel = body_ang_vel[:, 0, :]
    if ang_vel is None:
        ang_vel = _zeros_like_env(env, dtype=obj_data.root_pos_w.dtype)
    return ang_vel


def wrist_linear_velocity_penalty(env: EnvLike) -> torch.Tensor:
    """Squared palm linear velocity magnitude."""

    vel = palm_linear_velocity(env)
    return torch.sum(vel**2, dim=1)


def wrist_angular_velocity_penalty(env: EnvLike) -> torch.Tensor:
    """Squared palm angular velocity magnitude."""

    ang_vel = palm_angular_velocity(env)
    return torch.sum(ang_vel**2, dim=1)


def object_linear_velocity_penalty(env: EnvLike) -> torch.Tensor:
    """Squared magnitude of the object's linear velocity relative to the palm."""

    obj_vel = object_linear_velocity(env)
    rel = obj_vel - palm_linear_velocity(env)
    return torch.sum(rel**2, dim=1)


def object_angular_velocity_penalty(env: EnvLike) -> torch.Tensor:
    """Squared magnitude of the object's angular velocity relative to the palm."""

    obj_ang = object_angular_velocity(env)
    rel = obj_ang - palm_angular_velocity(env)
    return torch.sum(rel**2, dim=1)


def hand_center_distance(env: EnvLike) -> torch.Tensor:
    """Squared distance between the palm center and object center."""

    if env.obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    hand_pos = palm_world_position(env)
    obj_pos = env.obj.data.root_pos_w
    diff = obj_pos - hand_pos
    return torch.sum(diff**2, dim=1)


def heading_alignment_penalty(env: EnvLike) -> torch.Tensor:
    """Squared misalignment between palm approach axis and vector to the object."""

    if env.obj is None:
        return torch.zeros(env.num_envs, device=env.device)

    palm_pos = palm_world_position(env)
    obj_pos = env.obj.data.root_pos_w
    target_vec = obj_pos - palm_pos
    target_norm = torch.linalg.norm(target_vec, dim=1, keepdim=True).clamp(min=1e-6)
    target_dir = target_vec / target_norm

    palm_quat_xyzw = env.hand.data.root_quat_w
    palm_quat = torch.cat((palm_quat_xyzw[:, 3:4], palm_quat_xyzw[:, :3]), dim=-1)
    local_axis = palm_quat.new_zeros((env.num_envs, 3))
    local_axis[:, 2] = 1.0
    palm_dir = quat_apply(palm_quat, local_axis)
    palm_dir_norm = torch.linalg.norm(palm_dir, dim=1, keepdim=True).clamp(min=1e-6)
    palm_dir = palm_dir / palm_dir_norm
    diff = palm_dir - target_dir
    return torch.sum(diff**2, dim=1)


def wrist_rotation_alignment_penalty(env: EnvLike) -> torch.Tensor:
    """Squared magnitude of wrist orientation deviation from the spawn reference."""

    palm_quat_xyzw = env.hand.data.root_quat_w
    palm_quat = torch.cat((palm_quat_xyzw[:, 3:4], palm_quat_xyzw[:, :3]), dim=-1)
    default_xyzw = env._default_hand_state[:, 3:7]
    default_quat = torch.cat((default_xyzw[:, 3:4], default_xyzw[:, :3]), dim=-1)
    rel = quat_mul(quat_conjugate(default_quat), palm_quat)
    return torch.sum(rel[:, 1:] ** 2, dim=1)


def palm_world_position(env: EnvLike) -> torch.Tensor:
    """Return the palm position in world coordinates."""

    tf = getattr(env.hand.data, "link_tf_w", None)
    if tf is not None and tf.ndim == 4 and tf.shape[1] > 0:
        palm_tf = tf[:, env.hand.root_physx_view.root_body_handle, :, :]
        return palm_tf[:, :3, 3]
    return env.hand.data.root_pos_w


def action_smoothness(env: EnvLike) -> torch.Tensor:
    """Squared joint action magnitude encouraging smooth behaviour."""

    joint_dim = getattr(env, "_joint_action_dim", env.robot.data.default_joint_pos.shape[1])
    last_joint_action = env.action_buffer._circular_buffer.buffer[:, -1, :joint_dim]
    return torch.sum(last_joint_action**2, dim=1)


def reach_distance(env: EnvLike) -> torch.Tensor:
    """Euclidean distance between palm and grasp object."""

    if env.obj is None:
        return torch.zeros(env.num_envs, device=env.device)
    hand_pos = palm_world_position(env)
    obj_pos = env.obj.data.root_pos_w
    return torch.linalg.norm(obj_pos - hand_pos, dim=1)


def reach_success(env: EnvLike, falloff: float) -> torch.Tensor:
    """Exponential reach reward based on palm-to-object distance."""

    return torch.exp(-falloff * reach_distance(env))


def object_lifted(
    env: EnvLike,
    height_buffer: float,
    *,
    ground_height: float,
) -> torch.Tensor:
    """Return a float mask indicating whether the object is considered lifted."""

    if env.obj is None:
        return torch.zeros(env.num_envs, device=env.device)

    obj_pos = env.obj.data.root_pos_w
    if env.table is not None:
        table_z = env.table.data.root_pos_w[:, 2] + env._table_thickness * 0.5
        object_bottom = obj_pos[:, 2] + env._current_lowest
        lifted = object_bottom > (table_z + height_buffer)
    else:
        lifted = obj_pos[:, 2] > ground_height
    return lifted.float()


def hold_success(
    env: EnvLike,
    lifted_mask: torch.Tensor,
    *,
    min_duration: int,
) -> torch.Tensor:
    """Return a float mask indicating whether the object has been held long enough."""

    env._hold_counter = torch.where(lifted_mask.bool(), env._hold_counter + 1, torch.zeros_like(env._hold_counter))
    return (env._hold_counter > min_duration).float()


def affordance_sdf_bonus(env: EnvLike, decay: float) -> torch.Tensor:
    """Positive shaping term encouraging fingers to stay close to affordance regions."""

    if env._latest_aff_sdf is None:
        return torch.zeros(env.num_envs, device=env.device)
    aff_mean = torch.abs(env._latest_aff_sdf).mean(dim=1)
    return torch.exp(-decay * aff_mean)


def non_affordance_penalty(env: EnvLike) -> torch.Tensor:
    """Penalty for intruding into non-affordance regions."""

    if env._latest_non_sdf is None:
        return torch.zeros(env.num_envs, device=env.device)
    non_penalty = torch.relu(-env._latest_non_sdf).mean(dim=1)
    return non_penalty


__all__ = [
    "action_smoothness",
    "affordance_sdf_bonus",
    "hold_success",
    "non_affordance_penalty",
    "object_angular_velocity",
    "object_angular_velocity_penalty",
    "object_linear_velocity",
    "object_linear_velocity_penalty",
    "object_lifted",
    "hand_center_distance",
    "heading_alignment_penalty",
    "palm_angular_velocity",
    "palm_linear_velocity",
    "palm_world_position",
    "reach_distance",
    "reach_success",
    "wrist_angular_velocity_penalty",
    "wrist_linear_velocity_penalty",
    "wrist_rotation_alignment_penalty",
]
