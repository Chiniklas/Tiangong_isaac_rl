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

if TYPE_CHECKING:  # pragma: no cover
    from legged_lab.envs.graspxl_rl.graspxl_env import GraspXLEnv


EnvLike = "GraspXLEnv"  # string form avoids typing dependency


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
    "object_lifted",
    "palm_world_position",
    "reach_distance",
    "reach_success",
]
