from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul


@dataclass
class RewardWeights:
    delta_init_qpos_value: float = -0.1
    right_hand_pose: float = -0.05
    right_hand_dist: float = -1.0
    right_hand_finger_dist: float = -1.0
    right_hand_joint_dist: float = -0.5
    right_hand_body_dist: float = -0.5
    right_hand_exploration_dist: float = -0.2
    delta_target_hand_pca: float = -0.05
    goal_dist: float = -0.5
    goal_rew: float = 1.0
    hand_up: float = 1.0
    bonus: float = 1.0
    hand_up_goal_dist: float = 0.5
    action_penalty: float = 0.002

    max_finger_dist: float = 0.04
    max_hand_dist: float = 0.06
    max_goal_dist: float = 0.05


_TARGET_INIT_QPOS = torch.tensor(
    [
        0.1,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.6,
        0.0,
        -0.1,
        0.0,
        0.6,
        0.0,
        0.0,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        1.2,
        0.0,
        -0.2,
        0.0,
    ],
    dtype=torch.float32,
)
_TARGET_HAND_MASK = torch.tensor(
    [
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=torch.float32,
)
_TARGET_HAND_POSE = torch.zeros_like(_TARGET_INIT_QPOS)


def _match_dim(template: torch.Tensor, target_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = template.to(device=device, dtype=dtype)
    if values.numel() == target_dim:
        return values
    if values.numel() > target_dim:
        return values[:target_dim]
    pad = torch.zeros(target_dim - values.numel(), device=device, dtype=dtype)
    return torch.cat([values, pad], dim=0)


def _get_last_actions(env) -> torch.Tensor:
    buffer = getattr(env.action_buffer, "_circular_buffer", None)
    if buffer is None:
        return torch.zeros(env.num_envs, env.num_actions, device=env.device)
    return buffer.buffer[:, -1, :]


def _batched_quat_apply(quat_xyzw: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
    num_envs, num_vecs, _ = vecs.shape
    quat_flat = quat_xyzw.unsqueeze(1).expand(-1, num_vecs, -1).reshape(-1, 4)
    vec_flat = vecs.reshape(-1, 3)
    rotated = quat_apply(quat_flat, vec_flat)
    return rotated.reshape(num_envs, num_vecs, 3)


def _xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., 3:4], quat[..., :3]), dim=-1)


def _wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., 1:], quat[..., 0:1]), dim=-1)


def _compute_hand_to_object_pca_quat(object_pcas: torch.Tensor, object_rot: torch.Tensor, hand_rot: torch.Tensor):
    num_envs = hand_rot.shape[0]
    object_pca_rotates = _batched_quat_apply(object_rot, object_pcas)
    target_axes = object_pca_rotates[:, 0, :].clone()
    target_axes[:, [0, 1]] = target_axes[:, [1, 0]]
    target_axes[:, -1] = 0.0
    target_axes[:, 0] *= -1.0
    flip_mask = target_axes[:, 1] > 0
    target_axes[flip_mask] *= -1.0
    target_axes = target_axes / torch.clamp(target_axes.norm(dim=1, keepdim=True), min=1e-6)

    hand_axes = torch.tensor([0.0, 0.0, 1.0], device=hand_rot.device, dtype=hand_rot.dtype).repeat(num_envs, 1)
    hand_axes = _batched_quat_apply(hand_rot, hand_axes.unsqueeze(1)).squeeze(1)

    axis = torch.cross(hand_axes, target_axes, dim=-1)
    axis_norm = torch.clamp(axis.norm(dim=1, keepdim=True), min=1e-6)
    axis = axis / axis_norm
    cos_theta = torch.clamp(torch.sum(hand_axes * target_axes, dim=1), -1.0, 1.0)
    theta = torch.acos(cos_theta)
    sin_half = torch.sin(theta / 2.0).unsqueeze(1)
    quat_xyzw = torch.cat([axis * sin_half, torch.cos(theta / 2.0).unsqueeze(1)], dim=1)
    quat_xyzw = quat_xyzw / torch.clamp(quat_xyzw.norm(dim=1, keepdim=True), min=1e-6)
    quat_wxyz = _xyzw_to_wxyz(quat_xyzw)
    hand_rot_wxyz = _xyzw_to_wxyz(hand_rot)
    merged = quat_mul(quat_wxyz, hand_rot_wxyz)
    return object_pca_rotates, _wxyz_to_xyzw(merged)


def _batch_sided_distance(sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.cdist(sources, targets).min(dim=-1).values


def _object_point_cloud_world(env) -> torch.Tensor | None:
    pts_local = getattr(env, "_object_pc_local", None)
    if pts_local is None:
        return None
    num_envs = env.num_envs
    if pts_local.shape[0] != num_envs:
        pts_local = pts_local[0:1].repeat(num_envs, 1, 1)
    pts_local = pts_local.to(env.device, dtype=env.hand.data.root_pos_w.dtype)
    obj_quat = env.obj.data.root_quat_w
    obj_pos = env.obj.data.root_pos_w
    rotated = _batched_quat_apply(obj_quat, pts_local)
    return rotated + obj_pos.unsqueeze(1)


def _object_pca_axes_world(env) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    axes_local = getattr(env, "_object_pca_axes_tensor", None)
    if axes_local is None:
        return None, None
    num_envs = env.num_envs
    if axes_local.shape[0] != num_envs:
        axes_local = axes_local[0:1].repeat(num_envs, 1, 1)
    axes_local = axes_local.to(env.device, dtype=env.hand.data.root_pos_w.dtype)
    object_rot = env.obj.data.root_quat_w
    hand_rot = env.hand.data.root_quat_w
    _, target_rot = _compute_hand_to_object_pca_quat(axes_local, object_rot, hand_rot)
    return axes_local, target_rot


def compute_hand_reward(env, weights: RewardWeights | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if weights is None:
        weights = RewardWeights()

    object_points = _object_point_cloud_world(env)
    if object_points is None:
        raise RuntimeError("Object point cloud not available; ensure pc_fps metadata is provided.")

    actions = _get_last_actions(env)
    action_penalty = torch.sum(actions ** 2, dim=1)

    hand_pos = env.hand.data.root_pos_w
    obj_pos = env.obj.data.root_pos_w
    obj_quat = env.obj.data.root_quat_w
    goal_height = env._default_table_surface + 0.20
    goal_pos = torch.tensor(env._default_object_pos, device=env.device, dtype=obj_pos.dtype).unsqueeze(0).repeat(env.num_envs, 1)
    goal_pos[:, 2] = goal_height

    goal_dist = torch.linalg.norm(goal_pos - obj_pos, dim=1)
    goal_hand_dist = torch.linalg.norm(goal_pos - hand_pos, dim=1)

    tip_indices = getattr(env, "_tip_body_ids", [])
    tip_pos = env.hand.data.body_pos_w[:, tip_indices, :] if len(tip_indices) > 0 else None
    hand_body_pos = env.hand.data.body_pos_w

    right_hand_pc_dist = _batch_sided_distance(hand_pos.unsqueeze(1), object_points).squeeze(-1)
    if tip_pos is not None and tip_pos.shape[1] > 0:
        right_hand_finger_pc_dist = torch.sum(_batch_sided_distance(tip_pos, object_points), dim=-1)
    else:
        right_hand_finger_pc_dist = torch.zeros(env.num_envs, device=env.device, dtype=hand_pos.dtype)
    joint_dist = torch.sum(_batch_sided_distance(hand_body_pos, object_points), dim=-1)
    right_hand_joint_pc_dist = joint_dist * 5.0 / hand_body_pos.shape[1]
    right_hand_body_pc_dist = joint_dist * 5.0 / hand_body_pos.shape[1]

    heighest = torch.max(object_points[:, :, -1], dim=1)[0]
    lowest = torch.min(object_points[:, :, -1], dim=1)[0]

    joint_pos = env.robot.data.joint_pos
    target_init = _match_dim(_TARGET_INIT_QPOS, joint_pos.shape[1], env.device, joint_pos.dtype)
    delta_init_qpos_value = torch.linalg.norm(joint_pos - target_init.unsqueeze(0), dim=1)

    target_hand_pose = _match_dim(_TARGET_HAND_POSE, joint_pos.shape[1], env.device, joint_pos.dtype)
    target_hand_mask = _match_dim(_TARGET_HAND_MASK, joint_pos.shape[1], env.device, joint_pos.dtype)
    delta_qpos = torch.linalg.norm((torch.abs(joint_pos) - target_hand_pose.unsqueeze(0)) * target_hand_mask.unsqueeze(0) * (torch.abs(joint_pos) > 1.0), dim=1)

    _, target_hand_pca_rot = _object_pca_axes_world(env)
    if target_hand_pca_rot is None:
        delta_target_hand_pca = torch.zeros(env.num_envs, device=env.device, dtype=hand_pos.dtype)
    else:
        delta_target_hand_pca = 2.0 * torch.acos(
            torch.clamp(torch.abs(torch.sum(env.hand.data.root_quat_w * target_hand_pca_rot, dim=1)), 0.0, 1.0)
        )

    max_finger_dist = weights.max_finger_dist
    max_hand_dist = weights.max_hand_dist
    max_goal_dist = weights.max_goal_dist

    hold_value = 2
    finger_flag = (right_hand_finger_pc_dist <= max_finger_dist).int()
    hand_flag = (right_hand_pc_dist <= max_hand_dist).int()
    hold_flag = finger_flag + hand_flag

    object_points_sorted, _ = torch.sort(object_points, dim=-1)
    subset = object_points_sorted[:, : object_points_sorted.shape[1] // 4, :]
    rand_ids = torch.randint(0, subset.shape[1], (env.num_envs, 1), device=env.device)
    env_ids = torch.arange(env.num_envs, device=env.device).unsqueeze(1)
    exploration_target = subset[env_ids, rand_ids].squeeze(1)
    right_hand_exploration_dist = torch.linalg.norm(exploration_target - hand_pos, dim=1)

    goal_rew = torch.zeros_like(goal_dist)
    goal_rew = torch.where(hold_flag == hold_value, 1.0 * (0.9 - 2.0 * goal_dist), goal_rew)
    hand_up = torch.zeros_like(goal_dist)
    hand_up = torch.where(lowest >= 0.61, torch.where(hold_flag == hold_value, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
    hand_up = torch.where(
        lowest >= 0.80,
        torch.where(hold_flag == hold_value, 0.2 - goal_hand_dist * 0 + weights.hand_up_goal_dist * (0.2 - goal_dist), hand_up),
        hand_up,
    )
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(hold_flag == hold_value, torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

    init_reward = (
        weights.delta_init_qpos_value * delta_init_qpos_value
        + weights.right_hand_dist * right_hand_pc_dist
        + weights.delta_target_hand_pca * delta_target_hand_pca
        + weights.right_hand_exploration_dist * right_hand_exploration_dist
    )

    grasp_reward = (
        weights.right_hand_body_dist * right_hand_body_pc_dist
        + weights.right_hand_joint_dist * right_hand_joint_pc_dist
        + weights.right_hand_finger_dist * right_hand_finger_pc_dist
        + 2.0 * weights.right_hand_dist * right_hand_pc_dist
        + weights.goal_dist * goal_dist
        + weights.goal_rew * goal_rew
        + weights.hand_up * hand_up
        + weights.bonus * bonus
        + weights.right_hand_pose * delta_qpos
    )

    reward = torch.where(hold_flag != hold_value, init_reward, grasp_reward)
    reward -= weights.action_penalty * action_penalty

    logs: Dict[str, torch.Tensor] = {
        "reward/init": init_reward.detach().cpu(),
        "reward/grasp": grasp_reward.detach().cpu(),
        "reward/action_penalty": action_penalty.detach().cpu(),
        "debug/hold_flag": hold_flag.detach().cpu(),
        "debug/goal_dist": goal_dist.detach().cpu(),
    }

    return reward, logs


__all__ = ["compute_hand_reward", "RewardWeights"]
