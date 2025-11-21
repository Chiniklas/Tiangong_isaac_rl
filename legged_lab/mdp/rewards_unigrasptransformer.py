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


_RAW_TARGET_JOINT_ORDER = (
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
)
_RAW_TARGET_INIT_QPOS = torch.tensor(
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
_RAW_TARGET_HAND_MASK = torch.tensor(
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
_RAW_TARGET_HAND_POSE = torch.zeros_like(_RAW_TARGET_INIT_QPOS)

_TARGET_JOINT_NAMES = (
    "FFJ1",
    "FFJ2",
    "FFJ3",
    "FFJ4",
    "MFJ1",
    "MFJ2",
    "MFJ3",
    "MFJ4",
    "RFJ1",
    "RFJ2",
    "RFJ3",
    "RFJ4",
    "LFJ1",
    "LFJ2",
    "LFJ3",
    "LFJ4",
    "LFJ5",
    "THJ1",
    "THJ2",
    "THJ3",
    "THJ4",
    "THJ5",
)


def _reorder_raw_tensor(raw_tensor: torch.Tensor, raw_order: tuple[str, ...]) -> torch.Tensor:
    raw_map = {name: value for name, value in zip(raw_order, raw_tensor.tolist())}
    return torch.tensor([raw_map[name] for name in _TARGET_JOINT_NAMES], dtype=raw_tensor.dtype)


_TARGET_INIT_QPOS = _reorder_raw_tensor(_RAW_TARGET_INIT_QPOS, _RAW_TARGET_JOINT_ORDER)
_TARGET_HAND_MASK = _reorder_raw_tensor(_RAW_TARGET_HAND_MASK, _RAW_TARGET_JOINT_ORDER)
_TARGET_HAND_POSE = _reorder_raw_tensor(_RAW_TARGET_HAND_POSE, _RAW_TARGET_JOINT_ORDER)
_TARGET_INIT_QPOS_DICT = {name: value for name, value in zip(_TARGET_JOINT_NAMES, _TARGET_INIT_QPOS.tolist())}
_TARGET_HAND_POSE_DICT = {name: value for name, value in zip(_TARGET_JOINT_NAMES, _TARGET_HAND_POSE.tolist())}
_TARGET_HAND_MASK_DICT = {name: value for name, value in zip(_TARGET_JOINT_NAMES, _TARGET_HAND_MASK.tolist())}


def _get_last_actions(env) -> torch.Tensor:
    buffer = getattr(env.action_buffer, "_circular_buffer", None)
    if buffer is None:
        return torch.zeros(env.num_envs, env.num_actions, device=env.device)
    return buffer.buffer[:, -1, :]


def _normalize_joint_name(name: str) -> str:
    return name.split(":")[-1] if ":" in name else name


def _get_target_specs(env) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw target specs from config when provided, otherwise fall back to defaults."""
    raw_order = getattr(env, "_reward_target_joint_order", None) or _RAW_TARGET_JOINT_ORDER
    raw_init_qpos = getattr(env, "_reward_init_qpos", None)
    raw_hand_mask = getattr(env, "_reward_hand_mask", None)
    raw_hand_pose = getattr(env, "_reward_hand_pose", None)

    init_tensor = (
        torch.tensor(raw_init_qpos, dtype=torch.float32) if raw_init_qpos is not None else _RAW_TARGET_INIT_QPOS
    )
    mask_tensor = (
        torch.tensor(raw_hand_mask, dtype=torch.float32) if raw_hand_mask is not None else _RAW_TARGET_HAND_MASK
    )
    pose_tensor = torch.tensor(raw_hand_pose, dtype=torch.float32) if raw_hand_pose is not None else _RAW_TARGET_HAND_POSE
    return tuple(raw_order), init_tensor, mask_tensor, pose_tensor


def _select_reward_dofs(
    env,
    target_init_qpos: dict[str, float],
    target_hand_pose: dict[str, float],
    target_hand_mask: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    joint_names_raw = getattr(env.robot.data, "joint_names", None)
    if joint_names_raw is None:
        raise RuntimeError("Robot articulation does not expose joint_names.")
    joint_pos_full = env.robot.data.joint_pos
    device = joint_pos_full.device
    dtype = joint_pos_full.dtype

    selected = []
    target_values = []
    target_pose_values = []
    mask_values = []
    selected_names = []

    for idx, name in enumerate(joint_names_raw):
        normalized = _normalize_joint_name(name)
        target_value = target_init_qpos.get(normalized)
        if target_value is None:
            continue
        selected.append(joint_pos_full[:, idx])
        target_values.append(target_value)
        target_pose_values.append(target_hand_pose.get(normalized, 0.0))
        mask_values.append(target_hand_mask.get(normalized, 0.0))
        selected_names.append(normalized)

    if not selected:
        raise RuntimeError("Failed to match any reward DOFs to joint names.")

    joint_tensor = torch.stack(selected, dim=1)
    target_tensor = torch.tensor(target_values, device=device, dtype=dtype)
    target_pose_tensor = torch.tensor(target_pose_values, device=device, dtype=dtype)
    mask_tensor = torch.tensor(mask_values, device=device, dtype=dtype)
    return joint_tensor, target_tensor, target_pose_tensor, mask_tensor, selected_names


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


def _palm_world_position(env) -> torch.Tensor:
    """Return palm link world position; fall back to articulation root if unavailable."""
    body_pos = getattr(env.hand.data, "body_pos_w", None)
    if body_pos is not None:
        palm_idx = getattr(env, "_palm_body_index", None)
        if palm_idx is None:
            try:
                indices, _ = env.hand.find_bodies(name_keys=["palm"], preserve_order=True)
                if indices:
                    palm_idx = indices[0]
                    env._palm_body_index = palm_idx
            except Exception:
                palm_idx = None
        if palm_idx is not None and palm_idx < body_pos.shape[1]:
            return body_pos[:, palm_idx, :]
    return env.hand.data.root_pos_w


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
        # Lazy-load PCA axes if the tensor was not initialized (e.g., if the path existed but load failed earlier).
        pca_path = getattr(env, "_pca_axes_path", None) or getattr(getattr(env.cfg.scene, "spawn", None), "grasp_object", None)
        if pca_path and not isinstance(pca_path, str):
            try:
                pca_path = pca_path.pca_axes  # type: ignore[attr-defined]
            except Exception:
                pca_path = None
        if pca_path:
            try:
                import numpy as _np

                loaded = _np.load(pca_path)
                axes_local = torch.tensor(loaded, dtype=env.hand.data.root_pos_w.dtype, device=env.device)
                env._object_pca_axes_tensor = axes_local
            except Exception:
                axes_local = None
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
    """Mirror the original UniGraspTransformer reward shaping, but break the math down term by term.

    The computation is staged for clarity:
    1) Gather state needed for reward calculations.
    2) Compute raw (unweighted) reward components.
    3) Apply weights and aggregate the final reward/logs.
    """
    if weights is None:
        weights = RewardWeights()

    # --- Constants.
    max_finger_dist = weights.max_finger_dist
    max_hand_dist = weights.max_hand_dist
    max_goal_dist = weights.max_goal_dist
    hold_value = 2

    # --- Step 1: gather required data.
    # Hand-related data.
    hand_pos = _palm_world_position(env)
    hand_body_pos = env.hand.data.body_pos_w
    tip_indices = getattr(env, "_tip_body_ids", [])
    tip_pos = env.hand.data.body_pos_w[:, tip_indices, :] if len(tip_indices) > 0 else None
    actions = _get_last_actions(env)
    action_penalty = torch.sum(actions ** 2, dim=1)  # Quadratic action penalty.

    # Object-related data.
    object_points = _object_point_cloud_world(env)
    if object_points is None:
        raise RuntimeError("Object point cloud not available; ensure pc_fps metadata is provided.")
    obj_pos = env.obj.data.root_pos_w
    goal_height = env._default_table_surface + 0.20
    goal_pos = torch.tensor(env._default_object_pos, device=env.device, dtype=obj_pos.dtype).unsqueeze(0).repeat(env.num_envs, 1)
    goal_pos[:, 2] = goal_height
    goal_dist = torch.linalg.norm(goal_pos - obj_pos, dim=1)
    goal_hand_dist = torch.linalg.norm(goal_pos - hand_pos, dim=1)  # Hand-to-goal (airborne shaping).
    # Distances from hand surfaces to object point cloud.
    right_hand_pc_dist = _batch_sided_distance(hand_pos.unsqueeze(1), object_points).squeeze(-1)
    if tip_pos is not None and tip_pos.shape[1] > 0:
        right_hand_finger_pc_dist = torch.sum(_batch_sided_distance(tip_pos, object_points), dim=-1)
    else:
        right_hand_finger_pc_dist = torch.zeros(env.num_envs, device=env.device, dtype=hand_pos.dtype)
    joint_dist = torch.sum(_batch_sided_distance(hand_body_pos, object_points), dim=-1)
    right_hand_joint_pc_dist = joint_dist * 5.0 / hand_body_pos.shape[1]
    right_hand_body_pc_dist = joint_dist * 5.0 / hand_body_pos.shape[1]
    lowest = torch.min(object_points[:, :, -1], dim=1)[0]

    # Other data (joint states and orientation targets).
    raw_order, raw_init_qpos, raw_hand_mask, raw_hand_pose = _get_target_specs(env)
    target_init = _reorder_raw_tensor(raw_init_qpos, raw_order)
    target_hand_mask = _reorder_raw_tensor(raw_hand_mask, raw_order)
    target_hand_pose = _reorder_raw_tensor(raw_hand_pose, raw_order)
    target_init_dict = {name: value for name, value in zip(_TARGET_JOINT_NAMES, target_init.tolist())}
    target_hand_pose_dict = {name: value for name, value in zip(_TARGET_JOINT_NAMES, target_hand_pose.tolist())}
    target_hand_mask_dict = {name: value for name, value in zip(_TARGET_JOINT_NAMES, target_hand_mask.tolist())}
    joint_pos, target_init, target_hand_pose, target_hand_mask, reward_joint_names = _select_reward_dofs(
        env, target_init_dict, target_hand_pose_dict, target_hand_mask_dict
    )

    if not getattr(env, "_printed_reward_qpos_debug", False):
        try:
            current_vals = joint_pos[0].detach().cpu().tolist()
            target_vals = target_init.detach().cpu().tolist()
            print("[UniGraspReward] DOF targets:")
            for name, cur, tgt in zip(reward_joint_names, current_vals, target_vals):
                print(f"  {name}: current={cur:.6f}, target={tgt:.6f}")
        except Exception:
            pass
        env._printed_reward_qpos_debug = True
    # Distance from each joint to the reference "home" configuration; encourages reset posture.
    delta_init_qpos_value = torch.linalg.norm(joint_pos - target_init.unsqueeze(0), dim=1)

    # Deviation from the desired finger curling pose (masking out unconstrained joints).
    delta_qpos = torch.linalg.norm(
        (torch.abs(joint_pos) - target_hand_pose.unsqueeze(0))
        * target_hand_mask.unsqueeze(0)
        * (torch.abs(joint_pos) > 1.0),
        dim=1,
    )

    _, target_hand_pca_rot = _object_pca_axes_world(env)
    if target_hand_pca_rot is None:
        delta_target_hand_pca = torch.zeros(env.num_envs, device=env.device, dtype=hand_pos.dtype)
    else:
        # Angle between hand orientation and object PCA-aligned orientation.
        delta_target_hand_pca = 2.0 * torch.acos(
            torch.clamp(torch.abs(torch.sum(env.hand.data.root_quat_w * target_hand_pca_rot, dim=1)), 0.0, 1.0)
        )
    
    # --- Step 2: compute raw reward components (no outer weights applied yet).
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
    hand_up = torch.where(
        lowest >= 0.61, torch.where(hold_flag == hold_value, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up
    )
    hand_up = torch.where(
        lowest >= 0.80,
        torch.where(
            hold_flag == hold_value,
            0.2 - goal_hand_dist * 0 + weights.hand_up_goal_dist * (0.2 - goal_dist),
            hand_up,
        ),
        hand_up,
    )

    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(
        hold_flag == hold_value,
        torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus),
        bonus,
    )

    raw_init_terms: Dict[str, torch.Tensor] = {
        "delta_init_qpos_value": delta_init_qpos_value,
        "right_hand_dist": right_hand_pc_dist,
        "delta_target_hand_pca": delta_target_hand_pca,
        "right_hand_exploration_dist": right_hand_exploration_dist,
    }
    raw_grasp_terms: Dict[str, torch.Tensor] = {
        "right_hand_body_dist": right_hand_body_pc_dist,
        "right_hand_joint_dist": right_hand_joint_pc_dist,
        "right_hand_finger_dist": right_hand_finger_pc_dist,
        "right_hand_dist": 2.0 * right_hand_pc_dist,
        "goal_dist": goal_dist,
        "goal_rew": goal_rew,
        "hand_up": hand_up,
        "bonus": bonus,
        "right_hand_pose": delta_qpos,
    }

    # --- Step 3: apply weights and aggregate.
    init_terms: Dict[str, torch.Tensor] = {
        "reward/init/delta_init_qpos_value": weights.delta_init_qpos_value * raw_init_terms["delta_init_qpos_value"],
        "reward/init/right_hand_dist": weights.right_hand_dist * raw_init_terms["right_hand_dist"],
        "reward/init/delta_target_hand_pca": weights.delta_target_hand_pca * raw_init_terms["delta_target_hand_pca"],
        "reward/init/right_hand_exploration_dist": weights.right_hand_exploration_dist
        * raw_init_terms["right_hand_exploration_dist"],
    }
    init_reward = torch.zeros_like(next(iter(init_terms.values())))
    for value in init_terms.values():
        init_reward = init_reward + value

    grasp_terms: Dict[str, torch.Tensor] = {
        "reward/grasp/right_hand_body_dist": weights.right_hand_body_dist * raw_grasp_terms["right_hand_body_dist"],
        "reward/grasp/right_hand_joint_dist": weights.right_hand_joint_dist * raw_grasp_terms["right_hand_joint_dist"],
        "reward/grasp/right_hand_finger_dist": weights.right_hand_finger_dist * raw_grasp_terms["right_hand_finger_dist"],
        "reward/grasp/right_hand_dist": weights.right_hand_dist * raw_grasp_terms["right_hand_dist"],
        "reward/grasp/goal_dist": weights.goal_dist * raw_grasp_terms["goal_dist"],
        "reward/grasp/goal_rew": weights.goal_rew * raw_grasp_terms["goal_rew"],
        "reward/grasp/hand_up": weights.hand_up * raw_grasp_terms["hand_up"],
        "reward/grasp/bonus": weights.bonus * raw_grasp_terms["bonus"],
        "reward/grasp/right_hand_pose": weights.right_hand_pose * raw_grasp_terms["right_hand_pose"],
    }
    grasp_reward = torch.zeros_like(init_reward)
    for value in grasp_terms.values():
        grasp_reward = grasp_reward + value

    reward = torch.where(hold_flag != hold_value, init_reward, grasp_reward)
    reward -= weights.action_penalty * action_penalty

    logs: Dict[str, torch.Tensor] = {
        "reward/init": init_reward.detach().cpu(),
        "reward/init/total": init_reward.detach().cpu(),
        "reward/grasp": grasp_reward.detach().cpu(),
        "reward/grasp/total": grasp_reward.detach().cpu(),
    }
    for key, value in init_terms.items():
        logs[key] = value.detach().cpu()
    for key, value in grasp_terms.items():
        logs[key] = value.detach().cpu()
    logs["reward/action_penalty"] = action_penalty.detach().cpu()
    logs["debug/hold_flag"] = hold_flag.detach().cpu()
    logs["debug/goal_dist"] = goal_dist.detach().cpu()

    return reward, logs


__all__ = ["compute_hand_reward", "RewardWeights"]
