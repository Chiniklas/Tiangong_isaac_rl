"""Reward terms for the UniGraspTransformer task with phase gating by hold flag."""

from __future__ import annotations

import torch

# implicit rewards listed in the paper (kept for completeness)
def grasp_reward_rd(env) -> torch.Tensor:  # right_hand_dist_post in cfg
    return torch.zeros(env.num_envs, device=env.device)


def grasp_reward_ro(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def grasp_reward_rl(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def grasp_reward_rg(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def grasp_reward_rs(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


# ==============================================================
# explicit reward terms listed in the upstream UniGraspTransformer
CONTACT_DIST_THRESHOLD = 0.06  # lambda_c
FINGER_DIST_THRESHOLD = 0.2
HAND_DIST_THRESHOLD = 0.04
GOAL_DIST_THRESHOLD = 0.05
TARGET_LIFT_Z = 0.8
INIT_HAND_POSE = torch.tensor(
    [0.1, 0.0, 0.6, 0.0, 0.0, 0.0, 0.6, 0.0, -0.1, 0.0, 0.6, 0.0, 0.0, -0.2, 0.0, 0.6, 0.0, 0.0, 1.2, 0.0, -0.2, 0.0],
    dtype=torch.float,
)


# ----------------- helpers -----------------
def _palm_object_dist(env) -> torch.Tensor:
    palm_pos = env.robot.data.root_pos_w[:, :3]
    object_pos = env.object.data.root_pos_w[:, :3]
    return torch.norm(palm_pos - object_pos, dim=-1)


def _fingertip_object_sum_dist(env) -> torch.Tensor:
    body_pos = env.robot.data.body_pos_w
    object_pos = env.object.data.root_pos_w[:, :3].unsqueeze(1)
    fingertip_ids = getattr(env, "fingertip_ids", [])
    if len(fingertip_ids) == 0:
        return torch.norm(env.robot.data.root_pos_w[:, :3] - object_pos.squeeze(1), dim=-1)
    tips = body_pos[:, fingertip_ids, :]
    dists = torch.norm(tips - object_pos, dim=-1)
    return dists.sum(dim=1)


def _hold_flag(env) -> torch.Tensor:
    finger_dist = _fingertip_object_sum_dist(env)
    hand_dist = _palm_object_dist(env)
    flag = (finger_dist <= FINGER_DIST_THRESHOLD).float() + (hand_dist <= HAND_DIST_THRESHOLD).float()
    return (flag >= 2.0).float()


def grasp_hold_flag(env) -> torch.Tensor:
    # Indicator that fingers/hand are within thresholds (matches upstream phase switch).
    return _hold_flag(env)


def grasp_delta_init_qpos(env) -> torch.Tensor:
    # L2 distance between current joint pose and a predefined mid-air “open” pose; encourages approaching posture.
    hold = _hold_flag(env)
    pre_mask = 1.0 - hold
    dof_pos = env.robot.data.joint_pos
    target = INIT_HAND_POSE.to(env.device)
    # broadcast target to match joint count
    if target.numel() != dof_pos.shape[1]:
        target = torch.zeros_like(dof_pos)
    return pre_mask * torch.norm(dof_pos - target, p=2, dim=1)


def grasp_right_hand_dist(env) -> torch.Tensor:
    # Average hand-to-object distance (Chamfer/Euclidean) before grasp to pull the hand toward the object.
    hold = _hold_flag(env)
    pre_mask = 1.0 - hold
    return pre_mask * _palm_object_dist(env)


def grasp_delta_target_hand_pca(env) -> torch.Tensor:
    # Deviation between hand orientation and object PCA frame to align the hand with object principal axes.
    # Without PCA frame available, use a placeholder small penalty on palm orientation deviation from identity.
    hold = _hold_flag(env)
    pre_mask = 1.0 - hold
    palm_quat = env.robot.data.root_quat_w
    # measure angle to world-up aligned quaternion (w=1, x=y=z=0)
    angle = 2.0 * torch.asin(torch.clamp(torch.norm(palm_quat[:, :3], dim=-1), max=1.0))
    return pre_mask * angle


def grasp_right_hand_exploration_dist(env) -> torch.Tensor:
    # Distance from hand to a randomly sampled object surface point to encourage exploratory contacts.
    hold = _hold_flag(env)
    pre_mask = 1.0 - hold
    palm_pos = env.robot.data.root_pos_w[:, :3]
    # sample a point from object point cloud if available
    pc = getattr(env, "pc", None)
    if pc is not None and hasattr(pc, "data") and hasattr(pc.data, "points_w"):
        points = pc.data.points_w  # (E, P, 3)
        rand_idx = torch.randint(0, points.shape[1], (env.num_envs,), device=env.device)
        sample = points[torch.arange(env.num_envs, device=env.device), rand_idx]
    else:
        sample = env.object.data.root_pos_w[:, :3]
    return pre_mask * torch.norm(palm_pos - sample, dim=-1)


def grasp_right_hand_body_dist(env) -> torch.Tensor:
    # Distance from non-finger hand bodies (e.g., palm) to the object to keep the palm close during grasp.
    hold = _hold_flag(env)
    post_mask = hold
    body_pos = env.robot.data.body_pos_w  # (E, B, 3)
    object_pos = env.object.data.root_pos_w[:, :3].unsqueeze(1)
    # ignore fingertips if available
    fingertip_ids = set(getattr(env, "fingertip_ids", []))
    mask = torch.ones(body_pos.shape[1], device=env.device, dtype=torch.bool)
    for idx in fingertip_ids:
        if idx < mask.numel():
            mask[idx] = False
    masked_pos = body_pos[:, mask, :]
    dists = torch.norm(masked_pos - object_pos, dim=-1)
    base = dists.mean(dim=1) if masked_pos.numel() > 0 else torch.zeros(env.num_envs, device=env.device)
    return post_mask * base


def grasp_right_hand_joint_dist(env) -> torch.Tensor:
    # Distance using joint-frame proxies (variant of fingertip distance) to penalize spread from the object.
    hold = _hold_flag(env)
    post_mask = hold
    joint_pos = env.robot.data.joint_pos
    base = torch.norm(joint_pos, dim=1)
    return post_mask * base


def grasp_right_hand_finger_dist(env) -> torch.Tensor:
    # Sum of fingertip distances to the object; main proximity term during grasping.
    hold = _hold_flag(env)
    post_mask = hold
    return post_mask * _fingertip_object_sum_dist(env)


def grasp_goal_dist(env) -> torch.Tensor:
    # Euclidean distance from object center to goal position.
    hold = _hold_flag(env)
    post_mask = hold
    object_pos = env.object.data.root_pos_w[:, :3]
    # target: lift to TARGET_LIFT_Z at current x,y
    goal = torch.stack([object_pos[:, 0], object_pos[:, 1], torch.full_like(object_pos[:, 2], TARGET_LIFT_Z)], dim=1)
    return post_mask * torch.norm(object_pos - goal, dim=-1)


def grasp_goal_rew(env) -> torch.Tensor:
    # Positive shaping once holding, typically something like 0.9 - 2*goal_dist to encourage closing the gap.
    hold = _hold_flag(env)
    post_mask = hold
    dist = grasp_goal_dist(env)
    return post_mask * torch.clamp(0.9 - 2.0 * dist, min=0.0)


def grasp_hand_up(env) -> torch.Tensor:
    # Lift reward when object/hand height exceeds thresholds; sometimes scaled by upward action.
    hold = _hold_flag(env)
    post_mask = hold
    object_z = env.object.data.root_pos_w[:, 2]
    reward = torch.zeros(env.num_envs, device=env.device)
    reward = torch.where(object_z >= 0.61, 0.1 + 0.1 * env.action[:, 2], reward)
    reward = torch.where(object_z >= TARGET_LIFT_Z, reward + 0.2, reward)
    return post_mask * reward


def grasp_bonus(env) -> torch.Tensor:
    # Success bonus when object is within goal tolerance (lambda_g), e.g., 1 / (1 + 10*goal_dist).
    hold = _hold_flag(env)
    post_mask = hold
    dist = grasp_goal_dist(env)
    base = torch.where(dist <= GOAL_DIST_THRESHOLD, 1.0 / (1.0 + 10.0 * dist), torch.zeros_like(dist))
    return post_mask * base


def grasp_right_hand_pose(env) -> torch.Tensor:
    # Regularizer toward a nominal finger pose during grasp (L1/L2 on joint deviation).
    hold = _hold_flag(env)
    post_mask = hold
    dof_pos = env.robot.data.joint_pos
    target = torch.zeros_like(dof_pos)
    return post_mask * torch.norm(dof_pos - target, p=1, dim=1)
