from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from isaaclab.utils.math import quat_conjugate, quat_mul

if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")
else:
    _DEVICE = torch.device("cpu")


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


def _get_last_actions(env) -> torch.Tensor:
    buffer = getattr(env.action_buffer, "_circular_buffer", None)
    if buffer is None:
        return torch.zeros(env.num_envs, env.num_actions, device=env.device)
    return buffer.buffer[:, -1, :]


def _compute_hand_pose_deltas(env) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    default_pose = env.robot.data.default_joint_pos
    joint_pos = env.robot.data.joint_pos

    delta_init_qpos_value = torch.linalg.norm(joint_pos - default_pose, dim=1)

    # Use the default rest pose as the "target" pose that mimics the PCA aligned target in the reference code.
    target_mask = (torch.abs(default_pose) > 1.0).float()
    delta_qpos = torch.linalg.norm((torch.abs(joint_pos) - torch.abs(default_pose)) * target_mask, dim=1)

    # Use wrist orientation difference as a proxy for PCA deviation.
    hand_quat_xyzw = env.robot.data.root_quat_w
    hand_quat = torch.cat((hand_quat_xyzw[:, 3:4], hand_quat_xyzw[:, :3]), dim=-1)
    default_quat_xyzw = env._default_hand_state[:, 3:7]
    default_quat = torch.cat((default_quat_xyzw[:, 3:4], default_quat_xyzw[:, :3]), dim=-1)
    delta_quat = quat_mul(quat_conjugate(default_quat), hand_quat)
    delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.linalg.norm(delta_quat[:, 1:], dim=1), max=1.0))

    return delta_init_qpos_value, delta_qpos, delta_hand_rot_value


def _compute_hand_object_metrics(env) -> Dict[str, torch.Tensor]:
    obj_pos = env.obj.data.root_pos_w
    hand_pos = env.hand.data.root_pos_w
    tip_indices = getattr(env, "_tip_body_ids", [])
    tip_pos = env.hand.data.body_pos_w[:, tip_indices, :] if len(tip_indices) > 0 else None

    right_hand_dist = torch.linalg.norm(obj_pos - hand_pos, dim=1)

    if tip_pos is not None:
        tip_dists = torch.linalg.norm(tip_pos - obj_pos.unsqueeze(1), dim=2)
        right_hand_finger_dist = torch.sum(tip_dists, dim=1)
        right_hand_body_dist = torch.mean(tip_dists, dim=1)
        min_fingertip = torch.min(tip_dists, dim=1).values
    else:
        right_hand_finger_dist = torch.zeros(env.num_envs, device=env.device)
        right_hand_body_dist = torch.zeros_like(right_hand_finger_dist)
        min_fingertip = torch.full((env.num_envs,), float("inf"), device=env.device)

    joint_pos = env.robot.data.joint_pos
    joint_target = env.robot.data.default_joint_pos
    right_hand_joint_dist = torch.linalg.norm(joint_pos - joint_target, dim=1)

    exploration_target = torch.tensor(env._default_object_pos, device=env.device, dtype=obj_pos.dtype)
    exploration_target[2] += 0.15
    right_hand_exploration_dist = torch.linalg.norm(hand_pos - exploration_target, dim=1)

    return {
        "right_hand_dist": right_hand_dist,
        "right_hand_finger_dist": right_hand_finger_dist,
        "right_hand_body_dist": right_hand_body_dist,
        "right_hand_joint_dist": right_hand_joint_dist,
        "right_hand_exploration_dist": right_hand_exploration_dist,
        "min_fingertip_dist": min_fingertip,
    }


def _goal_metrics(env) -> Tuple[torch.Tensor, torch.Tensor]:
    obj_pos = env.obj.data.root_pos_w
    goal_height = env._default_table_surface + 0.20
    goal_pos = torch.tensor(env._default_object_pos, device=env.device, dtype=obj_pos.dtype)
    goal_pos[2] = goal_height
    goal_pos = goal_pos.unsqueeze(0).repeat(env.num_envs, 1)

    hand_pos = env.hand.data.root_pos_w
    goal_dist = torch.linalg.norm(goal_pos - obj_pos, dim=1)
    goal_hand_dist = torch.linalg.norm(goal_pos - hand_pos, dim=1)
    return goal_dist, goal_hand_dist


def _hold_flags(metrics: Dict[str, torch.Tensor], weights: RewardWeights) -> torch.Tensor:
    finger_flag = (metrics["min_fingertip_dist"] <= weights.max_finger_dist).int()
    hand_flag = (metrics["right_hand_dist"] <= weights.max_hand_dist).int()
    return finger_flag + hand_flag


def compute_hand_reward(env, weights: RewardWeights | None = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if weights is None:
        weights = RewardWeights()

    actions = _get_last_actions(env)
    action_penalty = torch.sum(actions ** 2, dim=1)

    delta_init_qpos_value, delta_qpos, delta_hand_rot_value = _compute_hand_pose_deltas(env)
    metrics = _compute_hand_object_metrics(env)
    goal_dist, goal_hand_dist = _goal_metrics(env)
    hold_flag = _hold_flags(metrics, weights)
    hold_value = 2

    init_reward = (
        weights.delta_init_qpos_value * delta_init_qpos_value
        + weights.right_hand_dist * metrics["right_hand_dist"]
        + weights.delta_target_hand_pca * delta_hand_rot_value
        + weights.right_hand_exploration_dist * metrics["right_hand_exploration_dist"]
    )

    grasp_reward = (
        weights.right_hand_body_dist * metrics["right_hand_body_dist"]
        + weights.right_hand_joint_dist * metrics["right_hand_joint_dist"]
        + weights.right_hand_finger_dist * metrics["right_hand_finger_dist"]
        + 2.0 * weights.right_hand_dist * metrics["right_hand_dist"]
        + weights.goal_dist * goal_dist
    )

    goal_rew = torch.where(
        hold_flag == hold_value,
        weights.goal_rew * (0.9 - 1.5 * goal_dist),
        torch.zeros_like(goal_dist),
    )
    grasp_reward += goal_rew

    hand_up = torch.where(
        metrics["right_hand_dist"] < weights.max_hand_dist,
        weights.hand_up * torch.relu(goal_hand_dist * -1.0),
        torch.zeros_like(goal_hand_dist),
    )
    grasp_reward += hand_up

    bonus = torch.where(
        goal_dist <= weights.max_goal_dist,
        weights.bonus / (1 + 10 * goal_dist),
        torch.zeros_like(goal_dist),
    )
    grasp_reward += bonus
    grasp_reward += weights.right_hand_pose * delta_qpos

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
