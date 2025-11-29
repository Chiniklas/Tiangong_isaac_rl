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
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import math
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
# from legged_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)

from legged_lab.assets.shadow_hand_with_fingertip.shadow_hand import SHADOW_HAND_CFG
from legged_lab.envs.base.my_confg import(
    DexHandCfg,
    MySceneCfg,
    TableCfg,
    GraspObjectCfg,
    GraspObjectGoalCfg,
    GraspObsScaleCfg,
    UnigrasptransformerNormalizationCfg
)

from legged_lab.envs.unigrasptransformer.helpers import(
    _load_yaml_cfg,
    _build_table_spawn,
    _build_hand_spawn,
    _pick_random_object_from_dir,
    _build_object_spawn,
    _build_table_cfg,
    _build_grasp_object_cfg,
    _build_hand_cfg,
)
# unpack hyperparameters from yaml files
SPAWN_CFG = _load_yaml_cfg("spawn_cfg.yaml")
WEIGHTS_CFG = _load_yaml_cfg("weights_cfg.yaml")
PPO_CFG = _load_yaml_cfg("ppo_cfg.yaml")

# Validate required reward weights exist in the YAML; surface a clear error early.
REQUIRED_WEIGHT_KEYS = [
    "hold_flag",
    "delta_init_qpos_value",
    "right_hand_dist",
    "delta_target_hand_pca",
    "right_hand_exploration_dist",
    "right_hand_body_dist",
    "right_hand_joint_dist",
    "right_hand_finger_dist",
    "right_hand_dist_post",
    "goal_dist",
    "goal_rew",
    "hand_up",
    "bonus",
    "right_hand_pose",
    "rd",
    "ro",
    "rl",
    "rg",
    "rs",
]
missing_weights = [key for key in REQUIRED_WEIGHT_KEYS if key not in WEIGHTS_CFG]
if missing_weights:
    raise KeyError(f"weights_cfg.yaml is missing required keys: {missing_weights}")

# isolate hyperparameters for three items from spawn_cfg
DEFAULT_TABLE_SPAWN = _build_table_spawn(SPAWN_CFG)
DEFAULT_OBJECT_SPAWN = _build_object_spawn(SPAWN_CFG)
DEFAULT_HAND_SPAWN = _build_hand_spawn(SPAWN_CFG)

# print("DEFAULT_TABLE_SPAWN separation successful")
# print(DEFAULT_TABLE_SPAWN)
# print("DEFAULT_OBJECT_SPAWN separation successful")
# print(DEFAULT_OBJECT_SPAWN)
# input()
#===============================================================
# build table, grasp and hand config from hyperparameters
TABLE_CFG = _build_table_cfg(DEFAULT_TABLE_SPAWN)
# print("TABLE_CFG construction successful")
# print(TABLE_CFG)

GRASP_OBJECT_CFG = _build_grasp_object_cfg(DEFAULT_OBJECT_SPAWN)
# print("GRASP_OBJECT_CFG construction successful")
# print(GRASP_OBJECT_CFG)

# override hand cfg init state with DEFAULT_HAND_SPAWN
HAND_CFG = _build_hand_cfg(DEFAULT_HAND_SPAWN, SHADOW_HAND_CFG)

@configclass
class UnigraspTransformerSceneCfg(MySceneCfg):
    """This is where you instantiate your custom scene cfg"""
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    seed: int = 42 # at training runtime, the scene seed comes from the agent

    # must have robot, table, grasp_object
    robot: ArticulationCfg = HAND_CFG
    table: TableCfg = TABLE_CFG
    grasp_object: GraspObjectCfg = GRASP_OBJECT_CFG
    object_goal: GraspObjectGoalCfg = GraspObjectGoalCfg()
    
@configclass
class GraspRewardCfg:
    """Placeholder for reward terms broken out by phase; fill weights/params when ready."""

    # pre-hold phase init_reward
    hold_flag = RewTerm(func=mdp.grasp_hold_flag, weight=WEIGHTS_CFG["hold_flag"])
    delta_init_qpos_value = RewTerm(func=mdp.grasp_delta_init_qpos, weight=WEIGHTS_CFG["delta_init_qpos_value"])
    right_hand_dist = RewTerm(func=mdp.grasp_right_hand_dist, weight=WEIGHTS_CFG["right_hand_dist"])
    delta_target_hand_pca = RewTerm(func=mdp.grasp_delta_target_hand_pca, weight=WEIGHTS_CFG["delta_target_hand_pca"])
    right_hand_exploration_dist = RewTerm(func=mdp.grasp_right_hand_exploration_dist, weight=WEIGHTS_CFG["right_hand_exploration_dist"])

    # post-hold phase grasp_reward
    right_hand_body_dist = RewTerm(func=mdp.grasp_right_hand_body_dist, weight=WEIGHTS_CFG["right_hand_body_dist"])
    right_hand_joint_dist = RewTerm(func=mdp.grasp_right_hand_joint_dist, weight=WEIGHTS_CFG["right_hand_joint_dist"])
    right_hand_finger_dist = RewTerm(func=mdp.grasp_right_hand_finger_dist, weight=WEIGHTS_CFG["right_hand_finger_dist"])
    right_hand_dist_post = RewTerm(func=mdp.grasp_reward_rd, weight=WEIGHTS_CFG["right_hand_dist_post"])
    goal_dist = RewTerm(func=mdp.grasp_goal_dist, weight=WEIGHTS_CFG["goal_dist"])
    goal_rew = RewTerm(func=mdp.grasp_goal_rew, weight=WEIGHTS_CFG["goal_rew"])
    hand_up = RewTerm(func=mdp.grasp_hand_up, weight=WEIGHTS_CFG["hand_up"])
    bonus = RewTerm(func=mdp.grasp_bonus, weight=WEIGHTS_CFG["bonus"])
    right_hand_pose = RewTerm(func=mdp.grasp_right_hand_pose, weight=WEIGHTS_CFG["right_hand_pose"])

    # paper placeholders (keep explicit)
    # NOTE: currently not used
    rd = RewTerm(func=mdp.grasp_reward_rd, weight=WEIGHTS_CFG["rd"])
    ro = RewTerm(func=mdp.grasp_reward_ro, weight=WEIGHTS_CFG["ro"])
    rl = RewTerm(func=mdp.grasp_reward_rl, weight=WEIGHTS_CFG["rl"])
    rg = RewTerm(func=mdp.grasp_reward_rg, weight=WEIGHTS_CFG["rg"])
    rs = RewTerm(func=mdp.grasp_reward_rs, weight=WEIGHTS_CFG["rs"])

@configclass
class UnigraspTransformerGraspEnv:
    device: str = "cuda:0"
    scene: UnigraspTransformerSceneCfg = UnigraspTransformerSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=HAND_CFG, # this robot is for scene generation
        table=TABLE_CFG,
        grasp_object=GRASP_OBJECT_CFG,
    )
    robot: DexHandCfg = DexHandCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=[],
    ) # this robot is for actor critic calculation

    reward = GraspRewardCfg()
    
    # TODO: to be finetuned, there are richer options in the original unigrasptransformer
    # domain randomization adds noise and delay to action 
    domain_rand: DomainRandCfg = DomainRandCfg(action_delay=ActionDelayCfg(enable=False))
    noise: NoiseCfg = NoiseCfg(add_noise=False)
    # normalization clips observation and add noise to obs if enabled
    normalization: UnigrasptransformerNormalizationCfg = UnigrasptransformerNormalizationCfg(
        obs_scales=GraspObsScaleCfg(),
    )
    # ============================
    sim: SimCfg = SimCfg()

@configclass
class UnigraspTransformerAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 8  # match original UniGrasp PPO nsteps
    max_iterations = 10000
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.8,
        noise_std_type="scalar",
        actor_hidden_dims=[1024, 1024, 512, 512],
        critic_hidden_dims=[1024, 1024, 512, 512],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.96,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )

    clip_actions = 1.0
    save_interval = 1000
    runner_class_name = "OnPolicyRunner"
    experiment_name = "unigrasptransformer_recreate"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "unigrasptransformer_recreate"
    wandb_project = "unigrasptransformer_recreate"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
