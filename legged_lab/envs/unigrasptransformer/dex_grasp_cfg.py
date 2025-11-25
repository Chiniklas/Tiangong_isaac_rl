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
    GraspObjectCfg
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
    
@configclass
class GraspRewardCfg:
    # this is the part where you implement reward configs
    pass

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
    
    # this part I don't know what use of it yet
    domain_rand: DomainRandCfg = DomainRandCfg(action_delay=ActionDelayCfg(enable=False))
    noise: NoiseCfg = NoiseCfg(add_noise=False)
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    # ============================
    sim: SimCfg = SimCfg()

@configclass
class UnigraspTransformerAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 50000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg()
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "AmpOnPolicyRunner"
    experiment_name = "walk"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk"
    wandb_project = "walk"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp parameter
    amp_reward_coef = 0.3
    amp_motion_files = ["legged_lab/envs/tienkung/datasets/motion_amp_expert/walk.txt"]
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.7
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 20
