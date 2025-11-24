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

from legged_lab.assets.shadow_hand_with_fingertip.shadow_hand import SHADOW_HAND_CFG
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
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401

# Load YAML configs at import so hyperparameters are available module-wide.
def _load_yaml_cfg(filename: str) -> Dict[str, Any]:
    cfg_path = Path(__file__).resolve().parent / "cfg" / filename
    try:
        import yaml
    except ImportError:
        return {}
    if not cfg_path.is_file():
        return {}
    content = cfg_path.read_text(encoding="utf-8")
    if not content.strip():
        return {}
    try:
        loaded = yaml.safe_load(content)
    except Exception:
        return {}
    return loaded or {}


SPAWN_CFG = _load_yaml_cfg("spawn_cfg.yaml")
WEIGHTS_CFG = _load_yaml_cfg("weights_cfg.yaml")
PPO_CFG = _load_yaml_cfg("ppo_cfg.yaml")

def _build_table_spawn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    table_cfg = cfg.get("table", {}) if isinstance(cfg, dict) else {}
    return {
        "enable": bool(table_cfg.get("enable", False)),
        "size": table_cfg.get("size"),
        "pos": table_cfg.get("pos"),
        "rot_xyzw": table_cfg.get("rot_xyzw"),
    }


def _build_hand_spawn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hand_cfg = cfg.get("hand", {}) if isinstance(cfg, dict) else {}
    return hand_cfg if isinstance(hand_cfg, dict) else {}

def _pick_random_object_from_dir(default_dir: str) -> Dict[str, Any]:
    """Pick a random metadata.json two levels under default_dir and return its contents."""
    dir_path = Path(default_dir).expanduser()
    if not dir_path.is_dir():
        raise ValueError(f"default_dir '{default_dir}' is not a valid directory for grasp object selection")
    candidates = list(dir_path.glob("*/*/metadata.json"))
    if not candidates:
        raise ValueError(f"default_dir '{default_dir}' contains no files to sample grasp objects from")
    meta_path = random.choice(candidates)
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        raise ValueError(f"Failed to read metadata at {meta_path}: {exc}") from exc
    meta["metadata_path"] = str(meta_path)
    return meta


def _build_object_spawn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    obj_cfg = cfg.get("object", {}) if isinstance(cfg, dict) else {}
    default_dir = obj_cfg.get("default_dir")
    object_path = obj_cfg.get("object_path")
    pc_fps = obj_cfg.get("pc_fps")
    pca_axes = obj_cfg.get("pca_axes")
    object_init = obj_cfg.get("object_init")
    metadata_path = obj_cfg.get("metadata_path")

    path_missing = object_path is None or (isinstance(object_path, str) and len(object_path.strip()) == 0)
    sampled_meta: Dict[str, Any] = {}
    if path_missing:
        if default_dir is None or (isinstance(default_dir, str) and len(default_dir.strip()) == 0):
            raise ValueError("object_path is missing and no default_dir provided in spawn_cfg.yaml")
        try:
            sampled_meta = _pick_random_object_from_dir(default_dir)
            object_path = sampled_meta.get("static_usd") or sampled_meta.get("object_path")
            pc_fps = pc_fps or sampled_meta.get("pc_fps")
            pca_axes = pca_axes or sampled_meta.get("pca_axes")
            object_init = object_init or sampled_meta.get("object_init")
            metadata_path = metadata_path or sampled_meta.get("metadata_path")
        except ValueError as exc:
            print(f"[WARN] {exc}. Disabling object spawn for this run.")
            obj_cfg["enable"] = False
            obj_cfg["spawn_mesh"] = False
            object_path = None
    return {
        "enable": bool(obj_cfg.get("enable", False) and (object_path is not None or not obj_cfg.get("spawn_mesh", False))),
        "default_dir": default_dir,
        "object_path": object_path,
        "spawn_mesh": bool(obj_cfg.get("spawn_mesh", False)),
        "show_point_cloud": bool(obj_cfg.get("show_point_cloud", False)),
        "show_pca_axes": bool(obj_cfg.get("show_pca_axes", False)),
        "size": obj_cfg.get("size"),
        "pos": obj_cfg.get("pos"),
        "rot_xyzw": obj_cfg.get("rot_xyzw"),
        "object_init": object_init,
        "pc_fps": pc_fps,
        "pca_axes": pca_axes,
        "metadata_path": metadata_path,
    }

DEFAULT_TABLE_SPAWN = _build_table_spawn(SPAWN_CFG)
DEFAULT_OBJECT_SPAWN = _build_object_spawn(SPAWN_CFG)
DEFAULT_HAND_SPAWN = _build_hand_spawn(SPAWN_CFG)
#===============================================================
@configclass
class UnigraspTransformerSceneCfg(BaseSceneCfg):
    """Scene config tailored for object grasping tasks."""
    # Default to the Shadow Hand asset; can be overridden via cfg if needed.
    robot: ArticulationCfg = SHADOW_HAND_CFG
    table: Optional[ArticulationCfg] = None
    grasp_object: Optional[ArticulationCfg] = None
    table_spawn: Dict[str, Any] = DEFAULT_TABLE_SPAWN
    object_spawn: Dict[str, Any] = DEFAULT_OBJECT_SPAWN
    hand_spawn: Dict[str, Any] = DEFAULT_HAND_SPAWN


@configclass
class LiteRewardCfg:
    pass

@configclass
class UnigraspTransformerGraspEnv:
    device: str = "cuda:0"
    scene: UnigraspTransformerSceneCfg = UnigraspTransformerSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=SHADOW_HAND_CFG,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
        table=None,  # table_spawn will be used
        grasp_object=None,  # object_spawn will be used
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll.*"],
    )
    reward = LiteRewardCfg()
    sim: SimCfg = SimCfg()
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

    # disabled velocity heading commands
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(0.0, 0.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=CommandRangesCfg(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        ),
    )
    


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
