import math
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg

# @configclass
# class RobotCfg:
#     actor_obs_history_length: int = 10
#     critic_obs_history_length: int = 10
#     action_scale: float = 0.25
#     terminate_contacts_body_names: list = []
#     feet_body_names: list = []

@configclass
class DexHandCfg:
    # this config is for actor critic calculation
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []


@configclass
class MySceneCfg:
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    table: ArticulationCfg = MISSING
    grasp_object: ArticulationCfg = MISSING


@configclass
class TableCfg:
    # this config is for table spawning
    enable: bool = True
    size: tuple = (0.6, 0.6, 0.03)
    pos: tuple = (0.0, 0.0, 0.25)
    rot_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    friction: float = 1.0
    restitution: float = 0.0

@configclass
class GraspObjectCfg:
    # this config is for grasp object spawning
    enable: bool = True
    default_dir: str | None = None  # directory to sample objects when object_path is empty
    object_path: str | None = None  # explicit USD 
    size: tuple = (0.1, 0.1, 0.1)
    pos: tuple = (0.0, 0.0, 0.5)
    rot_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    object_init: dict | None = None
    metadata_path: str | None = None

    # point cloud related features
    show_point_cloud: bool = False
    pc_fps_path: str| None = None
    
    # pca axes related features
    show_pca_axes: bool = False
    pca_axes_path: str | None = None

@configclass
class GraspObsScaleCfg:
    pass
    
