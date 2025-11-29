import math
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.envs.base.base_config import ActionDelayCfg

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
    use_relative_control: bool = False
    dof_speed_scale: float = 20.0
    transition_scale: float = 0.5
    orientation_scale: float = 0.1
    actions_moving_average: float | None = 1.0
    terminate_contacts_body_names: list = []
    asymmetric_obs: bool = False


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
class GraspObjectGoalCfg:
    # Placeholder config for a goal marker actor (e.g., small sphere above the object).
    enable: bool = True  # upstream always spawns the goal marker
    radius: float = 0.01
    # Upstream only uses a relative offset from the object start pose.
    displacement: tuple = (0.0, 0.0, 0.2)
    rot_xyzw: tuple = (0.0, 0.0, 0.0, 1.0)
    # Visual hints if/when a goal actor is spawned.
    color: tuple = (173 / 255, 90 / 255, 90 / 255)  # upstream reddish goal marker

    
@configclass
class GraspObsScaleCfg:
    # Per-field scales (mirror upstream placeholders).
    lin_vel: float = 1.0
    ang_vel: float = 0.2
    projected_gravity: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 0.2
    joint_torque: float = 10.0
    fingertip_force: float = 10.0
    fingertip_torque: float = 10.0

@configclass
class UnigrasptransformerNormalizationCfg:
    obs_scales: GraspObsScaleCfg = GraspObsScaleCfg()
    clip_observations: float | None = None
    clip_actions: float | None = None

@configclass
class myEventCfg:
    # Placeholder event terms; override with real funcs/modes/params when enabling DR.
    physics_material = EventTerm(
        func=None,
        mode=None,
        params={},
    )
    add_base_mass = EventTerm(
        func=None,
        mode=None,
        params={},
    )
    reset_base = EventTerm(
        func=None,
        mode=None,
        params={},
    )
    reset_robot_joints = EventTerm(
        func=None,
        mode=None,
        params={},
    )
    push_robot = EventTerm(
        func=None,
        mode=None,
        params={},
    )

@configclass
class myDomainRandCfg:
    # this is a place holder for my custom domain randomization cfg
    events: myEventCfg | None = None
    action_delay: ActionDelayCfg = ActionDelayCfg(enable=False)
