from pathlib import Path

from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
import torch
import yaml
from legged_lab.envs.base.base_config import (
    BaseSceneCfg,
    RobotCfg,
    NormalizationCfg,
    NoiseCfg,
    DomainRandCfg,
    SimCfg,
    RewardCfg,
    CommandsCfg,
    CommandRangesCfg,
    ActionDelayCfg,
    EventCfg,
)
from legged_lab.envs.base.base_env_config import BaseAgentCfg
import isaaclab.sim as sim_utils
from .spawn_cfg import GraspXLSpawnCfg, load_spawn_from_yaml
from legged_lab.assets.handright9253.inspirehand import INSPIRE_HAND_CFG
from legged_lab.utils.env_utils.scene_grasp import SceneCfg as GraspSceneCfg
from .logging_utils import log_debug

_REWARD_CFG_PATH = Path(__file__).with_name("reward_cfg.yaml")
if not _REWARD_CFG_PATH.exists():
    raise FileNotFoundError(f"Reward configuration YAML not found: {_REWARD_CFG_PATH}")
with _REWARD_CFG_PATH.open("r", encoding="utf-8") as _f:
    _REWARD_DEFAULTS = yaml.safe_load(_f) or {}
EXPECTED_KEYS = {
    "reach",
    "reach_exponent",
    "lift",
    "lift_height_buffer",
    "ground_lift_height",
    "hold",
    "hold_duration",
    "center",
    "heading",
    "wrist_align",
    "smooth",
    "affordance_sdf",
    "affordance_sdf_decay",
    "non_affordance_sdf",
    "wrist_lin_vel",
    "wrist_ang_vel",
    "obj_lin_vel",
    "obj_ang_vel",
}
missing = EXPECTED_KEYS.difference(_REWARD_DEFAULTS.keys())
if missing:
    raise KeyError(f"Missing keys in reward configuration YAML {_REWARD_CFG_PATH}: {sorted(missing)}")


@configclass
class GraspXLRewardCfg:
    reach: float = float(_REWARD_DEFAULTS["reach"])
    reach_exponent: float = float(_REWARD_DEFAULTS["reach_exponent"])
    lift: float = float(_REWARD_DEFAULTS["lift"])
    lift_height_buffer: float = float(_REWARD_DEFAULTS["lift_height_buffer"])
    ground_lift_height: float = float(_REWARD_DEFAULTS["ground_lift_height"])
    hold: float = float(_REWARD_DEFAULTS["hold"])
    hold_duration: int = int(_REWARD_DEFAULTS["hold_duration"])
    center: float = float(_REWARD_DEFAULTS["center"])
    heading: float = float(_REWARD_DEFAULTS["heading"])
    wrist_align: float = float(_REWARD_DEFAULTS["wrist_align"])
    smooth: float = float(_REWARD_DEFAULTS["smooth"])
    affordance_sdf: float = float(_REWARD_DEFAULTS["affordance_sdf"])
    affordance_sdf_decay: float = float(_REWARD_DEFAULTS["affordance_sdf_decay"])
    non_affordance_sdf: float = float(_REWARD_DEFAULTS["non_affordance_sdf"])
    wrist_lin_vel: float = float(_REWARD_DEFAULTS["wrist_lin_vel"])
    wrist_ang_vel: float = float(_REWARD_DEFAULTS["wrist_ang_vel"])
    obj_lin_vel: float = float(_REWARD_DEFAULTS["obj_lin_vel"])
    obj_ang_vel: float = float(_REWARD_DEFAULTS["obj_ang_vel"])


@configclass
class GraspXLResetCfg:
    max_lateral_distance: float = 1.0
    max_vertical_offset: float = 1.0


@configclass
class GraspXLEventCfg(EventCfg):
    physics_material = None
    add_base_mass = None
    reset_base = None
    reset_robot_joints = None
    push_robot = None


@configclass
class GraspXLGraspSceneCfg(BaseSceneCfg):
    scene_cfg_cls: type = GraspSceneCfg
    seed: int = 42
    max_episode_length_s: float = 20.0
    num_envs: int = 4
    env_spacing: float = 2.0
    terrain_type: str = "plane"
    terrain_generator = None
    robot = INSPIRE_HAND_CFG
    spawn: GraspXLSpawnCfg = GraspXLSpawnCfg()
    table: RigidObjectCfg | None = None
    grasp_object: RigidObjectCfg | None = None

    def __post_init__(self):
        try:
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:
            pass

        spawn_cfg = self.spawn

        override_object_info = load_spawn_from_yaml(spawn_cfg)

        if spawn_cfg.table.enable:
            table_spawn = sim_utils.CuboidCfg(
                size=spawn_cfg.table.size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=spawn_cfg.table.disable_gravity,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=spawn_cfg.table.color,
                    metallic=spawn_cfg.table.metallic,
                    roughness=spawn_cfg.table.roughness,
                ),
            )
            self.table = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=table_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=spawn_cfg.table.pos,
                    rot=spawn_cfg.table.rot,
                ),
            )
        else:
            self.table = None

        static_usd = getattr(spawn_cfg.grasp_object, "static_usd", None)
        if spawn_cfg.grasp_object.enable and static_usd is not None:
            usd_spawn = sim_utils.UsdFileCfg(
                usd_path=static_usd,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=spawn_cfg.grasp_object.disable_gravity,
                    max_depenetration_velocity=3.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            self.grasp_object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=usd_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=spawn_cfg.grasp_object.pos,
                    rot=spawn_cfg.grasp_object.rot,
                ),
            )
        elif spawn_cfg.grasp_object.enable:
            # fallback to simple cuboid if no static USD was provided
            object_spawn = sim_utils.CuboidCfg(
                size=spawn_cfg.grasp_object.size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=spawn_cfg.grasp_object.disable_gravity,
                    max_depenetration_velocity=3.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=spawn_cfg.grasp_object.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=spawn_cfg.grasp_object.color,
                    metallic=spawn_cfg.grasp_object.metallic,
                    roughness=spawn_cfg.grasp_object.roughness,
                ),
            )
            self.grasp_object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=object_spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=spawn_cfg.grasp_object.pos,
                    rot=spawn_cfg.grasp_object.rot,
                ),
            )
        else:
            self.grasp_object = None

        log_debug(
            "GraspXLGraspSceneCfg ready (object=%s)"
            % getattr(self.spawn.grasp_object, "object_id", None)
        )


@configclass
class GraspXLEnvCfg:
    device: str = "cuda:0"
    scene: GraspXLGraspSceneCfg = GraspXLGraspSceneCfg()
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=1,
        critic_obs_history_length=1,
        action_scale=0.25,
        terminate_contacts_body_names=[],
        feet_body_names=[],
    )
    reward: RewardCfg = RewardCfg()
    reward_scales: GraspXLRewardCfg = GraspXLRewardCfg()
    reset_cfg: GraspXLResetCfg = GraspXLResetCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    noise: NoiseCfg = NoiseCfg(add_noise=False)
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=GraspXLEventCfg(),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 1, "min_delay": 0}),
    )
    commands: CommandsCfg = CommandsCfg(
        heading_command=False,
        debug_vis=False,
        resampling_time_range=(8.0, 8.0),
        ranges=CommandRangesCfg(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )
    sim: SimCfg = SimCfg(dt=1 / 120.0, decimation=2)

    def __post_init__(self):
        log_debug(f"GraspXLEnvCfg ready (device={self.device})")


@configclass
class GraspXLAgentCfg(BaseAgentCfg):
    num_steps_per_env = 150
    max_iterations = 10000
    runner_class_name = "OnPolicyRunner"
    experiment_name = "graspxl_grasp"
    run_name = ""
    logger = "tensorboard"

    def __post_init__(self):
        log_debug(
            "GraspXLAgentCfg ready (steps_per_env=%d, max_iterations=%d)"
            % (self.num_steps_per_env, self.max_iterations)
        )
