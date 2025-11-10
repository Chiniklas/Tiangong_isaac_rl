from pathlib import Path

from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

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
from legged_lab.assets.handright9253.inspirehand import INSPIRE_HAND_CFG
from legged_lab.utils.env_utils.scene_grasp import SceneCfg as GraspSceneCfg

from .spawn_cfg import UniGraspTransformerSpawnCfg, load_spawn_from_yaml
from .logging_utils import log_debug


@configclass
class UniGraspTransformerRewardCfg:
    """Placeholder for reward scale parameters specific to UniGraspTransformer."""
    # Define task-specific reward scales here if needed.
    pass


@configclass
class UniGraspTransformerResetCfg:
    max_lateral_distance: float = 1.0
    max_vertical_offset: float = 1.0


@configclass
class UniGraspTransformerEventCfg(EventCfg):
    physics_material = None
    add_base_mass = None
    reset_base = None
    reset_robot_joints = None
    push_robot = None


@configclass
class UniGraspTransformerGraspSceneCfg(BaseSceneCfg):
    scene_cfg_cls: type = GraspSceneCfg
    seed: int = 42
    max_episode_length_s: float = 20.0
    num_envs: int = 4
    env_spacing: float = 2.0
    terrain_type: str = "plane"
    terrain_generator = None
    robot = INSPIRE_HAND_CFG

    spawn: UniGraspTransformerSpawnCfg = UniGraspTransformerSpawnCfg()
    table: RigidObjectCfg | None = None
    grasp_object: RigidObjectCfg | None = None

    def __post_init__(self):
        try:
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:
            pass

        spawn_cfg = self.spawn
        # Load from YAML only if a config path exists and no runtime override was provided.
        cfg_path = getattr(spawn_cfg, "config_path", None)
        override_present = (
            getattr(spawn_cfg, "_override_object_info", None) is not None
            or getattr(spawn_cfg.grasp_object, "static_usd", None) is not None
        )
        if cfg_path and not override_present:
            path = Path(cfg_path).expanduser()
            if path.exists():
                _ = load_spawn_from_yaml(spawn_cfg)
            else:
                # Skip loading if path is missing; tests may provide overrides later.
                pass

        # Table
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

        # Grasp object (USD or simple cuboid)
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
            "UniGraspTransformerGraspSceneCfg ready (object=%s)"
            % getattr(self.spawn.grasp_object, "object_id", None)
        )


@configclass
class UniGraspTransformerEnvCfg:
    device: str = "cuda:0"
    scene: UniGraspTransformerGraspSceneCfg = UniGraspTransformerGraspSceneCfg()
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=1,
        critic_obs_history_length=1,
        action_scale=0.25,
        terminate_contacts_body_names=[],
        feet_body_names=[],
    )
    reward: RewardCfg = RewardCfg()
    reward_scales: UniGraspTransformerRewardCfg = UniGraspTransformerRewardCfg()
    reset_cfg: UniGraspTransformerResetCfg = UniGraspTransformerResetCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    noise: NoiseCfg = NoiseCfg(add_noise=False)
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=UniGraspTransformerEventCfg(),
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
        log_debug(f"UniGraspTransformerEnvCfg ready (device={self.device})")


@configclass
class UniGraspTransformerAgentCfg(BaseAgentCfg):
    num_steps_per_env = 150
    max_iterations = 10000
    runner_class_name = "OnPolicyRunner"
    experiment_name = "unigrasptransformer_grasp"
    run_name = ""
    logger = "tensorboard"

    def __post_init__(self):
        log_debug(
            "UniGraspTransformerAgentCfg ready (steps_per_env=%d, max_iterations=%d)"
            % (self.num_steps_per_env, self.max_iterations)
        )
