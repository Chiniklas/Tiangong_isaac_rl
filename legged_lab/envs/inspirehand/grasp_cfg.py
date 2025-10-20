from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
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
)
from legged_lab.envs.base.base_env_config import BaseAgentCfg
from legged_lab.assets.handright9253.inspirehand import INSPIRE_HAND_CFG  # <-- your asset cfg
from legged_lab.utils.env_utils.scene_grasp import SceneCfg as GraspSceneCfg
import isaaclab.sim as sim_utils
from legged_lab.envs.inspirehand.spawn_cfg import InspireHandSpawnCfg

# -- empty reward config (we compute rewards in the env) --
@configclass
class InspireHandGraspRewardCfg(RewardCfg):
    pass

# -- empty event config: disables all randomization terms that expect legged names --
@configclass
class InspireHandEventCfg:
    pass


@configclass
class InspireHandRewardScales:
    reach: float = 1.0
    lift: float = 1.0
    hold: float = 0.5
    smooth: float = -0.001
    affordance_sdf: float = 0.3
    non_affordance_sdf: float = 0.4
    lift_height_buffer: float = 0.05
    ground_lift_height: float = 0.75


@configclass
class InspireHandResetCfg:
    max_lateral_distance: float = 0.5
    max_vertical_offset: float = 0.5


@configclass
class InspireHandGraspSceneCfg(BaseSceneCfg):
    scene_cfg_cls: type = GraspSceneCfg
    seed: int = 42
    max_episode_length_s: float = 8.0
    num_envs: int = 4
    env_spacing: float = 2.0
    terrain_type: str = "plane"
    terrain_generator = None
    robot = INSPIRE_HAND_CFG
    spawn: InspireHandSpawnCfg = InspireHandSpawnCfg()
    table: RigidObjectCfg | None = None
    grasp_object: RigidObjectCfg | None = None

    def __post_init__(self):
        try:
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:
            pass

        spawn_cfg = self.spawn

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

        if spawn_cfg.grasp_object.enable:
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

@configclass
class InspireHandGraspEnvCfg:
    device: str = "cuda:0"
    scene: InspireHandGraspSceneCfg = InspireHandGraspSceneCfg()
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=1,
        critic_obs_history_length=1,
        action_scale=0.25,
        terminate_contacts_body_names=[],  # none for now
        feet_body_names=[],                # none for now
    )
    reward: InspireHandGraspRewardCfg = InspireHandGraspRewardCfg()
    reward_scales: InspireHandRewardScales = InspireHandRewardScales()
    reset_cfg: InspireHandResetCfg = InspireHandResetCfg()
    normalization: NormalizationCfg = NormalizationCfg()
    noise: NoiseCfg = NoiseCfg(add_noise=False)
    # Disable all events and action delay for now
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=InspireHandEventCfg(),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 1, "min_delay": 0}),
    )
    # Commands are required by BaseEnv; keep neutral
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

@configclass
class InspireHandGraspAgentCfg(BaseAgentCfg):
    # light tweaks for a hand grasp task
    num_steps_per_env = 32
    max_iterations = 40000
    runner_class_name = "OnPolicyRunner"
    experiment_name = "inspirehand_grasp"
    run_name = ""
    logger = "tensorboard"   # or "wandb" if you prefer
    # keep PPO defaults from BaseAgentCfg, or customize here as needed
