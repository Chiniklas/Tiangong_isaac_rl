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

# -- empty reward config (we compute rewards in the env) --
@configclass
class InspireHandGraspRewardCfg(RewardCfg):
    pass

# -- empty event config: disables all randomization terms that expect legged names --
@configclass
class InspireHandEventCfg:
    pass

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

    # NEW: optional assets (None means “don’t spawn”)
    table: RigidObjectCfg | None = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6), metallic=0.0, roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.50, 0.0, 0.70), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    grasp_object: RigidObjectCfg | None = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=3.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3), metallic=0.2, roughness=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.73), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

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
