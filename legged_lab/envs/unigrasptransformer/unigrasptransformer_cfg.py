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
from legged_lab.assets.shadowhand import SHADOW_HAND_CFG
from copy import deepcopy
from legged_lab.utils.env_utils.scene_grasp import SceneCfg as GraspSceneCfg

from legged_lab.mdp.rewards_unigrasptransformer import RewardWeights
from .spawn_cfg import UniGraspTransformerSpawnCfg
from .logging_utils import log_debug
from .train_config import load_reward_weights_from_train_yaml


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


HAND_ASSET_REGISTRY = {
    "inspire": INSPIRE_HAND_CFG,
    "shadowhand": SHADOW_HAND_CFG,
}
FINGERTIP_PATTERNS = {
    "inspire": ["Link48", "Link4", "Link14", "Link24", "Link34"],
    "shadowhand": ["fftip", "mftip", "rftip", "lftip", "thtip"],
}


@configclass
@configclass
class UniGraspTransformerGraspSceneCfg(BaseSceneCfg):
    scene_cfg_cls: type = GraspSceneCfg
    seed: int = 42
    spawn: UniGraspTransformerSpawnCfg = UniGraspTransformerSpawnCfg()
    table: RigidObjectCfg | None = None
    grasp_object: RigidObjectCfg | None = None
    terrain_type: str = "plane"
    def __post_init__(self):
        try:
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:
            pass

        spawn_cfg = self.spawn
        hand_variant = getattr(spawn_cfg.hand, "asset_type", "inspire").lower()
        robot_cfg = deepcopy(HAND_ASSET_REGISTRY.get(hand_variant, INSPIRE_HAND_CFG))
        custom_asset_path = getattr(spawn_cfg.hand, "asset_path", None)
        if custom_asset_path:
            asset_path = Path(custom_asset_path).expanduser()
            if not asset_path.is_absolute():
                repo_root = Path(__file__).resolve().parents[3]
                candidate = (repo_root / asset_path).resolve()
                if candidate.exists():
                    asset_path = candidate
                else:
                    asset_path = (Path.cwd() / asset_path).resolve()
            asset_path = asset_path.resolve()
            if not asset_path.exists():
                raise FileNotFoundError(f"Hand asset path does not exist: {asset_path}")
            if asset_path.suffix.lower() == ".usd":
                spawn_cfg_src = robot_cfg.spawn
                robot_cfg.spawn = sim_utils.UsdFileCfg(
                    usd_path=asset_path.as_posix(),
                    articulation_props=getattr(spawn_cfg_src, "articulation_props", None),
                    mass_props=getattr(spawn_cfg_src, "mass_props", None),
                    rigid_props=getattr(spawn_cfg_src, "rigid_props", None),
                    collision_props=getattr(spawn_cfg_src, "collision_props", None),
                    activate_contact_sensors=True,
                )
            elif hasattr(robot_cfg.spawn, "asset_path"):
                robot_cfg.spawn.asset_path = asset_path.as_posix()
            elif hasattr(robot_cfg.spawn, "usd_path"):
                robot_cfg.spawn.usd_path = asset_path.as_posix()
        self.robot = robot_cfg
        log_debug(f"Selected hand asset '{hand_variant}'")
        if not getattr(spawn_cfg.hand, "fingertip_body_exprs", None):
            spawn_cfg.hand.fingertip_body_exprs = tuple(FINGERTIP_PATTERNS.get(hand_variant, FINGERTIP_PATTERNS["inspire"]))
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
        if spawn_cfg.grasp_object.enable and spawn_cfg.grasp_object.spawn_mesh and static_usd is not None:
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
        # Apply hand initial pose from YAML to robot init_state so we don't need to warp at init
        try:
            hand_pos = tuple(self.spawn.hand.pos)
            hand_rot = tuple(self.spawn.hand.orientation_xyzw)
            robot_cfg = deepcopy(self.robot)
            robot_cfg.init_state.pos = hand_pos
            robot_cfg.init_state.rot = hand_rot
            self.robot = robot_cfg
            log_debug("Applied YAML hand init pose to robot init_state")
        except Exception:
            pass


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
    reward_config_path: str = str(Path(__file__).with_name("cfg").joinpath("train.yaml"))
    ppo_config_path: str = str(Path(__file__).with_name("cfg").joinpath("ppo_config.yaml"))
    reward_weights: RewardWeights = RewardWeights()

    def __post_init__(self):
        log_debug(f"UniGraspTransformerEnvCfg ready (device={self.device})")
        try:
            path = Path(self.reward_config_path).expanduser()
            self.reward_weights = load_reward_weights_from_train_yaml(path)
        except Exception as exc:
            log_debug(f"Failed to load reward config at {self.reward_config_path}: {exc}")


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
