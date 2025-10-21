from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
import torch
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
import isaaclab.sim as sim_utils
from legged_lab.assets.handright9253.inspirehand import INSPIRE_HAND_CFG  # <-- your asset cfg
from legged_lab.envs.inspirehand.spawn_cfg import InspireHandSpawnCfg
from legged_lab.utils.env_utils.scene_grasp import SceneCfg as GraspSceneCfg

from .reward_cfg import InspireHandRewardCfg

# -- empty reward config (we compute rewards in the env) --
@configclass
class InspireHandGraspRewardCfg(RewardCfg):
    pass

# -- empty event config: disables all randomization terms that expect legged names --
@configclass
class InspireHandEventCfg:
    pass


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
    reward_scales: InspireHandRewardCfg = InspireHandRewardCfg()
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


def compute_reward(env):
    reward_cfg = env.cfg.reward_scales
    logs: dict[str, torch.Tensor] = {}

    last_joint_action = env.action_buffer._circular_buffer.buffer[:, -1, : env._joint_action_dim]
    r_smooth = reward_cfg.smooth * (last_joint_action**2).sum(dim=1)
    logs["reward/smooth"] = r_smooth.detach().cpu()

    if env.obj is None:
        return r_smooth, logs

    hand_pos = _palm_pos(env)
    obj_pos = env.obj.data.root_pos_w

    dist = torch.linalg.norm(obj_pos - hand_pos, dim=1)
    r_reach = reward_cfg.reach * torch.exp(-reward_cfg.reach_exponent * dist)
    logs["reward/reach"] = r_reach.detach().cpu()

    if env.table is not None:
        table_z = env.table.data.root_pos_w[:, 2] + env._table_thickness * 0.5
        object_bottom = obj_pos[:, 2] + env._current_lowest
        lifted = object_bottom > (table_z + reward_cfg.lift_height_buffer)
    else:
        lifted = obj_pos[:, 2] > reward_cfg.ground_lift_height
    r_lift = reward_cfg.lift * lifted.float()
    logs["reward/lift"] = r_lift.detach().cpu()

    env._hold_counter = torch.where(lifted, env._hold_counter + 1, torch.zeros_like(env._hold_counter))
    sustained = env._hold_counter > reward_cfg.hold_duration
    r_hold = reward_cfg.hold * sustained.float()
    logs["reward/hold"] = r_hold.detach().cpu()

    reward = r_smooth + r_reach + r_lift + r_hold

    if env._latest_aff_sdf is not None:
        aff_mean = torch.abs(env._latest_aff_sdf).mean(dim=1)
        reward = reward + reward_cfg.affordance_sdf * torch.exp(-reward_cfg.affordance_sdf_decay * aff_mean)
        logs["grasp/aff_sdf_mean"] = aff_mean.detach().cpu()

    if env._latest_non_sdf is not None:
        non_penalty = torch.relu(-env._latest_non_sdf).mean(dim=1)
        reward = reward - reward_cfg.non_affordance_sdf * non_penalty
        logs["grasp/non_aff_penalty"] = non_penalty.detach().cpu()

    return reward, logs


def _palm_pos(env):
    tf = getattr(env.hand.data, "link_tf_w", None)
    if tf is not None and tf.ndim == 3 and tf.shape[1] > 0:
        return tf[:, 0, :3]
    return env.hand.data.root_pos_w

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
