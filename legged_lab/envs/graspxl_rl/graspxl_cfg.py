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
from legged_lab.assets.handright9253.inspirehand import INSPIRE_HAND_CFG
from .spawn_cfg import GraspXLSpawnCfg
from legged_lab.utils.env_utils.scene_grasp import SceneCfg as GraspSceneCfg

from .reward_cfg import GraspXLRewardCfg


@configclass
class GraspXLGraspRewardCfg(RewardCfg):
    pass


@configclass
class GraspXLEventCfg:
    pass


@configclass
class GraspXLResetCfg:
    max_lateral_distance: float = 0.2
    max_vertical_offset: float = 0.2


@configclass
class GraspXLGraspSceneCfg(BaseSceneCfg):
    scene_cfg_cls: type = GraspSceneCfg
    seed: int = 42
    max_episode_length_s: float = 8.0
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
    reward: GraspXLGraspRewardCfg = GraspXLGraspRewardCfg()
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


def compute_reward(env):
    reward_cfg = env.cfg.reward_scales
    logs: dict[str, torch.Tensor] = {}

    if env.obj is None:
        reward = torch.zeros(env.num_envs, device=env.device, dtype=env.robot.data.root_pos_w.dtype)
        logs["reward/approach"] = reward.detach().cpu()
        logs["reward/lift"] = reward.detach().cpu()
        logs["reward/hold"] = reward.detach().cpu()
        return reward, logs

    hand_pos = _palm_pos(env)
    obj_pos = env.obj.data.root_pos_w

    dist = torch.linalg.norm(obj_pos - hand_pos, dim=1)
    r_reach = reward_cfg.reach * torch.exp(-reward_cfg.reach_exponent * dist)
    logs["reward/approach"] = r_reach.detach().cpu()

    # Debug: expose the first environment's hand/object positions to help diagnose flat rewards.
    # To re-enable console logging, uncomment the block below.
    #
    # if env.sim_step_counter % env.cfg.sim.decimation == 0:
    #     hand0 = hand_pos[0].detach().cpu()
    #     obj0 = obj_pos[0].detach().cpu()
    #     print(
    #         f"[DEBUG][GraspXL] step={env.sim_step_counter} "
    #         f"hand_env0={hand0.tolist()} obj_env0={obj0.tolist()}"
    #     )

    for idx, axis in enumerate("xyz"):
        logs[f"debug/hand_pos_env0_{axis}"] = hand_pos[0, idx].detach().cpu()
        logs[f"debug/object_pos_env0_{axis}"] = obj_pos[0, idx].detach().cpu()

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

    reward = r_reach + r_lift + r_hold
    return reward, logs


def _palm_pos(env):
    tf = getattr(env.hand.data, "link_tf_w", None)
    if tf is not None and tf.ndim == 3 and tf.shape[1] > 0:
        return tf[:, 0, :3]
    return env.hand.data.root_pos_w


@configclass
class GraspXLAgentCfg(BaseAgentCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    runner_class_name = "OnPolicyRunner"
    experiment_name = "graspxl_grasp"
    run_name = ""
    logger = "tensorboard"
