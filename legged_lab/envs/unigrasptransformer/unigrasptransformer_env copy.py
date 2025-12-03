"""
Observation flow from init
- __init__: builds sim/scene, grabs handles (robot, object, goal if present), sets buffers, event/reward managers, then reset.
- reset: clears buffers, applies reset events if configured, writes scene to sim, steps once.
- step: after physics, it refreshes stage-derived points and calls compute_observations.
- compute_observations: calls compute_current_observations (read sim tensors, build blocks: proprio, prev action, object state + goal delta, optional PCA, visual placeholder, optional time and hand–object dist), adds actor-only noise if enabled, pushes into history buffers, flattens history, clips if you enable it, returns actor/criti

Action flow from init
- __init__: sets action buffer/delay, scales/clips from cfg.
- reset: clears action buffer and prev_action.
- step:
    - Validate 24-D action; optional delay via DelayBuffer.
    - Clip to clip_actions, scale by action_scale, add defaults to build finger joint targets; wrist wrench stays as raw 6D.
    - Cache prev_action (wrench + processed finger targets).
    - For each substep (sim.decimation): apply wrist force/torque to palm body 0, set joint position targets, write to sim, step physics/render, update scene.
    - Post-step: increment episode length, compute rewards/termination, reset flagged envs, refresh points, compute/return obs, reward, reset flags, extras.
"""

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
import os.path as osp
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_mul, quat_conjugate

from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import UnigraspTransformerGraspEnv
from legged_lab.utils.env_utils.unigrasptransformer_scene import UniGraspSceneCfg
from legged_lab.assets.shadow_hand_with_fingertip.shadow_hand import ACTIVE_JOINTS

from rsl_rl.env import VecEnv
from legged_lab.envs.unigrasptransformer.helpers import (
    _load_yaml_cfg,
    compute_time_encoding,
    quat_from_euler_xyz,
    quat_to_euler_xyz,
    batch_sided_distance,
    get_point_cloud_world,
    get_hand_points_world
)

SPAWN_CFG = _load_yaml_cfg("spawn_cfg.yaml")
WEIGHTS_CFG = _load_yaml_cfg("weights_cfg.yaml")
PPO_CFG = _load_yaml_cfg("ppo_cfg.yaml")


class UniGraspTransformerEnv(VecEnv):
    def __init__(
        self,
        cfg: UnigraspTransformerGraspEnv,
        headless,
    ):
        # init task settings
        self.cfg: UnigraspTransformerGraspEnv = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        # create simulation
        self.sim, self.scene, ok = self.create_sim()
        self.robot = self.scene["robot"]
        self.object = self.scene["object"]
        self.goalobject = self.scene["object_goal"]

        # initialize hand points and object point cloud:
        self.object_pc, self.hand_points = None

        ### GET RELATED DATA!
        self.robotdata = self._load_shadow_hand_assets(self.robot)
        # includes per-env tensors + metadata (world frame):
        #   joint_names: List[str]
        #   body_names: List[str]
        #   fingertip_ids: []
        #   activated_joint_names: List[str] (18 active joints)
        #   hand_root_state: (E, 13) [pos, quat, linvel, angvel]
        #   hand_body_state: (E,nbody, 13) [pos, quat, linvel, angvel]
        #   hand_joint_pos: (E, Ndof)
        #   hand_joint_vel: (E, Ndof)
        #   hand_joint_force: (E, Ndof)
        #   hand_points: (E, Np, 3) if available

        self.objectdata = self._load_object_asset_info(self.object)
        # includes per-env object tensors (world frame):
        #   root_state: (E, 13) [pos, quat, linvel, angvel]
        #   pointcloud: (E, Npc, 3) if available

        self.goalpointdata = self._load_goal_point_asset_info(self.goalobject)
        # includes per-env goal tensors (world frame):
        #   root_state: (E, 13) [pos, quat, linvel, angvel]

        # init goal conditioned settings
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)
        
        # TODO: set object and hand initial pose and position (upstream places hand at a default pose,
        #       sets object to an init state, optionally rotates by a sampled prior yaw, zeroes velocities;
        #       no free-fall drop by default).

        # self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.init_buffers()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        print("INITIALIZATION SUCCESSFUL!")

    def create_sim(self):
        """Build simulation and scene; return sim, scene, and a simple success flag."""
        sim_cfg = sim_utils.SimulationCfg(
            device=self.cfg.device,
            dt=self.cfg.sim.dt,
            render_interval=self.cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=self.cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        sim = SimulationContext(sim_cfg)

        scene_cfg = UniGraspSceneCfg(config=self.cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        scene = InteractiveScene(scene_cfg)
        sim.reset()

        ok = sim is not None and scene is not None
        return sim, scene, ok

    def refresh_stage_points(self):
        """Refresh hand points and object point cloud from the USD stage (per env)."""
        hand_pts = get_hand_points_world(env_index=None)
        if isinstance(hand_pts, list):
            shapes = {p.shape for p in hand_pts}
            raise ValueError(f"Hand point counts differ across envs; shapes: {shapes}")
        self.hand_points = torch.as_tensor(hand_pts, device=self.device, dtype=torch.float32)
        print(f"Hand points loaded from stage directly: {self.hand_points.shape}")

        pc_np = get_point_cloud_world(env_index=None, prim_suffix="ObjectPC")
        if isinstance(pc_np, list):
            shapes = {p.shape for p in pc_np}
            raise ValueError(f"Point cloud counts differ across envs; shapes: {shapes}")
        self.object_points = torch.as_tensor(pc_np, device=self.device, dtype=torch.float32)
        print(f"Point cloud loaded from USD overlay: {self.object_points.shape}")

    def init_buffers(self):
        """
        buffers are basicly action buffer, obs buffer, and episode length buffer
        """
        # Per-episode extras and bookkeeping.
        self.extras = {}

        # unpack some hyperparameters
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = 24
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.action_scale = self.cfg.robot.action_scale
        self.actions_moving_average = getattr(self.cfg.robot, "actions_moving_average", None)

        ## Action buffer initialization
        # Action delay buffer to optionally inject latency between issued and applied actions.
        # NOTE: Currently the action delay is disabled
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            # Random per-env delay drawn within configured bounds.
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
        
        # Track previous action for observations (wrist force/torque + 18 finger targets).
        self.prev_action = torch.zeros(self.num_envs, 24, device=self.device)

        # Episode length buffer
        # Episode and timeout tracking.
        # episode_lenth_buf is equivalent to process_buf in the original unigrasptransformer, as it keep record of each envs simulation time
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        ## Observation buffer
        # Observation noise/scales.
        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        self.init_obs_buffer()

        # Resolved scene entity descriptor for the robot (ids for managers/reward terms).
        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        # Latest applied (clipped) action.
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        
        # Task outcome placeholders (to be wired in reward/termination logic).
        self.grasp_success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.grasp_hold_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Optional tactile/contact placeholders (populate once sensors are available).
        # NOTE: currently the fingertip sensors are not yet implemented
        self.fingertip_contact_forces = torch.zeros(self.num_envs, 5, 3, device=self.device)
        self.fingertip_contact_torques = torch.zeros(self.num_envs, 5, 3, device=self.device)
    
    def compute_current_observations(self):
        ### update current observations
        # refresh raw data from sim
        # update object related states from live sim tensors
        obj_state = self.object.data.root_state_w  # (E, 13)
        self.object_pose = obj_state
        self.object_pos = obj_state[:, 0:3]
        self.object_rot = obj_state[:, 3:7]
        self.object_linvel = obj_state[:, 7:10]
        self.object_angvel = obj_state[:, 10:13]
        # # simple placeholders for handle/back positions (offset along +X in object frame)
        # offset_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # self.object_handle_pos = self.object_pos  # no separate handle in USD; use center
        # self.object_back_pos = self.object_pos + quat_apply(self.object_rot, offset_x * 0.04)

        # update object points (E, 1024, 3) if available
        self.object_points = self.object_points
        self.object_points_centered = self.object_points - self.object_pos.unsqueeze(1)

        ## update hand related states
        # update hand root pose (palm)
        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.hand_body_state[:, idx, 0:3]
        self.right_hand_rot = self.hand_body_state[:, idx, 3:7]
        # apply small offsets to approximate palm center like upstream
        z_offset = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1) * 0.08
        y_offset = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1) * -0.02
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, z_offset)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, y_offset)
        
        # update fingertip states directly from fingertip bodies (no extra offsets)
        body_state = self.hand_body_state  # (E, B, 13)
        tip_pos = body_state[:, self.fingertip_ids, 0:3]
        tip_rot = body_state[:, self.fingertip_ids, 3:7]
        self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos = (
            tip_pos[:, 0], tip_pos[:, 1], tip_pos[:, 2], tip_pos[:, 3], tip_pos[:, 4]
        )
        self.right_hand_ff_rot, self.right_hand_mf_rot, self.right_hand_rf_rot, self.right_hand_lf_rot, self.right_hand_th_rot = (
            tip_rot[:, 0], tip_rot[:, 1], tip_rot[:, 2], tip_rot[:, 3], tip_rot[:, 4]
        )
        self.fingertip_pos = tip_pos
        self.fingertip_state = body_state[:, self.fingertip_ids][:, :, 0:13]

        # update hand_joint_pos and hand_joint_rot using valid body indices
        self.hand_joint_pos = body_state[:, self.valid_shadow_hand_bodies, 0:3]
        self.hand_joint_rot = body_state[:, self.valid_shadow_hand_bodies, 3:7]
        # update hand_body_pos (E,36,3): it is the defined 36 hand points
        self.hand_body_pos = self.hand_points
        # self.hand_body_pos = compute_hand_body_pos(self.hand_joint_pos, self.hand_joint_rot)
        
        ## update goal pose
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        ## update hand dof pose
        # get hand dof pose (per joint position)
        self.dof_pos = self.hand_joint_pos

        def world2obj_vec(vec):
            return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        def obj2world_vec(vec):
            return quat_apply(self.object_rot, vec) + self.object_pos
        def world2obj_quat(quat):
            return quat_mul(quat_conjugate(self.object_rot), quat)
        def obj2world_quat(quat):
            return quat_mul(self.object_rot, quat)
        
        # Distance from current hand pose to target hand pose
        self.delta_target_hand_pos = world2obj_vec(self.right_hand_pos) - self.target_hand_pos
        self.rel_hand_rot = world2obj_quat(self.right_hand_rot)
        self.delta_target_hand_rot = quat_mul(self.rel_hand_rot, quat_conjugate(self.target_hand_rot))
        self.delta_qpos = self.dof_pos - self.target_qpos

        # Distance from hand pos to object point clouds
        self.right_hand_pc_dist = batch_sided_distance(self.right_hand_pos.unsqueeze(1), self.object_points).squeeze(-1)        
        self.right_hand_pc_dist = torch.where(self.right_hand_pc_dist >= 0.5, 0.5 + 0 * self.right_hand_pc_dist, self.right_hand_pc_dist)

        # Distance from hand finger pos to object point clouds
        self.right_hand_finger_pos = torch.stack([self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos], dim=1)
        self.right_hand_finger_pc_dist = torch.sum(batch_sided_distance(self.right_hand_finger_pos, self.object_points), dim=-1)
        self.right_hand_finger_pc_dist = torch.where(self.right_hand_finger_pc_dist >= 3.0, 3.0 + 0 * self.right_hand_finger_pc_dist, self.right_hand_finger_pc_dist)

        # Distance from all hand joint pos to object point clouds
        self.right_hand_joint_pc_dist = torch.sum(batch_sided_distance(self.hand_joint_pos, self.object_points), dim=-1) * 5 / self.hand_joint_pos.shape[1]
        self.right_hand_joint_pc_dist = torch.where(self.right_hand_joint_pc_dist >= 3.0, 3.0 + 0 * self.right_hand_joint_pc_dist, self.right_hand_joint_pc_dist)
        # Distance from all hand body pos to object point clouds
        self.right_hand_body_pc_batch_dist = batch_sided_distance(self.hand_body_pos, self.object_points)
        self.right_hand_body_pc_dist = torch.sum(self.right_hand_body_pc_batch_dist, dim=-1) * 5 / self.hand_body_pos.shape[1]
        self.right_hand_body_pc_dist = torch.where(self.right_hand_body_pc_dist >= 3.0, 3.0 + 0 * self.right_hand_body_pc_dist, self.right_hand_body_pc_dist)
        # Distance from hand rot to target pca rot
        self.delta_target_hand_pca = 2 * torch.acos(torch.abs(torch.clamp(torch.sum(self.right_hand_rot * self.target_hand_pca_rot, dim=1), -1.0, 1.0)))
        
        ## compute current hand object states(unpose)
        self.hand_object_states = self.compute_hand_object_states()

        ## vision_based settings
        if self.vision_based:
            pass
        else:
            pass

        ### compute current full state from current observation
        self.asymmetric_obs = False 
        self.compute_current_state(self.asymmetric_obs)

    def compute_current_state(self):
        """
        Build observations using cached per-env tensors loaded in the helpers.

        hand_dofs (66): 22 joint pos, 22 joint vel, 22 joint force (zeros if unavailable)
        hand_fingers (95): fingertip state (5*13) + fingertip force/torque zeros (5*6)
        hand_states (6): wrist pos (3) + wrist euler (3)
        actions (24): previous action
        objects (16): object pos/quat/linvel/angvel + goal delta
        object_visual (64): placeholder zeros
        times (29): step + sinusoidal encoding (optional)
        hand_objects (36): hand point cloud to object distances (padded/truncated)
        """
        # compute full current state from the previously computed observations
        # get unpose quat 
        self.get_unpose_quat()
        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##

        # init obs dict
        obs_dict = dict()

        # # ---------------------- ShadowHand Observation 167 ---------------------- # #
        # 0:66, 22x3 shadow_hand dof positions, velocities, and forces
        hand_dof_pos = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        hand_dof_vel = self.vel_obs_scale * self.shadow_hand_dof_vel
        hand_dof_force = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]
        obs_dict['hand_dofs'] = torch.cat([hand_dof_pos, hand_dof_vel, hand_dof_force], dim=-1)

        # 66:131, 13x5 shadow_hand finger position, orientation, linear and angular velocities
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(5): aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])

        # 131:161: 6x5 shadow_hand finger force and torques, do not need repose
        finger_force_torques = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        obs_dict['hand_fingers'] = torch.cat([aux, finger_force_torques], dim=-1)

        # 161:167: 3+3 shadow_hand(I think it is palm) position, orientation
        hand_pos = self.unpose_point(self.right_hand_pos)
        hand_euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
        obs_dict['hand_states'] = torch.cat([hand_pos, hand_euler_xyz[0].unsqueeze(-1), hand_euler_xyz[1].unsqueeze(-1), hand_euler_xyz[2].unsqueeze(-1)], dim=-1)

        # # ---------------------- Action Observation 24 ---------------------- # #
        # 167:191: action
        self.actions[:, 0:3] = self.unpose_vec(self.actions[:, 0:3])
        self.actions[:, 3:6] = self.unpose_vec(self.actions[:, 3:6])
        obs_dict['actions'] = self.actions

        # # ---------------------- Object Observation 16 / 25 ---------------------- # #
        # 191:207 object pos, rot, linvel, angvel
        object_pos = self.unpose_point(self.object_pose[:, 0:3])  # 3
        object_rot = self.unpose_quat(self.object_pose[:, 3:7])  # 4
        object_linvel = self.unpose_vec(self.object_linvel)  # 3
        object_angvel = self.vel_obs_scale * self.unpose_vec(self.object_angvel)  # 4
        object_hand_dist = self.unpose_vec(self.goal_pos - self.object_pos)  # 3
        obs_dict['objects'] = torch.cat([object_pos, object_rot, object_linvel, object_angvel, object_hand_dist], dim=-1)

        # encode obj_pca, append object_pca at the end
        if 'encode_obj_pca' in self.config['Modes'] and self.config['Modes']['encode_obj_pca']:
            obs_dict['objects'] = torch.cat([obs_dict['objects'], self.object_pcas.reshape(self.num_envs, -1)], dim=-1)
        
        # zero_object_state
        if 'zero_object_state' in self.config['Modes'] and self.config['Modes']['zero_object_state']:
            obs_dict['objects'] = torch.zeros_like(obs_dict['objects'], device=self.device)
        
        # # ---------------------- Object Visual Observation 128 ---------------------- # #
        # 207:335 object visual feature, default 0
        obs_dict['object_visual'] = self.object_points_visual_features * 0
        # zero_object_visual_feature
        if self.algo == 'ppo' and 'zero_object_visual_feature' in self.config['Modes'] and self.config['Modes']['zero_object_visual_feature']:
            obs_dict['object_visual'] = torch.zeros_like(obs_dict['object_visual'], device=self.device)
        if self.algo == 'dagger_value' and 'zero_object_visual_feature' in self.config['Distills']  and self.config['Distills']['zero_object_visual_feature']:
            obs_dict['object_visual'] = torch.zeros_like(obs_dict['object_visual'], device=self.device)
        # encode dynamic object visual features
        if self.use_dynamic_visual_feats: obs_dict['object_visual'] = self.object_points_visual_features

        # # ---------------------- Time Observation 29 ---------------------- # #
        # 335:364 encode time vector
        if self.config['Modes']['encode_obs_time']:
            obs_dict['times'] = torch.cat([self.progress_buf.unsqueeze(-1), compute_time_encoding(self.progress_buf, 28)], dim=-1)
        
        # # ---------------------- Hand-Object Observation 36 ---------------------- # #
        # 364:400 encode hand object dist
        if 'encode_hand_object_dist' in self.config['Modes'] and self.config['Modes']['encode_hand_object_dist']:
            obs_dict['hand_objects'] = self.right_hand_body_pc_batch_dist
        
        # # ---------------------- Vision Based Setting ---------------------- # #
        # TODO: update vision_based observations
        self.vision_based = False
        if self.vision_based:
            # update objects with rendered object_centers
            obs_dict['objects'] *= 0.
            obs_dict['objects'][:, :3] = self.vision_based_tracker['object_centers']
            # update objects with estimated velocities
            if 'use_object_velocities' in self.config['Distills'] and self.config['Distills']['use_object_velocities']:
                obs_dict['objects'][:, 3:6] = self.vision_based_tracker['object_velocities']
            # update objects with estimated pcas
            if 'use_object_pcas' in self.config['Distills'] and self.config['Distills']['use_object_pcas']:
                obs_dict['objects'][:, 6:15] = self.vision_based_tracker['object_pcas'].reshape(self.vision_based_tracker['object_pcas'].shape[0], -1)
            # update object_visual with rendered object_features
            obs_dict['object_visual'] = self.vision_based_tracker['object_features']
            # update hand_objects with rendered hand_object_dists
            obs_dict['hand_objects'] = self.vision_based_tracker['hand_object_dists']
        
        # Make Final Obs List
        self.obs_names = ['hand_dofs', 'hand_fingers', 'hand_states', 'actions', 'objects', 'object_visual', 'times', 'hand_objects', 'object_ids', 'object_hots']
        # Cat Final Obs Buff
        self.obs_buf = torch.cat([obs_dict[name] for name in self.obs_names if name in obs_dict], dim=-1)

        # Make Final Obs Interval Dict
        start_temp, self.obs_infos = 0, {'names': [name for name in self.obs_names if name in obs_dict], 'intervals': {}}
        for name in self.obs_names:
            if name not in obs_dict: continue
            self.obs_infos['intervals'][name] = [start_temp, start_temp + obs_dict[name].shape[-1]]
            start_temp += obs_dict[name].shape[-1]
        
        return

    def compute_observations(self):
        # a higher level observation wrapper which takes per timestep observation and add noise and sensor data
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        
        # actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        # critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        print(f"Resetting envs: {env_ids.tolist()}")
        # input()

        # Reset buffer
        self.extras["log"] = dict()

        self.scene.reset(env_ids)
        #  just resets the scene manager’s bookkeeping and actors to whatever state it currently holds

        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        # Restore hand pose and joint state to defaults after scene/events.
        self.robot.write_root_state_to_sim(self.robot_default_root_state[env_ids], env_ids)
        # Specify env_ids so we only write into the subset being reset (avoids broadcasting errors).
        self.robot.write_joint_position_to_sim(self.robot_default_joint_pos[env_ids], env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim(self.robot_zero_joint_vel[env_ids], env_ids=env_ids)
        # Initialize repose yaw (upstream derives from sampled prior; here default to zero) and update pose quat.
        self.z_theta[env_ids] = 0.0
        self.update_pose_quat()
        self.prev_action[env_ids] = torch.zeros_like(self.prev_action[env_ids])
        if hasattr(self, "prev_joint_targets"):
            self.prev_joint_targets[env_ids] = self.robot_default_joint_pos[env_ids, -18:]

        # Restore object and goal states to defaults (zero velocities).
        obj_state = self.object_default_root_state[env_ids].clone()
        obj_state[:, 7:13] = self.object_zero_vel[env_ids]
        self.object.data.root_state_w[env_ids] = obj_state
        if self.goal_object is not None:
            goal_state = self.goal_default_root_state[env_ids].clone()
            goal_state[:, 7:13] = 0.0
            self.goal_object.data.root_state_w[env_ids] = goal_state

        # TODO: reset goal pose to initial pose or randomized pose

        # TODO: reset internal targets: prev_targets/cur_targets

        # clear buffers
        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        # the input actions are ppo outputs cliped within [-1,1]
        # you need to unclip it and apply to simulation
        # TODO: the control strategy is a bit off from the original unigrasptransformer
        # Expect 24D actions: 6 (wrist force/torque placeholders) + 18 finger joints.
        # Guard against invalid actions (NaN/Inf) coming from the policy.
        if not torch.isfinite(actions).all():
            self.extras.setdefault("log", {})
            self.extras["log"]["invalid_action"] = True
            actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        num_act = 24
        if actions.shape[1] != num_act:
            raise ValueError(f"action dimension mismatch: expected {num_act}, got {actions.shape[1]}")
        else:
            pass
            # print("action dimensions match expected size for step")
        
        # splitting into wrist action and joint action
        wrist_wrench, finger_targets = self.calculate_real_action(actions) # action modes(direct or relative) is already considered about in the calculate_real_action()

        # cache applied action for observations
        self.prev_action = torch.cat([wrist_wrench, finger_targets], dim=-1)

        # 18 actuated joints (masters); mimic joints follow via USD coupling
        finger_joint_ids = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19, 21],
            device=self.device,
            dtype=torch.long,
        )

        # apply wrist wrench on the palm (match upstream body index)
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        torques = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces[:, 0, :] = wrist_wrench[:, :3]
        torques[:, 0, :] = wrist_wrench[:, 3:]

        # Apply one action over multiple physics substeps (higher-rate physics than control)
        # and accumulate fingertip contact forces/speeds for averaging.
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            # apply force and torque control on the palm
            self.robot.set_external_force_and_torque(
                forces=forces, torques=torques, body_ids=[self.palm_body_id], is_global=True
            )
            # apply position control on the joints
            self.robot.set_joint_position_target(finger_targets, joint_ids=finger_joint_ids)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

            # TODO: contact sensor stuff I havenot figured out
            # self.avg_feet_force_per_step += torch.norm(
            #     self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            # )

        if not self.headless:
            self.sim.render()

        # data buffer processing
        self.episode_length_buf += 1
        reward_buf = self.reward_manager.compute(self.step_dt)

        # currently only rely on timeout reset
        self.reset_buf, self.time_out_buf = self.check_reset()
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        # refresh stage-derived object point cloud and hand points each control step
        self.refresh_stage_points()

        # get observation calculation
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        """
        Reset conditions placeholder.
        TODO: get upstream reset conditions
        Currently: only timeouts trigger resets.
        """
        # Placeholders for future flags (currently disabled).
        reach_goal = False  # set True when goal tolerance is reached
        success = False     # set True when success criteria are met
        current_success = False
        consecutive_success = False

        # goal based reset: object goal distance within a threshold
        goal_reset_buf = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        if reach_goal:
            goal_reset_buf[:] = True

        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf = time_out_buf | goal_reset_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.lin_vel * self.obs_scales.lin_vel
            noise_vec[3:6] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[6:9] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[9:12] = 0
            noise_vec[12 : 12 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[12 + self.num_actions : 12 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[12 + self.num_actions * 2 : 12 + self.num_actions * 3] = 0.0
            noise_vec[12 + self.num_actions * 3 : 18 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def update_pose_quat(self):
        """
        Build per-env pose quaternion from z-axis yaw (matches upstream UniGrasp "repose" handling).

        This rotates the wrist wrench into the posed frame; if z_theta is unset, falls back to identity.
        """
        if hasattr(self, "z_theta") and self.z_theta is not None:
            self.pose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta),
                torch.zeros_like(self.z_theta),
                self.z_theta,
            )
        else:
            self.pose_z_theta_quat = self.identity_quat
    
    def calculate_real_action(self, action):
        """
        Map normalized policy actions to wrist wrench + joint targets.

        Mimics original UniGrasp control:
        - If use_relative_control is True: integrate deltas with a speed scale.
        - Otherwise: direct position targets around defaults (scaled).
        """
        # Clamp incoming policy output to avoid runaway torques/positions from out-of-range samples.
        clip = self.clip_actions if self.clip_actions is not None else 1.0
        action = torch.clamp(action, -clip, clip)
        use_relative = getattr(self.cfg.robot, "use_relative_control", False)
        dof_speed_scale = getattr(self.cfg.robot, "dof_speed_scale", 1.0)
        transition_scale = getattr(self.cfg.robot, "transition_scale", 1.0)
        orientation_scale = getattr(self.cfg.robot, "orientation_scale", 1.0)

        wrist = action[:, :6]
        finger = action[:, 6:]

        # default and limits
        default_finger_pos = self.robot.data.default_joint_pos[:, -18:]
        joint_lower = getattr(self.robot.data, "joint_lower_limits", None)
        joint_upper = getattr(self.robot.data, "joint_upper_limits", None)
        has_limits = joint_lower is not None and joint_upper is not None

        # reorient and scale wrist wrench similar to upstream (pose_vec + gains).
        pose_quat = getattr(self, "pose_z_theta_quat", self.identity_quat)
        # forces/torques are scaled by dt and gains to match original magnitude mapping.
        # Use physics dt (not control step dt) to mirror upstream scaling.
        wrist_force = wrist[:, :3] * self.physics_dt * transition_scale * 100000.0
        wrist_torque = wrist[:, 3:6] * self.physics_dt * orientation_scale * 1000.0
        wrist_force = quat_apply(pose_quat, wrist_force)
        wrist_torque = quat_apply(pose_quat, wrist_torque)
        wrist_wrench = torch.cat([wrist_force, wrist_torque], dim=-1)

        if use_relative:
            # integrate deltas
            if not hasattr(self, "prev_joint_targets"):
                self.prev_joint_targets = default_finger_pos.clone()
            delta = finger * dof_speed_scale * self.step_dt
            joint_targets = self.prev_joint_targets + delta
            # clamp if limits available
            if has_limits:
                joint_targets = torch.clamp(joint_targets, joint_lower[:, -18:], joint_upper[:, -18:])
            self.prev_joint_targets = joint_targets.detach()
        else:
            # map normalized actions to joint limits (upstream uses scale to [lower, upper])
            if has_limits:
                span = joint_upper[:, -18:] - joint_lower[:, -18:]
                joint_targets = 0.5 * (finger + 1.0) * span + joint_lower[:, -18:]
                if self.actions_moving_average is not None:
                    prev = self.prev_joint_targets if hasattr(self, "prev_joint_targets") else default_finger_pos
                    joint_targets = (
                        self.actions_moving_average * joint_targets
                        + (1.0 - self.actions_moving_average) * prev
                    )
                joint_targets = torch.clamp(joint_targets, joint_lower[:, -18:], joint_upper[:, -18:])
            else:
                joint_targets = finger * self.action_scale + default_finger_pos

        return wrist_wrench, joint_targets

### ----------------------helpers------------------------ ###
    def _load_shadow_hand_assets(self, robot: Articulation):
        """
        Package per-env shadow hand data into a dictionary of tensors and metadata.

        Prim paths are not exposed through the articulation API, so they are omitted.
        """
        # raw data
        self.hand_root_state = robot.data.root_state_w  # (E, 13)
        self.hand_body_state = robot.data.body_state_w  # (E, nbody, 13)
        self.hand_joint_pos = robot.data.joint_pos  # (E, Ndof)
        self.hand_joint_vel = robot.data.joint_vel  # (E, Ndof)
        self.hand_joint_force = getattr(robot.data, "joint_force", torch.zeros_like(self.hand_joint_pos))
        
        # indicators
        self.joint_names = list(robot.data.joint_names)
        self.body_names = list(robot.data.body_names)
        self.hand_body_idx_dict = {name: i for i, name in enumerate(self.body_names)}
        self.hand_joint_idx_dict = {name: i for i, name in enumerate(self.joint_names)}
        self.fingertip_ids, _ = robot.find_bodies(
            name_keys=["fftip", "mftip", "rftip", "lftip", "thtip"], preserve_order=True
        )
        self.activated_joint_names = ACTIVE_JOINTS
        self.valid_shadow_hand_bodies = [1,7,8,9,12,13,14,16,18,19,21,22,24,25,27,29,31]


    def _load_object_asset_info(self, obj):
        """
        Package per-env grasp object data (world frame).

        Prim paths are not exposed through the rigid object API.
        """
        self.object_root_state = obj.data.root_state_w  # (E, 13)

    def _load_goal_point_asset_info(self, goal_obj):
        """
        Package per-env goal point data (world frame).
        """
        self.goal_states = goal_obj.data.root_state_w  # (E, 13)


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_hand_reward(
    modes: Dict[str, bool], 
    weights: Dict[str, float],
    object_init_z, 
    delta_qpos, delta_target_hand_pos, delta_target_hand_rot,
    id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf,
    successes, current_successes, consecutive_successes,
    max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
    right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool,
    # New obervations for computing reward
    object_points, right_hand_pc_dist, right_hand_finger_pc_dist, right_hand_joint_pc_dist, right_hand_body_pc_dist, delta_target_hand_pca
):
    # # ---------------------- State Update  ---------------------- # #
    # Action penalty
    action_penalty = torch.sum(actions ** 2, dim=-1)
    # Object lowest and heighest surface point
    heighest = torch.max(object_points[:, :, -1], dim=1)[0]
    lowest = torch.min(object_points[:, :, -1], dim=1)[0]

    # # ---------------------- Target Initial Hand State ---------------------- # #
    # Assign target initial hand pos in the midair
    target_z = heighest + 0.05
    target_xy = object_pos[:, :2]
    target_init_pos = torch.cat([target_xy, target_z.unsqueeze(-1)], dim=-1)
    # Distance from hand pos to target axis
    right_hand_axis_dist = torch.norm(target_xy - right_hand_pos[:, :2], p=2, dim=-1)
    # Distance from hand pos to target height point
    right_hand_init_dist = torch.norm(target_init_pos - right_hand_pos, p=2, dim=-1)

    # Assign target initial hand pose in the midair
    target_init_pose = torch.tensor([0.1, 0., 0.6, 0., 0., 0., 0.6, 0., -0.1, 0., 0.6, 0., 0., -0.2, 0., 0.6, 0., 0., 1.2, 0., -0.2, 0.], dtype=dof_pos.dtype, device=dof_pos.device)
    delta_init_qpos_value = torch.norm(dof_pos - target_init_pose, p=2, dim=-1)

    # right_hand_pose: regularize finger tip pose
    if 'right_hand_pose' not in weights: weights['right_hand_pose'] = 0.
    target_hand_pose = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=dof_pos.dtype, device=dof_pos.device)
    target_hand_mask = torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.], dtype=dof_pos.dtype, device=dof_pos.device)
    delta_qpos = torch.norm((torch.abs(dof_pos) - target_hand_pose) * target_hand_mask * (torch.abs(dof_pos) > 1.), p=2, dim=-1)

    # # ---------------------- Goal Distances ---------------------- # #
    # Distance from the object/hand pos to the goal pos
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)

    # # ---------------------- Hand Distances ---------------------- # #
    # Replace hand_pos_dist with hand_pc_dist
    right_hand_dist = right_hand_pc_dist
    right_hand_body_dist = right_hand_body_pc_dist
    right_hand_joint_dist = right_hand_joint_pc_dist
    right_hand_finger_dist = right_hand_finger_pc_dist

    # # ---------------------- Reward Weights ---------------------- # #
    # unpack hyper params
    max_finger_dist, max_hand_dist, max_goal_dist = weights['max_finger_dist'], weights['max_hand_dist'], weights['max_goal_dist']

    # right_hand_body_pc_dist
    if 'right_hand_body_dist' not in weights: weights['right_hand_body_dist'] = 0.

    # # ---------------------- Reward Computing ---------------------- # #
    # goal_conditioned
    if not goal_cond:
        # # ---------------------- Hold Detection / Reward Before Hold ---------------------- # #
        # hold_flag: hand pos and finger reach object region
        hold_value = 2
        hold_flag = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()
        
        # flag_joint_dist: hold flag with all joint dist
        if 'flag_joint_dist' in modes and modes['flag_joint_dist']:
            hold_flag = (right_hand_joint_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()

        # flag_body_dist: hold flag with all body dist
        if 'flag_body_dist' in modes and modes['flag_body_dist']:
            hold_flag = (right_hand_body_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()
        
        # # ---------------------- Hand Object Exploration ---------------------- # #
        object_points_sorted, _ = torch.sort(object_points, dim=-1)
        object_points_sorted = object_points_sorted[:, :object_points_sorted.shape[1]//4, :]
        random_indices = torch.randint(0, object_points_sorted.shape[1], (object_points_sorted.shape[0], 1))
        exploration_target_pos = object_points_sorted[torch.arange(object_points_sorted.shape[0]).unsqueeze(1), random_indices].squeeze(1)
        right_hand_exploration_dist = torch.norm(exploration_target_pos - right_hand_pos, p=2, dim=-1)

        # # ---------------------- Reward After Holding ---------------------- # #
        # Distanc from object pos to goal target pos
        goal_rew = torch.zeros_like(goal_dist)
        goal_rew = torch.where(hold_flag == hold_value, 1.0 * (0.9 - 2.0 * goal_dist), goal_rew)
        # Distance from hand pos to goal target pos
        hand_up = torch.zeros_like(goal_dist)
        hand_up = torch.where(lowest >= 0.61, torch.where(hold_flag == hold_value, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.80, torch.where(hold_flag == hold_value, 0.2 - goal_hand_dist * 0 + weights['hand_up_goal_dist'] * (0.2 - goal_dist), hand_up), hand_up)
        # Already hold the object and Already reach the goal
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(hold_flag == hold_value, torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

        # # ---------------------- Total Reward ---------------------- # #
        # init_reward: let hand approach inital height-axis point 
        init_reward = weights['delta_init_qpos_value'] * delta_init_qpos_value 
        init_reward += weights['right_hand_dist'] * right_hand_dist
        init_reward += weights['delta_target_hand_pca'] * delta_target_hand_pca 
        init_reward += weights['right_hand_exploration_dist'] * right_hand_exploration_dist 
        
        # grasp_reward: let hand fingers approach object, lift object to goal
        grasp_reward = weights['right_hand_body_dist'] * right_hand_body_dist + weights['right_hand_joint_dist'] * right_hand_joint_dist
        grasp_reward += weights['right_hand_finger_dist'] * right_hand_finger_dist + 2.0 * weights['right_hand_dist'] * right_hand_dist
        grasp_reward += weights['goal_dist'] * goal_dist + weights['goal_rew'] * goal_rew + weights['hand_up'] * hand_up + weights['bonus'] * bonus
        grasp_reward += weights['right_hand_pose'] * delta_qpos
        # Total Reward: init reward + grasp reward
        reward = torch.where(hold_flag != hold_value, init_reward, grasp_reward)

    else:
        # Difference between hand pose to target hand pose
        delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
        delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
        delta_qpos_value = torch.norm(delta_qpos, p=1, dim=-1)
        delta_value = 0.6 * delta_hand_pos_value + 0.04 * delta_hand_rot_value + 0.1 * delta_qpos_value 
        # Target flag: whether hand pose reaches the target hand pose
        target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 6.0).int()
        
        # Difference between object rotation and target rotation
        quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        # goal_hand_rew: already reached the target hand pose and hold the object, distance from object to goal target
        flag = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int() + target_flag
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 5, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)
        # hand_up: already hold the object, distance from hand height to goal target height
        flag2 = (right_hand_finger_dist <= max_finger_dist).int() + (right_hand_dist <= max_hand_dist).int()
        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= 0.63, torch.where(flag2 == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.80, torch.where(flag2 == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)
        # bonus: already reached the goal
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus)
        # Total Reward: hand pose/finger to object, object to goal target, hand height to goal target height, goal reach bonus
        reward = - 0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5 * delta_value
    
    # Init reset_buff
    resets = reset_buf
    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    # Reset goal also
    goal_resets = resets
    # Compute successes: reach the goal during running
    successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), successes)
    # Compute final_successes: reach the goal at the end
    final_successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), torch.zeros_like(successes))
    # Compute current_successes: reach the episode length and reach the goal
    current_successes = torch.where(resets == 1, successes, current_successes)
    # Compute cons_successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes, final_successes
