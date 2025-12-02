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
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply

from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import UnigraspTransformerGraspEnv
from legged_lab.utils.env_utils.unigrasptransformer_scene import UniGraspSceneCfg
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
        # unpacking some hyperparameters
        self.cfg: UnigraspTransformerGraspEnv = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        # set obs gating parameters
        self.encode_object_pca_obs = False
        self.encode_time_embed = True
        self.encode_hand_object_dist = True

        # build simulator
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        # call scene builder and create scene
        scene_cfg = UniGraspSceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        # Extract live scene components. These are bound to the created scene and used for sim I/O.
        # - self.robot: articulation API for issuing controls and reading state.
        # - self.fingertip_ids: indices of fingertip bodies for convenience in rewards/obs.
        # - self.object: the grasp target rigid object.
        # - self.goal_object: the independently spawned goal point
        
        ### GET RELATED DATA!
        ## robot related initialization
        self.robot: Articulation = self.scene["robot"]
        self.fingertip_ids, _ = self.robot.find_bodies(
            name_keys=["fftip", "mftip", "rftip", "lftip", "thtip"], preserve_order=True
        )
        # palm/wrist body (upstream applies wrench here)
        palm_ids, _ = self.robot.find_bodies(name_keys=["palm"], preserve_order=True)
        self.palm_body_id = int(palm_ids[0]) if len(palm_ids) > 0 else 1
        # Cache default states for clean resets.
        self.robot_default_root_state = self.robot.data.root_state_w.clone()
        self.robot_default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.robot_zero_joint_vel = torch.zeros_like(self.robot.data.joint_vel)
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # Per-env repose angle around z (mirrors upstream UniGrasp handling of random yaw);
        # used to rotate wrist wrench into the posed frame when available.
        self.z_theta = torch.zeros(self.num_envs, device=self.device)
        self.pose_z_theta_quat = self.identity_quat.clone()

        ## object related initialization
        self.object = self.scene["object"]
        self.object_default_root_state = self.object.data.root_state_w.clone()
        self.object_zero_vel = torch.zeros_like(self.object.data.root_state_w[:, 7:13])
        # Goal pose
        self.goal_object = self.scene["object_goal"]
        self.goal_states = self.goal_object.data.root_state_w  # (E, 13)
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        if self.goal_object is not None:
            self.goal_default_root_state = self.goal_object.data.root_state_w.clone()
        # initialization of hand points and object pc; refreshed each step from stage
        self.refresh_stage_points()

        # TODO: set object and hand initial pose and position (upstream places hand at a default pose,
        #       sets object to an init state, optionally rotates by a sampled prior yaw, zeroes velocities;
        #       no free-fall drop by default).


        print("DEBUGGING data extraction")
        ## DEBUGGING robot related data
        print(self.robot)
        # robot data debugs
        print(self.robot.data.__dict__.keys())
        # print(self.robot.data.root_state_w)
        self.joint_names = self.robot.data.joint_names
        print("Joint order:", self.joint_names)
        self.body_names = list(self.robot.data.body_names)
        print("body order:", [f"{i}:{name}" for i, name in enumerate(self.body_names)])
        ## DEBUGGING object related data
        # print(self.object)
        # print(self.pc)
        # input()

        # self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.init_buffers()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        print("INITIALIZATION SUCCESSFUL!")

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
        # TODO: check each observation domain to make sure they have the right reading
        # Observation terms (mirrors upstream ordering/lengths; many are zeros/placeholders until sensors/encoders are wired):
        #  hand_dofs (66): 22 joint pos, 22 joint vel, 22 joint force (zeros)
        #  hand_fingers (95): fingertip pos/quat/linvel/angvel for 5 tips (5*13), fingertip force/torque (zeros, 5*6)
        #  hand_states (6): wrist pos (3) + wrist euler (3)
        #  actions (24): previous action (wrist wrench + finger targets)
        #  objects (16): object pos/quat/linvel/angvel + goal delta
        #  object_visual (64): placeholder zeros
        #  times (29): step count + sinusoidal encoding (optional)
        #  hand_objects (36): hand point cloud to object distances (padded/truncated)

        # TODO: obs unpose and scaling
        robot_data = self.robot.data
        object_data = self.object.data

        # hand_dofs
        joint_angle = robot_data.joint_pos[:, :22]
        joint_vel = robot_data.joint_vel[:, :22]
        joint_force = torch.zeros_like(joint_vel)
        hand_dofs = torch.cat([joint_angle, joint_vel, joint_force], dim=-1)

        # hand_fingers
        body_state = robot_data.body_state_w  # (E, B, 13)
        fingertip_state = body_state[:, self.fingertip_ids, 0:13].reshape(self.num_envs, -1)  # 5*13
        fingertip_force_torque = torch.zeros((self.num_envs, 5 * 6), device=self.device)
        hand_fingers = torch.cat([fingertip_state, fingertip_force_torque], dim=-1)

        # hand_states
        wrist_pos = robot_data.root_state_w[:, 0:3]
        wrist_rot_euler_xyz = quat_to_euler_xyz(robot_data.root_state_w[:, 3:7])
        hand_states = torch.cat([wrist_pos, wrist_rot_euler_xyz], dim=-1)

        # previous actions (cached)
        actions = self.prev_action

        # objects
        obj_center = object_data.root_state_w[:, 0:3]
        obj_quat = object_data.root_state_w[:, 3:7]
        obj_lin_vel = object_data.root_state_w[:, 7:10]
        obj_ang_vel = object_data.root_state_w[:, 10:13]
        if self.goal_object is not None:
            self.goal_states = self.goal_object.data.root_state_w
            self.goal_pos = self.goal_states[:, 0:3]
            self.goal_rot = self.goal_states[:, 3:7]
            obj_goal_dist = self.goal_pos - obj_center
        else:
            obj_goal_dist = torch.zeros((self.num_envs, 3), device=self.device)
        objects = torch.cat([obj_center, obj_quat, obj_lin_vel, obj_ang_vel, obj_goal_dist], dim=-1)
        if getattr(self, "zero_object_state", False):
            objects = torch.zeros_like(objects)

        # optional PCA block
        object_pca = None
        if getattr(self, "encode_object_pca_obs", False):
            object_pca = torch.zeros((self.num_envs, 9), device=self.device)

        # object_visual placeholder: upstream code actually allocates 128-D (PointNet feature) but their hardcoded num_obs comment still says 64;
        # here we keep a 64-D zero block until a visual encoder is wired. (upstream legacy feature)
        object_visual = torch.zeros((self.num_envs, 64), device=self.device)
        if getattr(self, "zero_object_visual_feature", False):
            object_visual = torch.zeros_like(object_visual)

        # time embedding
        obs_parts = [hand_dofs, hand_fingers, hand_states, actions, objects]
        if object_pca is not None:
            obs_parts.append(object_pca)
        obs_parts.append(object_visual)
        if self.encode_time_embed:
            current_time = self.episode_length_buf.unsqueeze(-1).float()
            time_sin_cos = compute_time_encoding(self.episode_length_buf.float(), 28)
            time_embed = torch.cat([current_time, time_sin_cos], dim=-1)
            obs_parts.append(time_embed)

        # hand-object distances (pad/truncate to 36 to mirror upstream default)
        if self.encode_hand_object_dist:
            hand_body_pos = self.hand_points
            object_pc = self.object_points
            hand_object_dist = batch_sided_distance(hand_body_pos, object_pc).view(self.num_envs, -1)
            if hand_object_dist.shape[1] < 36:
                pad = torch.zeros((self.num_envs, 36 - hand_object_dist.shape[1]), device=self.device)
                hand_object_dist = torch.cat([hand_object_dist, pad], dim=-1)
            elif hand_object_dist.shape[1] > 36:
                hand_object_dist = hand_object_dist[:, :36]
            obs_parts.append(hand_object_dist)

        current_actor_obs = torch.cat(obs_parts, dim=-1)
        if not self.cfg.robot.asymmetric_obs:
            current_critic_obs = current_actor_obs
        else:
            raise ValueError("you should patch critic observation with extra information than actor obs")
        return current_actor_obs, current_critic_obs

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
        self.robot.write_joint_position_to_sim(self.robot_default_joint_pos[env_ids])
        self.robot.write_joint_velocity_to_sim(self.robot_zero_joint_vel[env_ids])
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
