# UniGraspTransformer hand environment (simplified from TienKungEnv):
# - Shadow Hand as the default robot, no AMP pipeline, no terrain curriculum.
# - Hand-only body IDs (fingertips); limb IDs left empty as placeholders.
# - Observation pipeline currently stubbed; fill task-specific obs as needed.
# - Retains base sim/scene/buffer/step structure for compatibility with RL runners.

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_rotate
from scipy.spatial.transform import Rotation

from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import UnigraspTransformerGraspEnv
from legged_lab.utils.env_utils.unigrasptransformer_scene import UniGraspSceneCfg
from rsl_rl.env import VecEnv
from legged_lab.envs.unigrasptransformer.helpers import (
    _load_yaml_cfg,
    compute_time_encoding,
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
        # - self.pc: point cloud sensor attached to the object.
        self.robot: Articulation = self.scene["robot"]
        self.fingertip_ids, _ = self.robot.find_bodies(
            name_keys=["fftip", "mftip", "rftip", "lftip", "thtip"], preserve_order=True
        )
        self.object = self.scene["object"]

        # initialization of hand points and object pc; refreshed each step from stage
        self.refresh_stage_points()

        #TODO: get point cloud local and from real overlay, check if they are the same, if yes, then point cloud data importing is successful
        # option1: get point cloud from pc overlay (per env)
        # pc_np = get_point_cloud_world(env_index=None, prim_suffix="ObjectPC")  # (E,P,3) numpy
        # self.object_points = torch.as_tensor(pc_np, device=self.device, dtype=torch.float32)
        # print(f"Point cloud loaded from USD overlay: {self.object_points.shape}")

        
        # option2: get point cloud from npy (world frame), convert to object-local for each env
        # self.object_points_local = None
        # pc_path = getattr(getattr(self.cfg.scene, "grasp_object", None), "pc_fps_path", None)
        # if pc_path:
        #     try:
        #         pts_np = np.load(Path(pc_path).expanduser())
        #         if pts_np.shape[1] >= 3:
        #             pc_world = torch.as_tensor(pts_np[:, :3], device=self.device, dtype=torch.float32)  # (P,3)
        #             obj_pos = self.object.data.root_state_w[:, 0:3]          # (E,3)
        #             obj_quat = self.object.data.root_state_w[:, 3:7]         # (E,4)
        #             quat_obj_conj = quat_conjugate(obj_quat)                 # (E,4)
        #             pc_world_exp = pc_world.unsqueeze(0).expand(self.num_envs, -1, -1)  # (E,P,3)
        #             pc_local = quat_apply(quat_obj_conj.unsqueeze(1), pc_world_exp - obj_pos.unsqueeze(1))  # (E,P,3)
        #             self.object_points_local = pc_local
        #             print(f"Loaded npy point cloud from {pc_path}, world->local shape {self.object_points_local.shape}")
        #     except Exception as exc:
        #         print(f"Failed to load/convert object point cloud from {pc_path}: {exc}")
        # else:
        #     print("pc_path not found!")

        # # quick consistency check: overlay vs npy directly (both expected in world frame, per env)
        # if getattr(self, "object_points", None) is not None and self.object_points_local is not None:
        #     if self.object_points.shape != self.object_points_local.shape:
        #         print(f"Overlay vs npy point cloud shape mismatch: {self.object_points.shape} vs {self.object_points_local.shape}")
        #     else:
        #         diff = self.object_points - self.object_points_local
        #         max_err = diff.abs().max().item()
        #         mean_err = diff.abs().mean().item()
        #         status = "IDENTICAL" if max_err < 1e-5 else "DIFFERS"
        #         print(f"Overlay vs npy point cloud diff -> max {max_err:.6f}, mean {mean_err:.6f} [{status}]")

        print("DEBUGGING data extraction")
        ## DEBUGGING robot related data
        print(self.robot)
        # general robot debugs
        # print(self.robot.__dict__)  # internal refs
        # print(dir(self.robot))      # method/attr names
        # robot data debugs
        # print(self.robot.data.__dict__.keys())
        # print(self.robot.data.root_state_w)
        self.joint_names = self.robot.data.joint_names
        print("Joint order:", self.joint_names)
        self.body_names = list(self.robot.data.body_names)
        print("body order:", [f"{i}:{name}" for i, name in enumerate(self.body_names)])
        # # adopt UniGrasp-style valid body selection by name (order matters)
        # preferred_valid_names = [
        #     "palm",
        #     "ffproximal", "ffmiddle", "ffdistal",
        #     "mfproximal", "mfmiddle", "mfdistal",
        #     "rfknuckle", "rfmiddle", "rfdistal",
        #     "lfmetacarpal", "lfknuckle", "lfmiddle", "lfdistal",
        #     "thbase", "thhub", "thdistal",
        # ]
        # self.valid_body_indices = [i for name in preferred_valid_names for i, n in enumerate(self.body_names) if n == name]
        # self.valid_body_names = [self.body_names[i] for i in self.valid_body_indices]
        # # skip offsets within the valid list (mirror original 2,5,8,12 skips)
        # self.skip_body_offsets = [2, 5, 8, 12]
        # self.left_out_body_indices = [self.valid_body_indices[i] for i in self.skip_body_offsets if i < len(self.valid_body_indices)]
        # print("valid bodies (indices):", self.valid_body_indices)
        # print("left-out bodies (indices):", self.left_out_body_indices) # should be all the finger middle part

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

        ## Action buffer initialization
        # Action delay buffer to optionally inject latency between issued and applied actions.
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
        self.fingertip_contact_forces = torch.zeros(self.num_envs, 5, 3, device=self.device)
        self.fingertip_contact_torques = torch.zeros(self.num_envs, 5, 3, device=self.device)
    

    def compute_current_observations(self):
        # TODO: check each observation domain to make sure they have the right reading
        # dim = 167 + 24 + 16 + 36 + 29
        #       proprioception(167)
        #           wrist position(3) and rotation(3)
        #           Finger-joint angle(22), angular velocity(22) and force(22)
        #           Fingertip position(5*3), quaternion rotation(5*4), linear velocity(5*3), angular velocity(5*3), force(5*3) and torque(5*3)
        #       Previous action(24)
        #           wrist force(3) and torque(3); finger-joint angles(18)
        #       Object state(16)
        #           Object center(3), quaternion rotation(4), linear velocity(3), angular velocity(3), object-goal distance(3)
        #       Hand-Object Distance(36)
        #           hand body points to object point cloud distances (36)
        #       Time(29)
        #           current time(1), sine-cosine time embedding(28)

        # get raw data
        robot_data = self.robot.data
        object_data = self.object.data
        prev_action = self.prev_action # extract previous action
     
        # unpack observation from real readings
        wrist_pos = robot_data.root_state_w[:, 0:3]
        wrist_rot_quat = robot_data.root_state_w[:, 3:7]
        wrist_rot_euler_xyz = quat_to_euler_xyz(wrist_rot_quat)  # convert quat (xyzw) to Euler xyz

        joint_angle = robot_data.joint_pos[:, :22]
        joint_vel = robot_data.joint_vel[:, :22]
        joint_force = robot_data.joint_measured_force[:, :22]
        
        body_state = self.robot.data.body_state_w  # (E, B, 13)
        fingertip_pos = body_state[:, self.fingertip_ids, 0:3].reshape(self.num_envs, -1)  # (E, 5*3)
        fingertip_rot = body_state[:, self.fingertip_ids, 3:7].reshape(self.num_envs, -1)  # (E, 5*4)
        fingertip_lin_vel = body_state[:, self.fingertip_ids, 7:10].reshape(self.num_envs, -1)  # (E, 5*3)
        fingertip_ang_vel = body_state[:, self.fingertip_ids, 10:13].reshape(self.num_envs, -1)  # (E, 5*3)
        
        # TODO: fingertip_force and torque can not be acquired by contact sensors, still working on how to solve it.
        fingertip_force = torch.zeros((self.num_envs, 5 * 3), device=self.device)
        fingertip_torque = torch.zeros((self.num_envs, 5 * 3), device=self.device)
        proprio = torch.cat(
            [
                wrist_pos,
                wrist_rot_euler_xyz,
                joint_angle,
                joint_vel,
                joint_force,
                fingertip_pos,
                fingertip_rot,
                fingertip_lin_vel,
                fingertip_ang_vel,
                fingertip_force,
                fingertip_torque,
            ],
            dim=-1,
        )

        # previous action: 6 wrist wrench + 18 finger targets
        prev_wrist_force = prev_action[:,:3]
        prev_wrist_torque = prev_action[:,3:6]
        prev_finger_angles = prev_action[:,6:]
        prev_action_state = torch.cat([prev_wrist_force, 
                                 prev_wrist_torque, 
                                 prev_finger_angles], 
                                 dim=-1)

        # object pose/velocity in world frame
        obj_center = object_data.root_state_w[:, 0:3]
        obj_quat = object_data.root_state_w[:, 3:7]
        obj_lin_vel = object_data.root_state_w[:, 7:10]
        obj_ang_vel = object_data.root_state_w[:, 10:13]
        # TODO: wire in goal position to compute goal distance; placeholder zero for now.
        obj_goal_dist = torch.zeros((self.num_envs, 3), device=self.device)
        object_state = torch.cat(
            [obj_center,
             obj_quat, 
             obj_lin_vel, 
             obj_ang_vel, 
             obj_goal_dist],
            dim=-1,
        )
        
        ## extract hand points and point cloud from the stage, and calculate distance
        # it seems the original unigrasptransformer hardcoded the interesting body indexes
        # from the original mjcf, the body index looks like this:
        """
        0 hand mount, 1 palm, 2 ffknuckle, 3 ffproximal, 4 ffmiddle, 
        5 ffdistal, 6 mfknuckle, 7 mfproximal, 8 mfmiddle, 9 mfdistal, 
        10 rfknuckle, 11 rfproximal, 12 rfmiddle, 13 rfdistal, 14 lfmetacarpal, 
        15 lfknuckle, 16 lfproximal, 17 lfmiddle, 18 lfdistal, 19 thbase, 
        20 thproximal, 21 thhub, 22 thmiddle, 23 thdistal
        """
        # and they pick: valid_shadow_hand_bodies = [1,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,23] to add normal offsets
        # and for 2,5,8,12, they apply special offsets

        # in our usd, the body order is:
        """
        body order: ['0:world', '1:palm', '2:ffknuckle', '3:lfmetacarpal', '4:mfknuckle', '5:rfknuckle', '6:thbase', '7:palm_center_marker', '8:ffproximal', '9:lfknuckle', '10:mfproximal', '11:rfproximal', '12:thproximal', '13:palm_dir_marker', '14:ffmiddle', '15:lfproximal', '16:mfmiddle', '17:rfmiddle', '18:thhub', '19:ffdistal', '20:lfmiddle', '21:mfdistal', '22:rfdistal', '23:thmiddle', '24:fftip', '25:lfdistal', '26:mftip', '27:rftip', '28:thdistal', '29:lftip', '30:thtip']
        valid bodies (indices): [1, 8, 14, 19, 10, 16, 21, 5, 17, 22, 3, 9, 20, 25, 6, 18, 28]
        left-out bodies (indices): [14, 16, 17, 20]
        """

        # get hand body points from stage at runtime
        hand_body_pos = self.hand_points
        # get object point data from stage at runtime
        object_pc = self.object_points
        # calculate hand object distance
        hand_object_dist = batch_sided_distance(hand_body_pos, object_pc).view(self.num_envs, -1)

        ## time embeddings
        # one thing could happen is that our physics step and control step are not at the same frequency
        current_time = self.episode_length_buf.unsqueeze(-1).float()
        time_sin_cos = compute_time_encoding(self.episode_length_buf.float(), 28)
        time_embed = torch.cat([current_time, 
                                time_sin_cos], 
                                dim=-1)

        current_actor_obs = torch.cat([proprio, prev_action_state, object_state, hand_object_dist, time_embed], dim=-1)
        current_critic_obs = current_actor_obs
        # TODO: is it necessary to seperate actor critic observations?
        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        # a higher level observation wrapper which takes per timestep observation and add noise and sensor data
        # in our dex grasp case, there are two modes
        # one is purely state based policy training, you should only use the perstep observation
        # second is vision based policy training, you should extend the perstep observation
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.extras["log"] = dict()

        # TODO: mirror upstream UniGraspTransformer reset:
        # - Randomize goal pose / target pose (goal_env_ids handling)
        # - Sample grasp priors / target hand poses if available
        # - Reset hand DOF state/targets to defaults (with optional noise)
        # - Restore saved root states for hand/object/table
        # - Apply random object yaw and align hand orientation (optionally to PCA)
        # - Apply static init states if configured

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

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
        # TODO: the control strategy is a bit off from the original unigrasptransformer
        # Expect 24D actions: 6 (wrist force/torque placeholders) + 18 finger joints.
        num_act = 24
        if actions.shape[1] != num_act:
            raise ValueError(f"action dimension mismatch: expected {num_act}, got {actions.shape[1]}")
        
        # action process
        if self.cfg.domain_rand.action_delay.enable:
            delayed_actions = self.action_buffer.compute(actions)
        else:
            delayed_actions = actions
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos

        # our action process
        # Expect 24D actions: 6 (wrist force/torque placeholders) + 18 finger joints.
        num_act = 24
        if actions.shape[1] != num_act:
            raise ValueError(f"action dimension mismatch: expected {num_act}, got {actions.shape[1]}")

        # clip/scale 
        # TODO: clip and scaling still not clear
        clip_actions = getattr(self, "clip_actions", self.cfg.normalization.clip_actions)
        action_scale = getattr(self, "action_scale", self.cfg.robot.action_scale)
        actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # split
        wrist_wrench = actions[:, :6]  # fx, fy, fz, tx, ty, tz
        finger_actions = actions[:, 6:]  # 18 finger joints

        # map finger actions to joint targets; assume last 18 joints correspond to fingers
        # TODO: figure out the unigrasptransformer action type, do they use delta action?
        default_finger_pos = self.robot.data.default_joint_pos[:, -18:]
        processed_finger_actions = finger_actions * action_scale + default_finger_pos

        # build full joint target: keep non-finger joints at default
        processed_actions = self.robot.data.default_joint_pos.clone()
        processed_actions[:, -18:] = processed_finger_actions

        # cache applied action for observations
        self.prev_action = torch.cat([wrist_wrench, processed_finger_actions], dim=-1)

        # apply wrist wrench on the root link (body id 0) in the global frame
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        torques = torch.zeros((self.num_envs, 1, 3), device=self.device)
        forces[:, 0, :] = wrist_wrench[:, :3]
        torques[:, 0, :] = wrist_wrench[:, 3:]
        
        # Apply one action over multiple physics substeps (higher-rate physics than control)
        # and accumulate fingertip contact forces/speeds for averaging.
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            # apply force and torque control on the palm
            self.robot.set_external_force_and_torque(forces=forces, torques=torques, body_ids=[0], is_global=True)
            # apply position control on the joints
            self.robot.set_joint_position_target(processed_actions) # 18 dim joint positions
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
        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        # refresh stage-derived object point cloud and hand points each control step
        self.refresh_stage_points()

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > 1.0,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
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
    
    def calculate_real_action(self, action):
        """
        Map normalized policy actions to wrist wrench + joint targets.

        Mimics original UniGrasp control:
        - If use_relative_control is True: integrate deltas with a speed scale.
        - Otherwise: direct position targets around defaults (scaled).
        """
        use_relative = getattr(self.cfg.robot, "use_relative_control", False)
        dof_speed_scale = getattr(self.cfg.robot, "dof_speed_scale", 1.0)

        wrist = action[:, :6]
        finger = action[:, 6:]

        # default and limits
        default_finger_pos = self.robot.data.default_joint_pos[:, -18:]
        joint_lower = getattr(self.robot.data, "joint_lower_limits", None)
        joint_upper = getattr(self.robot.data, "joint_upper_limits", None)

        if use_relative:
            # integrate deltas
            if not hasattr(self, "prev_joint_targets"):
                self.prev_joint_targets = default_finger_pos.clone()
            delta = finger * dof_speed_scale * self.step_dt
            joint_targets = self.prev_joint_targets + delta
            # clamp if limits available
            if joint_lower is not None and joint_upper is not None:
                joint_targets = torch.clamp(joint_targets, joint_lower[:, -18:], joint_upper[:, -18:])
            self.prev_joint_targets = joint_targets.detach()
        else:
            # direct targets around defaults with action_scale
            joint_targets = finger * self.action_scale + default_finger_pos
            if joint_lower is not None and joint_upper is not None:
                joint_targets = torch.clamp(joint_targets, joint_lower[:, -18:], joint_upper[:, -18:])

        return wrist, joint_targets
