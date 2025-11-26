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
    _load_yaml_cfg
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
        self.cfg: UnigraspTransformerGraspEnv = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

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
        self.pc = self.scene["object_point_cloud"]

        print("DEBUGGING data extraction")
        print(self.robot)
        # general robot debugs
        # print(self.robot.__dict__)  # internal refs
        # print(dir(self.robot))      # method/attr names

        # robot data debugs
        # print(self.robot.data.__dict__.keys())
        # print(self.robot.data.root_state_w)
        print(self.robot.data.joint_names) # this is the joint order: ['FFJ4', 'LFJ5', 'MFJ4', 'RFJ4', 'THJ5', 'FFJ3', 'LFJ4', 'MFJ3', 'RFJ3', 'THJ4', 'FFJ2', 'LFJ3', 'MFJ2', 'RFJ2', 'THJ3', 'FFJ1', 'LFJ2', 'MFJ1', 'RFJ1', 'THJ2', 'LFJ1', 'THJ1']
        
        # print(self.object)
        # print(self.pc)
        # input()

        # self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.init_buffers()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        # if "startup" in self.event_manager.available_modes:
        #     self.event_manager.apply(mode="startup")
        # self.reset(env_ids)

    def init_buffers(self):
        """
        buffers are basicly action buffer, obs buffer, and episode length buffer
        """
        # Per-episode extras and bookkeeping.
        self.extras = {}

        # unpack some hyperparameters
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
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

        # Episode length buffer
        # Episode and timeout tracking.
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
        previous_action = self.action_buffer
        sim_step_counter = getattr(self, "sim_step_counter", 0)
        sim_time = float(sim_step_counter) * float(getattr(self, "step_dt", 0.0))


        # unpack observation from real readings
        wrist_pos = robot_data.root_state_w[:, 0:3]
        wrist_rot = robot_data.root_state_w[:, 3:7] # TODO: the wrist rot is in quaternion form, but the paper ask for 3 dims
        
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
                wrist_rot,
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

        prev_wrist_force = torch.zeros((self.num_envs, 3), device=self.device)
        prev_wrist_torque = torch.zeros((self.num_envs, 3), device=self.device)
        prev_finger_angles = torch.zeros((self.num_envs, 18), device=self.device)
        prev_action = torch.cat([prev_wrist_force, 
                                 prev_wrist_torque, 
                                 prev_finger_angles], 
                                 dim=-1)

        obj_center = torch.zeros((self.num_envs, 3), device=self.device)
        obj_quat = torch.zeros((self.num_envs, 4), device=self.device)
        obj_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        obj_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        obj_goal_dist = torch.zeros((self.num_envs, 3), device=self.device)
        object_state = torch.cat([obj_center, 
                                  obj_quat,
                                  obj_lin_vel,
                                  obj_ang_vel, 
                                  obj_goal_dist], 
                                  dim=-1)

        
        hand_object_dist = torch.zeros((self.num_envs, 36), device=self.device)

        current_time = torch.zeros((self.num_envs, 1), device=self.device)
        time_sin_cos = torch.zeros((self.num_envs, 28), device=self.device)
        time_embed = torch.cat([current_time, 
                                time_sin_cos], 
                                dim=-1)

        current_actor_obs = torch.cat([proprio, prev_action, object_state, hand_object_dist, time_embed], dim=-1)
        current_critic_obs = current_actor_obs
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
        ## action process
        # this part is for later delay and domain randomization, currently we don't need it.
        # if self.cfg.domain_rand.action_delay.enable:
        #     delayed_actions = self.action_buffer.compute(actions)
        # else:
        #     delayed_actions = actions
        # self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        # processed_actions = self.action * self.action_scale + self.robot.data.default_joint_pos

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

        self.episode_length_buf += 1
        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

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

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

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
