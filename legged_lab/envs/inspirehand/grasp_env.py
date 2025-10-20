# legged_lab/envs/inspirehand/grasp_env.py

import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
from legged_lab.envs.base.base_env import BaseEnv
from .grasp_cfg import InspireHandGraspEnvCfg
from legged_lab.assets.inspirehand.object_library import GraspObjectLibrary, GraspObjectInfo


class InspireHandGraspEnv(BaseEnv):
    cfg: InspireHandGraspEnvCfg

    def __init__(
        self,
        cfg: InspireHandGraspEnvCfg,
        headless: bool | None = None,
        *,
        render_mode=None,
        **kwargs,
    ):
        self.render_mode = render_mode

        self._object_library = GraspObjectLibrary()
        self._object_infos = tuple(info for info in self._object_library.all_objects() if info.static_usd)
        if not self._object_infos:
            raise RuntimeError(
                "No converted grasp objects found. Run the USD conversion tool before constructing the environment."
            )
        self._object_rng = random.Random(cfg.scene.seed)
        self._default_object_pos = tuple(cfg.scene.grasp_object.init_state.pos)
        self._default_object_rot = tuple(cfg.scene.grasp_object.init_state.rot)
        if cfg.scene.table is not None:
            self._table_thickness = cfg.scene.table.spawn.size[2]
            self._default_table_surface = cfg.scene.table.init_state.pos[2] + self._table_thickness * 0.5
        else:
            self._table_thickness = 0.0
            self._default_table_surface = 0.0
        self._object_clearance = 0.01
        self._current_object: Optional[GraspObjectInfo] = None
        self._current_lowest = 0.0
        self._initial_object_info = self._sample_object_info()
        cfg.scene.grasp_object.spawn = sim_utils.UsdFileCfg(
            usd_path=self._initial_object_info.static_usd.as_posix(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=3.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        cfg.scene.grasp_object.init_state.pos = tuple(self._default_object_pos)
        cfg.scene.grasp_object.init_state.rot = tuple(self._default_object_rot)

        if headless is None:
            if isinstance(render_mode, bool):
                headless = not render_mode
            else:
                headless = render_mode is None

        super().__init__(cfg, headless)

        # Always present
        self.hand  = self.scene["robot"]
        self.table = self.scene["table"]
        self.obj   = self.scene["object"]

        self._default_hand_state = self.robot.data.root_state_w.clone()
        self._zero_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self._apply_object_info(self._initial_object_info, initial=True)

        fingertip_patterns = ["Link48", "Link4", "Link14", "Link24", "Link34"]
        tip_indices, tip_names = self.hand.find_bodies(name_keys=fingertip_patterns, preserve_order=True)
        if len(tip_indices) != len(fingertip_patterns):
            raise RuntimeError(
                f"Failed to locate all fingertip bodies. Found {tip_names}, expected {fingertip_patterns}."
            )
        self._tip_body_ids = tip_indices
        self._num_tips = len(tip_indices)
        self._local_tip_normals = torch.tensor(
            [[0.0, 0.0, 1.0]] * self._num_tips, dtype=torch.float, device=self.device
        )

        if self._table_thickness > 0.0:
            self._default_table_surface = float(self.table.data.root_pos_w[0, 2] + self._table_thickness * 0.5)
        else:
            self._default_table_surface = float(self.table.data.root_pos_w[0, 2])

        # action split: finger joints + palm translation
        self._joint_action_dim = self.robot.data.default_joint_pos.shape[1]
        self._palm_trans_action_dim = 3
        self._palm_rot_action_dim = 3
        self.num_actions = self._joint_action_dim + self._palm_trans_action_dim + self._palm_rot_action_dim
        self._palm_trans_action_scale = 0.03  # meters per env step for |action|=1
        self._palm_rot_action_scale = 0.2    # radians per env step for |action|=1

        # rebuild delay buffer to handle extended action dimension
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        # refresh observation buffers with updated action dimension
        self.init_obs_buffer()

        self._aff_sdf_grid = None
        self._aff_sdf_min = None
        self._aff_sdf_max = None
        self._non_sdf_grid = None
        self._non_sdf_min = None
        self._non_sdf_max = None
        self._latest_aff_sdf = None
        self._latest_non_sdf = None

        self._hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._current_object = self._initial_object_info
        self._load_object_sdfs(self._initial_object_info)

    # # ---------- helpers ----------
    # def _try_get_scene_entity(self, name: str):
    #     """Return scene[name] if it exists, else None."""
    #     try:
    #         return self.scene[name]
    #     except KeyError:
    #         return None

    # ---------- observations ----------
    def compute_current_observations(self):
        q = self.hand.data.joint_pos
        dq = self.hand.data.joint_vel
        num_envs = q.shape[0]

        # rel pos hand->object if object exists; otherwise zeros(3)
        aff_sdf_vals = None
        non_sdf_vals = None

        palm_p = self.hand.data.root_pos_w
        palm_quat_xyzw = self.hand.data.root_quat_w
        palm_quat = torch.cat((palm_quat_xyzw[:, 3:4], palm_quat_xyzw[:, :3]), dim=-1)
        palm_ang_vel = self.hand.data.root_com_ang_vel_w

        if self.obj is not None:
            obj_p = self.obj.data.body_pos_w[:, 0, :]
            obj_quat_xyzw = self.obj.data.body_quat_w[:, 0, :]
            obj_quat = torch.cat((obj_quat_xyzw[:, 3:4], obj_quat_xyzw[:, :3]), dim=-1)
            rel_p = obj_p - palm_p
        else:
            obj_quat = None
            rel_p = torch.zeros(self.num_envs, 3, device=self.device, dtype=q.dtype)

        if self.obj is not None and obj_quat is not None:
            obj_pos = obj_p
            obj_rot = obj_quat
            obj_rot_conj = quat_conjugate(obj_rot)

            tip_pos_w = self.hand.data.body_pos_w[:, self._tip_body_ids, :]
            rel_tip_w = tip_pos_w - obj_pos.unsqueeze(1)
            tip_pos_o = quat_apply(obj_rot_conj.unsqueeze(1).expand(-1, self._num_tips, -1), rel_tip_w)

            tip_quat_w = self.hand.data.body_quat_w[:, self._tip_body_ids, :]
            tip_normals_w = quat_apply(
                tip_quat_w.reshape(-1, 4),
                self._local_tip_normals.unsqueeze(0).expand(num_envs, -1, -1).reshape(-1, 3),
            ).reshape(num_envs, self._num_tips, 3)
            tip_normals_o = quat_apply(obj_rot_conj.unsqueeze(1).expand(-1, self._num_tips, -1), tip_normals_w)
        else:
            tip_pos_o = torch.zeros(num_envs, self._num_tips, 3, device=self.device, dtype=q.dtype)
            tip_normals_o = torch.zeros_like(tip_pos_o)

        if self._aff_sdf_grid is not None and self.obj is not None:
            aff_sdf_vals = self._sample_sdf_grid(self._aff_sdf_grid, self._aff_sdf_min, self._aff_sdf_max, tip_pos_o)
        else:
            aff_sdf_vals = torch.zeros(num_envs, self._num_tips, device=self.device, dtype=q.dtype)
        if self._non_sdf_grid is not None and self.obj is not None:
            non_sdf_vals = self._sample_sdf_grid(self._non_sdf_grid, self._non_sdf_min, self._non_sdf_max, tip_pos_o)
        else:
            non_sdf_vals = torch.zeros_like(aff_sdf_vals)

        self._latest_aff_sdf = aff_sdf_vals
        self._latest_non_sdf = non_sdf_vals

        fingertip_features = torch.cat(
            (tip_pos_o.reshape(num_envs, -1), tip_normals_o.reshape(num_envs, -1)),
            dim=-1,
        )

        sdf_features = torch.cat((aff_sdf_vals.reshape(num_envs, -1), non_sdf_vals.reshape(num_envs, -1)), dim=-1)

        if self.obj is not None and obj_quat is not None:
            wrist_rel_quat = quat_mul(quat_conjugate(obj_quat), palm_quat)
        else:
            wrist_rel_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=q.dtype).repeat(num_envs, 1)

        wrist_features = torch.cat((wrist_rel_quat, palm_ang_vel), dim=-1)

        actor_obs = torch.cat([q, dq, rel_p, wrist_features, fingertip_features, sdf_features], dim=-1)
        critic_obs = actor_obs
        return actor_obs, critic_obs

    def compute_observations(self):
        actor_obs, critic_obs = self.compute_current_observations()
        return actor_obs, critic_obs

    # ---------- step / rewards ----------
    def reset(self, env_ids):
        if isinstance(env_ids, torch.Tensor):
            env_tensor = env_ids
        else:
            env_tensor = torch.as_tensor(env_ids, device=self.device)

        result = super().reset(env_ids)
        if self._current_object is not None:
            self.extras.setdefault("info", {})
            self.extras["info"]["grasp/object_id"] = self._current_object.object_id
        self._warp_hand_to_default(env_ids)
        return result

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)

        joint_actions = delayed_actions[:, : self._joint_action_dim]
        palm_trans_actions = delayed_actions[
            :, self._joint_action_dim : self._joint_action_dim + self._palm_trans_action_dim
        ]
        palm_rot_actions = delayed_actions[:, self._joint_action_dim + self._palm_trans_action_dim :]

        joint_actions = torch.clip(joint_actions, -self.clip_actions, self.clip_actions).to(self.device)
        joint_targets = joint_actions * self.action_scale + self.robot.data.default_joint_pos

        self._apply_palm_motion(palm_trans_actions, palm_rot_actions)

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(joint_targets)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        _ = self.reward_manager.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        reward_buf = self._get_rewards()
        extras = self.extras
        extras["observations"]["critic"] = actor_obs
        return actor_obs, reward_buf, self.reset_buf, extras

    def _get_rewards(self) -> torch.Tensor:
        # Smoothness (tiny penalty) based on last action from BaseEnv’s delay buffer
        last_joint_action = self.action_buffer._circular_buffer.buffer[:, -1, : self._joint_action_dim]
        r_smooth = -0.001 * (last_joint_action**2).sum(dim=1)

        # If we don't have an object yet, only smoothness applies
        if self.obj is None:
            return r_smooth

        palm_p = self.hand.data.root_pos_w
        obj_p  = self.obj.data.root_pos_w

        # reach reward
        dist = torch.linalg.norm(obj_p - palm_p, dim=1)
        r_reach = torch.exp(-3.0 * dist)

        # lift reward: uses table if present, otherwise a heuristic z-threshold
        if self.table is not None:
            table_z = self.table.data.root_pos_w[:, 2] + self._table_thickness * 0.5
            object_bottom = obj_p[:, 2] - self._current_lowest
            lifted = object_bottom > (table_z + self.cfg.reward_scales.lift_height_buffer)
        else:
            lifted = obj_p[:, 2] > self.cfg.reward_scales.ground_lift_height

        r_lift = lifted.float() * 1.0

        # hold bonus for sustained lift
        self._hold_counter = torch.where(lifted, self._hold_counter + 1, torch.zeros_like(self._hold_counter))
        r_hold = 0.5 * (self._hold_counter > 10).float()

        reward = r_reach + r_lift + r_hold + r_smooth

        self.extras.setdefault("log", {})
        self.extras["log"]["reward/reach"] = r_reach.detach().cpu()
        self.extras["log"]["reward/lift"] = r_lift.detach().cpu()
        self.extras["log"]["reward/hold"] = r_hold.detach().cpu()
        self.extras["log"]["reward/smooth"] = r_smooth.detach().cpu()

        if self._latest_aff_sdf is not None:
            aff_mean = torch.abs(self._latest_aff_sdf).mean(dim=1)
            reward += 0.3 * torch.exp(-10.0 * aff_mean)
            self.extras["log"]["grasp/aff_sdf_mean"] = aff_mean.detach().cpu()
        if self._latest_non_sdf is not None:
            non_penalty = torch.relu(-self._latest_non_sdf).mean(dim=1)
            reward -= 0.4 * non_penalty
            self.extras["log"]["grasp/non_aff_penalty"] = non_penalty.detach().cpu()

        return reward

    def check_reset(self):
        # simple time-limit termination
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf = time_out_buf.clone()

        palm_pos = self.hand.data.root_pos_w
        obj_pos = self.obj.data.body_pos_w[:, 0, :] if self.obj is not None else palm_pos

        max_lateral = self.cfg.reset_cfg.max_lateral_distance
        max_vertical = self.cfg.reset_cfg.max_vertical_offset

        too_far = torch.linalg.norm(palm_pos[:, :2] - obj_pos[:, :2], dim=1) > max_lateral
        too_high = palm_pos[:, 2] > obj_pos[:, 2] + max_vertical
        reset_buf |= too_far | too_high | torch.isnan(palm_pos).any(dim=1)

        return reset_buf, time_out_buf

    def _apply_palm_motion(self, palm_trans: torch.Tensor, palm_rot: torch.Tensor):
        if palm_trans.numel() == 0 and palm_rot.numel() == 0:
            return

        root_state = self.robot.data.root_state_w.clone()

        if palm_trans.numel() != 0:
            clamped_trans = torch.clip(palm_trans, -1.0, 1.0) * self._palm_trans_action_scale
            root_state[:, :3] += clamped_trans

        if palm_rot.numel() != 0:
            clamped_rot = torch.clip(palm_rot, -1.0, 1.0) * self._palm_rot_action_scale
            angles = torch.linalg.norm(clamped_rot, dim=-1, keepdim=True)
            axis = torch.where(
                angles > 1e-6,
                clamped_rot / torch.clamp(angles, min=1e-6),
                torch.zeros_like(clamped_rot),
            )
            half_angles = 0.5 * angles
            delta_quat = torch.zeros((root_state.shape[0], 4), device=self.device, dtype=root_state.dtype)
            delta_quat[:, 0] = torch.cos(half_angles.squeeze(-1))
            delta_quat[:, 1:] = axis * torch.sin(half_angles.squeeze(-1)).unsqueeze(-1)

            current_quat_xyzw = root_state[:, 3:7]
            current_quat = torch.cat((current_quat_xyzw[:, 3:4], current_quat_xyzw[:, :3]), dim=-1)
            updated_quat = quat_mul(delta_quat, current_quat)
            updated_quat_xyzw = torch.cat((updated_quat[:, 1:], updated_quat[:, 0:1]), dim=-1)
            root_state[:, 3:7] = updated_quat_xyzw

        if self.table is not None:
            min_height = self.table.data.root_pos_w[:, 2] + 0.02
            root_state[:, 2].clamp_(min=min_height)

        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(self._zero_root_vel)

    # ---------- object management ----------
    def _sample_object_info(self) -> GraspObjectInfo:
        return self._object_rng.choice(self._object_infos)

    def _load_random_object(self) -> GraspObjectInfo:
        info = self._sample_object_info()
        self._apply_object_info(info)
        return info

    def _apply_object_info(self, info: GraspObjectInfo, *, initial: bool = False):
        self._current_object = info
        self._current_lowest = info.lowest_point or 0.0
        self._load_object_sdfs(info)

        base_x, base_y, _ = self._default_object_pos
        if initial:
            dx = dy = 0.0
            yaw = 0.0
        else:
            dx = self._object_rng.uniform(-0.04, 0.04)
            dy = self._object_rng.uniform(-0.04, 0.04)
            yaw = self._object_rng.uniform(-0.5, 0.5)

        table_surface = self._default_table_surface
        z_pos = table_surface - self._current_lowest + self._object_clearance

        dtype = self._zero_root_vel.dtype
        pos = torch.tensor([base_x + dx, base_y + dy, z_pos], device=self.device, dtype=dtype)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        quat_xyzw = torch.tensor([0.0, 0.0, sy, cy], device=self.device, dtype=dtype)

        pose = torch.zeros((self.num_envs, 7), device=self.device, dtype=dtype)
        pose[:, 0] = pos[0]
        pose[:, 1] = pos[1]
        pose[:, 2] = pos[2]
        pose[:, 3:] = quat_xyzw
        self.obj.write_root_pose_to_sim(pose)

        vel = torch.zeros((self.num_envs, 6), device=self.device, dtype=dtype)
        self.obj.write_root_velocity_to_sim(vel)

    def _load_object_sdfs(self, info: GraspObjectInfo):
        def _load(path: Optional[Path]):
            if path is None or not path.exists():
                return None, None, None
            data = np.load(path)
            grid = torch.from_numpy(data["sdf"]).to(self.device)
            min_bounds = torch.from_numpy(data["min_bounds"]).to(self.device).float()
            max_bounds = torch.from_numpy(data["max_bounds"]).to(self.device).float()
            return grid, min_bounds, max_bounds

        self._aff_sdf_grid, self._aff_sdf_min, self._aff_sdf_max = _load(info.affordance_sdf)
        self._non_sdf_grid, self._non_sdf_min, self._non_sdf_max = _load(info.non_affordance_sdf)
        self._latest_aff_sdf = None
        self._latest_non_sdf = None

    def _warp_hand_to_default(self, env_ids):
        if not hasattr(self, "_default_hand_state"):
            return

        if isinstance(env_ids, torch.Tensor):
            indices = env_ids.to(torch.long)
        elif isinstance(env_ids, slice) or env_ids is None:
            indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if indices.numel() == 0:
            return

        root_state = self.robot.data.root_state_w.clone()
        root_state[indices] = self._default_hand_state[indices]
        self.robot.write_root_pose_to_sim(root_state[indices, :7], env_ids=indices)
        self.robot.write_root_velocity_to_sim(self._zero_root_vel[indices], env_ids=indices)

    def _sample_sdf_grid(self, grid: torch.Tensor, min_bounds: torch.Tensor, max_bounds: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if grid is None:
            return torch.zeros(points.shape[:-1], device=self.device, dtype=points.dtype)

        res = torch.tensor(grid.shape, device=self.device, dtype=points.dtype)
        res_minus_one = res - 1.0

        norm = (points - min_bounds) / (max_bounds - min_bounds)
        norm = torch.clamp(norm, 0.0, 1.0) * res_minus_one

        idx0 = torch.floor(norm).long()
        idx1 = torch.minimum(idx0 + 1, (res_minus_one.long()))
        frac = norm - idx0.float()

        c000 = grid[idx0[..., 0], idx0[..., 1], idx0[..., 2]]
        c100 = grid[idx1[..., 0], idx0[..., 1], idx0[..., 2]]
        c010 = grid[idx0[..., 0], idx1[..., 1], idx0[..., 2]]
        c110 = grid[idx1[..., 0], idx1[..., 1], idx0[..., 2]]
        c001 = grid[idx0[..., 0], idx0[..., 1], idx1[..., 2]]
        c101 = grid[idx1[..., 0], idx0[..., 1], idx1[..., 2]]
        c011 = grid[idx0[..., 0], idx1[..., 1], idx1[..., 2]]
        c111 = grid[idx1[..., 0], idx1[..., 1], idx1[..., 2]]

        fx, fy, fz = frac[..., 0], frac[..., 1], frac[..., 2]
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz
