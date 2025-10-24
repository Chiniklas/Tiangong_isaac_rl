from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.assets.inspirehand.object_library import GraspObjectInfo

from .unigrasptransformer_cfg import UniGraspTransformerEnvCfg
from .grasp_helpers import compute_reward, apply_palm_motion, sample_sdf_grid, warp_hand_to_default
from .logging_utils import log_debug


class UniGraspTransformerEnv(BaseEnv):
    cfg: UniGraspTransformerEnvCfg

    def __init__(
        self,
        cfg: UniGraspTransformerEnvCfg,
        headless: bool | None = None,
        *,
        render_mode=None,
        **kwargs,
    ):
        self.render_mode = render_mode

        object_info = getattr(cfg.scene.spawn, "_override_object_info", None)
        if object_info is None:
            raise RuntimeError(
                "No object override provided. Ensure cfg.scene.spawn.config_path points to a YAML specifying 'object_dir'."
            )

        self._current_object: Optional[GraspObjectInfo] = object_info
        self._aff_sdf_data_np = cfg.scene.spawn.grasp_object.affordance_sdf_data
        self._non_sdf_data_np = cfg.scene.spawn.grasp_object.non_affordance_sdf_data

        self._default_object_pos = tuple(cfg.scene.grasp_object.init_state.pos)
        self._default_object_rot = tuple(cfg.scene.grasp_object.init_state.rot)
        if cfg.scene.table is not None:
            self._table_thickness = cfg.scene.table.spawn.size[2]
            self._default_table_surface = cfg.scene.table.init_state.pos[2] + self._table_thickness * 0.5
        else:
            self._table_thickness = 0.0
            self._default_table_surface = 0.0
        self._object_clearance = 0.01
        lowest = object_info.lowest_point
        if lowest is None:
            lowest = cfg.scene.spawn.grasp_object.lowest_point
        if lowest is None:
            lowest = 0.0
        self._current_lowest = float(lowest if lowest <= 0.0 else -lowest)
        if cfg.scene.spawn.grasp_object.pos is not None:
            self._default_object_pos = tuple(cfg.scene.spawn.grasp_object.pos)
        if cfg.scene.spawn.grasp_object.rot is not None:
            self._default_object_rot = tuple(cfg.scene.spawn.grasp_object.rot)

        if headless is None:
            if isinstance(render_mode, bool):
                headless = not render_mode
            else:
                headless = render_mode is None

        super().__init__(cfg, headless)
        log_debug(
            f"UniGraspTransformerEnv initialized (num_envs={self.num_envs}, object={object_info.object_id})"
        )

        self.hand = self.scene["robot"]
        self.table = self.scene["table"]
        self.obj = self.scene["object"]

        self._hand_spawn_cfg = cfg.scene.spawn.hand
        self._default_hand_state = self.robot.data.root_state_w.clone()
        self._hand_root_offset = torch.zeros(3, dtype=self._default_hand_state.dtype, device=self.device)
        self._zero_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self._aff_sdf_grid = None
        self._aff_sdf_min = None
        self._aff_sdf_max = None
        self._non_sdf_grid = None
        self._non_sdf_min = None
        self._non_sdf_max = None
        self._latest_aff_sdf = None
        self._latest_non_sdf = None

        self._initialize_sdf_buffers()
        self._set_object_pose()

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

        if self.table is not None:
            if self._table_thickness > 0.0:
                self._default_table_surface = float(
                    self.table.data.root_pos_w[0, 2] + self._table_thickness * 0.5
                )
            else:
                self._default_table_surface = float(self.table.data.root_pos_w[0, 2])
        else:
            self._default_table_surface = float(self._default_object_pos[2])

        target_pos = self._default_hand_state.new_tensor([0.0, 0.0, 0.75])
        self._default_hand_state[:, :3] = target_pos - self._hand_root_offset
        palm_down_xyzw = torch.tensor(
            (0.0, 0.70710678, 0.0, 0.70710678),
            dtype=self._default_hand_state.dtype,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self._default_hand_state[:, 3:7] = palm_down_xyzw
        warp_hand_to_default(self, slice(None))

        self._joint_action_dim = self.robot.data.default_joint_pos.shape[1]
        self._palm_trans_action_dim = 3
        self._palm_rot_action_dim = 3
        self.num_actions = self._joint_action_dim + self._palm_trans_action_dim + self._palm_rot_action_dim
        self._palm_trans_action_scale = 0.03
        self._palm_rot_action_scale = 0.2

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

        self.init_obs_buffer()
        self._hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _initialize_sdf_buffers(self) -> None:
        def _as_tensor(array: Optional[np.ndarray]) -> Optional[torch.Tensor]:
            if array is None:
                return None
            return torch.from_numpy(array).to(device=self.device, dtype=torch.float32)

        aff_data = getattr(self, "_aff_sdf_data_np", None)
        if aff_data is not None:
            self._aff_sdf_grid = _as_tensor(aff_data.get("grid"))
            self._aff_sdf_min = _as_tensor(aff_data.get("min_bounds"))
            self._aff_sdf_max = _as_tensor(aff_data.get("max_bounds"))
        else:
            self._aff_sdf_grid = None
            self._aff_sdf_min = None
            self._aff_sdf_max = None

        non_data = getattr(self, "_non_sdf_data_np", None)
        if non_data is not None:
            self._non_sdf_grid = _as_tensor(non_data.get("grid"))
            self._non_sdf_min = _as_tensor(non_data.get("min_bounds"))
            self._non_sdf_max = _as_tensor(non_data.get("max_bounds"))
        else:
            self._non_sdf_grid = None
            self._non_sdf_min = None
            self._non_sdf_max = None

        self._latest_aff_sdf = None
        self._latest_non_sdf = None

    def _set_object_pose(self) -> None:
        if self.obj is None:
            return

        dtype = self.obj.data.root_pos_w.dtype
        origins = self.scene.env_origins.to(device=self.device, dtype=dtype)
        base_x, base_y, _ = self._default_object_pos
        z_pos = self._default_table_surface - self._current_lowest + self._object_clearance

        obj_pos = torch.tensor([base_x, base_y, z_pos], dtype=dtype, device=self.device)
        obj_rot = torch.tensor(self._default_object_rot, dtype=dtype, device=self.device)

        obj_pose = torch.zeros(self.num_envs, 13, dtype=dtype, device=self.device)
        obj_pose[:, :3] = origins + obj_pos
        obj_pose[:, 3:7] = obj_rot

        self.obj.write_root_pose_to_sim(obj_pose[:, :7])
        obj_vel = torch.zeros(self.num_envs, 6, dtype=dtype, device=self.device)
        self.obj.write_root_velocity_to_sim(obj_vel)

    def compute_current_observations(self):
        num_envs = self.num_envs

        q = self.robot.data.joint_pos
        dq = self.robot.data.joint_vel

        root_state = self.robot.data.root_pos_w
        rel_p = root_state - self.scene.env_origins

        palm_quat_xyzw = self.robot.data.root_quat_w
        palm_quat = torch.cat((palm_quat_xyzw[:, 3:4], palm_quat_xyzw[:, :3]), dim=-1)
        default_xyzw = self._default_hand_state[:, 3:7]
        default_quat = torch.cat((default_xyzw[:, 3:4], default_xyzw[:, :3]), dim=-1)
        wrist_rel_quat = quat_mul(quat_conjugate(default_quat), palm_quat)

        palm_ang_vel = self.robot.data.root_ang_vel_w

        if self.obj is not None:
            obj_p = self.obj.data.root_pos_w
            obj_quat = self.obj.data.root_quat_w
            obj_quat = torch.cat((obj_quat[:, 3:4], obj_quat[:, :3]), dim=-1)
            obj_rot_conj = quat_conjugate(obj_quat)
        else:
            obj_p = torch.zeros(num_envs, 3, device=self.device, dtype=q.dtype)
            obj_rot_conj = None

        if self.table is not None:
            table_height = self.table.data.root_pos_w[:, 2] + self._table_thickness * 0.5
            rel_p[:, 2] = rel_p[:, 2] - table_height
        else:
            rel_p[:, 2] = rel_p[:, 2] - self._default_table_surface

        if self.obj is not None:
            obj_pos = obj_p
            rel_p = torch.cat((rel_p, obj_pos - self.scene.env_origins), dim=-1)
        else:
            rel_p = torch.cat((rel_p, torch.zeros_like(rel_p)), dim=-1)

        if self.obj is not None:
            obj_pos = obj_p
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
            aff_sdf_vals = sample_sdf_grid(self._aff_sdf_grid, self._aff_sdf_min, self._aff_sdf_max, tip_pos_o)
        else:
            aff_sdf_vals = torch.zeros(num_envs, self._num_tips, device=self.device, dtype=q.dtype)
        if self._non_sdf_grid is not None and self.obj is not None:
            non_sdf_vals = sample_sdf_grid(self._non_sdf_grid, self._non_sdf_min, self._non_sdf_max, tip_pos_o)
        else:
            non_sdf_vals = torch.zeros_like(aff_sdf_vals)

        self._latest_aff_sdf = aff_sdf_vals
        self._latest_non_sdf = non_sdf_vals

        fingertip_features = torch.cat(
            (tip_pos_o.reshape(num_envs, -1), tip_normals_o.reshape(num_envs, -1)),
            dim=-1,
        )
        sdf_features = torch.cat((aff_sdf_vals.reshape(num_envs, -1), non_sdf_vals.reshape(num_envs, -1)), dim=-1)
        wrist_features = torch.cat((wrist_rel_quat, palm_ang_vel), dim=-1)

        actor_obs = torch.cat([q, dq, rel_p, wrist_features, fingertip_features, sdf_features], dim=-1)
        critic_obs = actor_obs
        return actor_obs, critic_obs

    def compute_observations(self):
        actor_obs, critic_obs = self.compute_current_observations()
        return actor_obs, critic_obs

    def reset(self, env_ids):
        if isinstance(env_ids, torch.Tensor):
            env_tensor = env_ids
        else:
            env_tensor = torch.as_tensor(env_ids, device=self.device)

        result = super().reset(env_ids)
        if self._current_object is not None:
            self.extras.setdefault("info", {})
            self.extras["info"]["grasp/object_id"] = self._current_object.object_id
        warp_hand_to_default(self, env_ids)
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

        apply_palm_motion(self, palm_trans_actions, palm_rot_actions)

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

        reward_buf, reward_logs = compute_reward(self)
        self.extras.setdefault("log", {})
        self.extras["log"].update(reward_logs)
        extras = self.extras
        extras["observations"]["critic"] = actor_obs
        return actor_obs, reward_buf, self.reset_buf, extras

    def check_reset(self):
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


__all__ = ["UniGraspTransformerEnv"]
