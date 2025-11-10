from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.assets.inspirehand.object_library import GraspObjectInfo

from .unigrasptransformer_cfg import UniGraspTransformerEnvCfg
from .grasp_helpers import compute_reward, apply_palm_motion, warp_hand_to_default
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
        # Optional dataset helpers
        self._pc_fps_path = getattr(cfg.scene.spawn.grasp_object, "pc_fps", None)
        self._pca_axes_path = getattr(cfg.scene.spawn.grasp_object, "pca_axes", None)
        self._object_init_path = getattr(cfg.scene.spawn.grasp_object, "object_init", None)
        self._pc_fps_np = None
        self._pca_axes_np = None
        self._object_init_np = None
        try:
            import numpy as _np
            if self._pc_fps_path:
                self._pc_fps_np = _np.load(self._pc_fps_path)
            if self._pca_axes_path:
                self._pca_axes_np = _np.load(self._pca_axes_path)
            if self._object_init_path:
                import pickle as _pkl
                with open(self._object_init_path, "rb") as _f:
                    self._object_init_np = _pkl.load(_f)
        except Exception:
            pass

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
        self._last_actions = torch.zeros(self.num_envs, 0, device=self.device)
        # No affordance/SDF buffers in UniGraspTransformer variant
        self._aff_sdf_grid = None
        self._aff_sdf_min = None
        self._aff_sdf_max = None
        self._non_sdf_grid = None
        self._non_sdf_min = None
        self._non_sdf_max = None
        self._latest_aff_sdf = None
        self._latest_non_sdf = None

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

        # Place hand 0.2m above the table surface, palm-down with local -X pointing toward table.
        # A +90° rotation about Y maps local -X to world -Z.
        hand_start_z = float(self._default_table_surface + 0.2)
        target_pos = self._default_hand_state.new_tensor([0.0, 0.0, hand_start_z])
        self._default_hand_state[:, :3] = target_pos - self._hand_root_offset
        # 90 degrees about the Y axis (palm-down if -X is palm normal). Quaternion stored as XYZW.
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

    # (Removed) SDF buffer initialization

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
        """Build observations to match the original UniGraspTransformer full-state layout.

        This mirrors the block structure used in the reference StateBasedGrasp:
        - hand_dofs (66): joint pos (22), joint vel (22), joint forces (22, zero-filled)
        - hand_fingers (95): per-finger kinematics (5*13) + forces/torques (5*6, zero-filled)
        - hand_states (6): root pos (3) + Euler xyz (3, zero-filled)
        - actions (24): last applied action (padded/truncated)
        - objects (17): obj pos(3), obj rot quat(4), obj lin vel(3), obj ang vel(4), goal_vec(3)
        - object_visual (128): zeros
        - times (29): step idx + time encoding (zeros)
        - hand_objects (36): zeros
        """
        num_envs = self.num_envs
        device = self.device
        dtype = self.robot.data.joint_pos.dtype

        def _pad_trunc(x: torch.Tensor, target: int) -> torch.Tensor:
            if x.shape[1] == target:
                return x
            if x.shape[1] > target:
                return x[:, :target]
            pad = torch.zeros(x.shape[0], target - x.shape[1], device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)

        # Hand DOFs (66)
        q = self.robot.data.joint_pos  # (N, n_dof)
        dq = self.robot.data.joint_vel  # (N, n_dof)
        q22 = _pad_trunc(q, 22)
        dq22 = _pad_trunc(dq, 22)
        f22 = torch.zeros(num_envs, 22, device=device, dtype=dtype)
        hand_dofs = torch.cat([q22, dq22, f22], dim=-1)

        # Hand fingers (95)
        tip_pos_w = self.hand.data.body_pos_w[:, self._tip_body_ids, :]  # (N,5,3)
        if hasattr(self.hand.data, "body_quat_w"):
            tip_quat_w = self.hand.data.body_quat_w[:, self._tip_body_ids, :]
        else:
            tip_quat_w = torch.zeros(num_envs, self._num_tips, 4, device=device, dtype=dtype)
            tip_quat_w[..., 3] = 1.0
        if hasattr(self.hand.data, "body_lin_vel_w"):
            tip_lin_vel_w = self.hand.data.body_lin_vel_w[:, self._tip_body_ids, :]
        else:
            tip_lin_vel_w = torch.zeros_like(tip_pos_w)
        if hasattr(self.hand.data, "body_ang_vel_w"):
            tip_ang_vel_w = self.hand.data.body_ang_vel_w[:, self._tip_body_ids, :]
        else:
            tip_ang_vel_w = torch.zeros_like(tip_pos_w)

        finger_kin = torch.cat([tip_pos_w, tip_quat_w, tip_lin_vel_w, tip_ang_vel_w], dim=-1)  # (N,5,13)
        finger_kin = finger_kin.reshape(num_envs, -1)  # (N,65)
        finger_ft = torch.zeros(num_envs, 30, device=device, dtype=dtype)
        hand_fingers = torch.cat([finger_kin, finger_ft], dim=-1)

        # Hand states (6): root pos + Euler (zeros)
        hand_pos = self.hand.data.root_pos_w  # (N,3)
        hand_euler = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        hand_states = torch.cat([hand_pos, hand_euler], dim=-1)

        # Actions (24): last applied (padded/truncated)
        last_act = getattr(self, "_last_actions", None)
        if last_act is None or last_act.numel() == 0:
            last_act = torch.zeros(num_envs, self.num_actions, device=device, dtype=dtype)
        act24 = _pad_trunc(last_act, 24)

        # Objects (17): pos, rot, lin vel, ang vel, goal vec
        if self.obj is not None:
            obj_pos = self.obj.data.root_pos_w
            obj_rot = self.obj.data.root_quat_w
        else:
            obj_pos = torch.zeros(num_envs, 3, device=device, dtype=dtype)
            obj_rot = torch.zeros(num_envs, 4, device=device, dtype=dtype)
            obj_rot[:, 3] = 1.0
        if hasattr(self.obj.data if self.obj is not None else self.hand.data, "root_lin_vel_w"):
            obj_lin_vel = (self.obj.data.root_lin_vel_w if self.obj is not None else torch.zeros_like(obj_pos))
        else:
            obj_lin_vel = torch.zeros_like(obj_pos)
        # 4-dim ang vel placeholder
        obj_ang_vel4 = torch.zeros(num_envs, 4, device=device, dtype=dtype)
        # Goal vector: from object to a goal height above table
        goal_z = self._default_table_surface + 0.20
        goal_pos = torch.tensor(self._default_object_pos, device=device, dtype=dtype).unsqueeze(0).repeat(num_envs, 1)
        goal_pos[:, 2] = goal_z
        goal_vec = goal_pos - obj_pos
        objects = torch.cat([obj_pos, obj_rot, obj_lin_vel, obj_ang_vel4, goal_vec], dim=-1)

        # Object visual (128), Times(29), Hand-objects(36)
        object_visual = torch.zeros(num_envs, 128, device=device, dtype=dtype)
        times = torch.zeros(num_envs, 29, device=device, dtype=dtype)
        times[:, 0] = self.episode_length_buf.to(dtype)
        hand_objects = torch.zeros(num_envs, 36, device=device, dtype=dtype)

        actor_obs = torch.cat(
            [hand_dofs, hand_fingers, hand_states, act24, objects, object_visual, times, hand_objects], dim=-1
        )
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
        # If object_init metadata is available, apply an initial rotation (and optional pos) for these envs
        # During __init__, BaseEnv may call reset() before self.obj is assigned.
        # Use getattr to avoid attribute errors in that early call.
        if self._object_init_np is not None and len(env_ids) > 0 and getattr(self, "obj", None) is not None:
            try:
                import numpy as _np
                ids = env_ids.to(dtype=torch.long, device=self.device)
                # Prefer test states if available
                states = self._object_init_np.get("test") or self._object_init_np.get("train")
                if states is not None and len(states) > 0:
                    picks = _np.random.randint(0, len(states), size=int(ids.shape[0]))
                    sel = torch.from_numpy(states[picks]).to(device=self.device, dtype=self.robot.data.root_pos_w.dtype)
                    obj_pose = self.obj.data.root_state_w.clone()
                    # Keep object at current z with slight clearance
                    obj_pose[ids, :3] = self.scene.env_origins[ids] + sel[:, :3]
                    obj_pose[ids, 3:7] = sel[:, 3:7]
                    self.obj.write_root_pose_to_sim(obj_pose[ids, :7], env_ids=ids)
            except Exception:
                pass
        warp_hand_to_default(self, env_ids)
        return result

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        # cache last actions for observation
        self._last_actions = delayed_actions.detach()

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
