from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
from legged_lab.envs.base.base_env import BaseEnv

from .unigrasptransformer_cfg import UniGraspTransformerEnvCfg
from .grasp_helpers import compute_reward, apply_palm_motion, warp_hand_to_default
from .logging_utils import log_debug
from .viz.observations_table import ObservationsTableViewer
from .viz.rewards_table import RewardsTableViewer


def _pad_trunc(x: torch.Tensor, target: int) -> torch.Tensor:
    if x.shape[1] == target:
        return x
    if x.shape[1] > target:
        return x[:, :target]
    pad = torch.zeros(x.shape[0], target - x.shape[1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def _xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 3:4], quat[..., :3]], dim=-1)


def _wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[..., 1:], quat[..., 0:1]], dim=-1)


def _quat_to_euler_xyz(quat_xyzw: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat_xyzw.unbind(-1)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(sinp.abs() >= 1.0, torch.sign(sinp) * (torch.pi / 2.0), torch.asin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


def _positional_encoding(progress: torch.Tensor, dim: int) -> torch.Tensor:
    div_term = torch.arange(0, dim, 2, dtype=torch.float32, device=progress.device)
    div_term = torch.exp(div_term * -(torch.log(torch.tensor(10000.0, device=progress.device)) / dim))
    encoding = torch.zeros(progress.shape[0], dim, device=progress.device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(progress.unsqueeze(1) * div_term.unsqueeze(0))
    encoding[:, 1::2] = torch.cos(progress.unsqueeze(1) * div_term.unsqueeze(0))
    return encoding.to(progress.dtype)


def _world_to_object_vec(vec: torch.Tensor, obj_pos: torch.Tensor, obj_conj: torch.Tensor) -> torch.Tensor:
    rel = vec - obj_pos
    conj = obj_conj
    while conj.dim() < rel.dim():
        conj = conj.unsqueeze(-2)
    conj = conj.expand(rel.shape[:-1] + (4,))
    rotated = quat_apply(conj.reshape(-1, 4), rel.reshape(-1, 3))
    return rotated.reshape_as(rel)


def _world_to_object_quat(quat_xyzw: torch.Tensor, obj_conj: torch.Tensor) -> torch.Tensor:
    quat_wxyz = _xyzw_to_wxyz(quat_xyzw)
    conj_wxyz = _xyzw_to_wxyz(obj_conj)
    while conj_wxyz.dim() < quat_wxyz.dim():
        conj_wxyz = conj_wxyz.unsqueeze(-2)
    conj_wxyz = conj_wxyz.expand_as(quat_wxyz)
    merged = quat_mul(conj_wxyz.reshape(-1, 4), quat_wxyz.reshape(-1, 4)).reshape_as(quat_wxyz)
    return _wxyz_to_xyzw(merged)


_HAND_JOINT_LABELS = (
    "palm",
    "ffknuckle",
    "ffproximal",
    "ffmiddle",
    "mfknuckle",
    "mfproximal",
    "mfmiddle",
    "rfknuckle",
    "rfproximal",
    "rfmiddle",
    "lfknuckle",
    "lfproximal",
    "lfmiddle",
    "thbase",
    "thproximal",
    "thmiddle",
    "thdistal",
)

_PALM_OFFSET_VECS = (
    torch.tensor([0.03, -0.005, 0.0], dtype=torch.float32),
    torch.tensor([-0.03, -0.005, 0.0], dtype=torch.float32),
    torch.tensor([0.0, -0.005, 0.03], dtype=torch.float32),
    torch.tensor([0.0, -0.005, 0.06], dtype=torch.float32),
    torch.tensor([0.03, -0.005, 0.06], dtype=torch.float32),
    torch.tensor([0.015, -0.005, 0.015], dtype=torch.float32),
    torch.tensor([-0.03, -0.005, 0.03], dtype=torch.float32),
)

_DEFAULT_OFFSET_LABELS = (
    "ffknuckle",
    "ffproximal",
    "ffmiddle",
    "mfknuckle",
    "mfproximal",
    "mfmiddle",
    "rfknuckle",
    "rfproximal",
    "rfmiddle",
    "lfknuckle",
    "lfproximal",
)

_SPECIAL_OFFSET_VECS = {"thproximal": (torch.tensor([-0.015, 0.0, 0.02], dtype=torch.float32),)}
_DEFAULT_OFFSET_VECTOR = torch.tensor([0.0, 0.0, 0.02], dtype=torch.float32)




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

        # Use spawn_cfg directly; no legacy GraspObjectInfo override needed
        self._current_object = None
        # Optional dataset helpers
        self._pc_fps_path = getattr(cfg.scene.spawn.grasp_object, "pc_fps", None)
        self._pca_axes_path = getattr(cfg.scene.spawn.grasp_object, "pca_axes", None)
        self._object_init_path = getattr(cfg.scene.spawn.grasp_object, "object_init", None)
        self._pc_fps_np = None
        self._pca_axes_np = None
        self._object_init_np = None
        self._object_pc_local = None
        self._object_pca_axes_tensor = None
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
        if self._pc_fps_np is not None:
            pts = torch.tensor(self._pc_fps_np, dtype=torch.float32)
            pts = pts.unsqueeze(0).repeat(cfg.scene.num_envs, 1, 1)
            self._object_pc_local = pts.to(cfg.device)
        if self._pca_axes_np is not None:
            axes = torch.tensor(self._pca_axes_np, dtype=torch.float32)
            axes = axes.unsqueeze(0).repeat(cfg.scene.num_envs, 1, 1)
            self._object_pca_axes_tensor = axes.to(cfg.device)

        # Default object pose from scene if object exists; otherwise fall back to zeros
        if cfg.scene.grasp_object is not None:
            self._default_object_pos = tuple(cfg.scene.grasp_object.init_state.pos)
            self._default_object_rot = tuple(cfg.scene.grasp_object.init_state.rot)
        else:
            self._default_object_pos = (0.0, 0.0, 0.0)
            self._default_object_rot = (0.0, 0.0, 0.0, 1.0)

        if cfg.scene.table is not None:
            self._table_thickness = cfg.scene.table.spawn.size[2]
            self._default_table_surface = cfg.scene.table.init_state.pos[2] + self._table_thickness * 0.5
        else:
            self._table_thickness = 0.0
            self._default_table_surface = 0.0
        # No lowest-point logic needed for UniGraspTransformer dataset
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
            f"UniGraspTransformerEnv initialized (num_envs={self.num_envs}, object={getattr(cfg.scene.spawn.grasp_object, 'object_id', None)})"
        )

        self.hand = self.scene["robot"]
        try:
            self.table = self.scene["table"]
        except KeyError:
            self.table = None
        try:
            self.obj = self.scene["object"]
        except KeyError:
            self.obj = None

        self._hand_spawn_cfg = cfg.scene.spawn.hand
        self._default_hand_state = self.robot.data.root_state_w.clone()
        self._hand_root_offset = torch.zeros(3, dtype=self._default_hand_state.dtype, device=self.device)
        self._zero_root_vel = torch.zeros(self.num_envs, 6, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._palm_dir_center_apis: list[Any] | None = None
        self._palm_dir_curve_paths: list[str] | None = None
        self._palm_dir_local: torch.Tensor | None = None
        self._palm_dir_offset_local: torch.Tensor | None = None
        self._palm_dir_scale: float = 0.0

        self._set_object_pose()

        fingertip_patterns = list(
            getattr(cfg.scene.spawn.hand, "fingertip_body_exprs", ("Link48", "Link4", "Link14", "Link24", "Link34"))
        )
        fingertip_names = ["thumb", "index", "middle", "ring", "little"]
        tip_indices, tip_names = self.hand.find_bodies(name_keys=fingertip_patterns, preserve_order=True)
        if len(tip_indices) != len(fingertip_patterns):
            raise RuntimeError(
                f"Failed to locate all fingertip bodies. Found {tip_names}, expected {fingertip_patterns}."
            )
        self._tip_body_ids = tip_indices
        self._num_tips = len(tip_indices)
        try:
            mapping_str = ", ".join(
                f"{i+1}:{fname}({p})" for i, (fname, p) in enumerate(zip(fingertip_names, fingertip_patterns))
            )
            log_debug(f"Fingertip mapping -> {mapping_str}")
        except Exception:
            pass
        self._local_tip_normals = torch.tensor(
            [[0.0, 0.0, 1.0]] * self._num_tips, dtype=torch.float, device=self.device
        )

        self._init_hand_body_samples()

        if self.table is not None:
            if self._table_thickness > 0.0:
                self._default_table_surface = float(
                    self.table.data.root_pos_w[0, 2] + self._table_thickness * 0.5
                )
            else:
                self._default_table_surface = float(self.table.data.root_pos_w[0, 2])
        else:
            self._default_table_surface = float(self._default_object_pos[2])

        # Cache initial hand pose from YAML for potential use on reset; robot is spawned with this pose via scene cfg
        hand_pos_cfg = torch.tensor(self._hand_spawn_cfg.pos, dtype=self.robot.data.root_pos_w.dtype, device=self.device)
        hand_rot_cfg = torch.tensor(self._hand_spawn_cfg.orientation_xyzw, dtype=self.robot.data.root_pos_w.dtype, device=self.device)
        self._default_hand_state[:, :3] = hand_pos_cfg - self._hand_root_offset
        self._default_hand_state[:, 3:7] = hand_rot_cfg

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

        # Create optional dataset overlays (point cloud / PCA axes) as part of scene spawn
        self._init_debug_overlays()

        # Observation debug controls
        self._print_obs_summary = False
        self._print_obs_entries = False
        self._enable_obs_table = os.environ.get("UNIGRASP_OBS_TABLE", "0") == "1"
        self._obs_table_env = int(os.environ.get("UNIGRASP_OBS_ENV", "0"))
        self._obs_table_viewer: Optional[ObservationsTableViewer] = None
        if self._enable_obs_table:
            try:
                self._obs_table_viewer = ObservationsTableViewer()
            except Exception as exc:
                print(f"[WARN] Failed to init observation table viewer: {exc}")
                self._enable_obs_table = False
        self._enable_reward_table = os.environ.get("UNIGRASP_REWARD_TABLE", "0") == "1"
        self._reward_table_env = int(os.environ.get("UNIGRASP_REWARD_ENV", "0"))
        self._reward_table_viewer: Optional[RewardsTableViewer] = None
        if self._enable_reward_table:
            try:
                self._reward_table_viewer = RewardsTableViewer()
            except Exception as exc:
                print(f"[WARN] Failed to init reward table viewer: {exc}")
                self._enable_reward_table = False

    # (Removed) SDF buffer initialization

    def _set_object_pose(self) -> None:
        if self.obj is None:
            return

        # Place object according to YAML-configured initial pose.
        dtype = self.obj.data.root_pos_w.dtype
        origins = self.scene.env_origins.to(device=self.device, dtype=dtype)
        obj_pos = torch.tensor(self._default_object_pos, dtype=dtype, device=self.device)
        obj_rot = torch.tensor(self._default_object_rot, dtype=dtype, device=self.device)

        obj_pose = torch.zeros(self.num_envs, 13, dtype=dtype, device=self.device)
        obj_pose[:, :3] = origins + obj_pos
        obj_pose[:, 3:7] = obj_rot

        self.obj.write_root_pose_to_sim(obj_pose[:, :7])
        obj_vel = torch.zeros(self.num_envs, 6, dtype=dtype, device=self.device)
        self.obj.write_root_velocity_to_sim(obj_vel)

    def _init_debug_overlays(self) -> None:
        """Author per-env overlays under each Object prim based on YAML flags.

        Overlays are created once in object-local space so they follow objects automatically
        without per-frame updates. Safe no-ops if object or data paths are missing.
        """
        try:
            import numpy as _np  # type: ignore
            import omni.usd  # type: ignore
            from pxr import Gf, UsdGeom  # type: ignore
        except Exception:
            return

        stage = omni.usd.get_context().get_stage()

        def _ensure_debug_root(env_idx: int, entity: str):
            root = f"/World/envs/env_{env_idx}/{entity}/Debug"
            UsdGeom.Xform.Define(stage, root)
            return root

        hand_spawn = self.cfg.scene.spawn.hand
        if getattr(hand_spawn, "show_palm_dir", False):
            dtype = self.hand.data.root_pos_w.dtype
            device = self.device
            self._palm_dir_local = torch.tensor(hand_spawn.palm_dir_local, dtype=dtype, device=device)
            self._palm_dir_offset_local = torch.tensor(
                getattr(hand_spawn, "palm_dir_offset_local", (0.0, 0.0, 0.0)), dtype=dtype, device=device
            )
            self._palm_dir_scale = float(hand_spawn.palm_dir_scale)
            self._palm_dir_center_apis = []
            self._palm_dir_curve_paths = []
            for env_idx in range(self.num_envs):
                debug_root = _ensure_debug_root(env_idx, "Robot")
                center_path = f"{debug_root}/PalmCenter"
                center_xform = UsdGeom.Xform.Define(stage, center_path)
                sphere = UsdGeom.Sphere.Define(stage, f"{center_path}/Geom")
                sphere.CreateRadiusAttr(0.015)
                sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.9, 1.0)])
                center_api = UsdGeom.XformCommonAPI(center_xform)
                self._palm_dir_center_apis.append(center_api)

                curve_path = f"{debug_root}/PalmDir"
                curve = UsdGeom.BasisCurves.Define(stage, curve_path)
                curve.CreateTypeAttr("linear")
                curve.CreateCurveVertexCountsAttr([2])
                curve.CreateWidthsAttr([0.02])
                curve.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.9, 1.0)])
                curve.GetPointsAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)])
                self._palm_dir_curve_paths.append(curve_path)

        spawn = self.cfg.scene.spawn.grasp_object
        if self.obj is None or not getattr(spawn, "enable", False):
            return

        if getattr(spawn, "show_point_cloud", False) and getattr(spawn, "pc_fps", None):
            try:
                pc = _np.load(spawn.pc_fps).astype(_np.float32)
            except Exception:
                pc = None
            if pc is not None:
                pc_max = 4096
                if pc.shape[0] > pc_max:
                    sel = _np.random.permutation(pc.shape[0])[:pc_max]
                    pc = pc[sel]
                points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in pc]
                for env_idx in range(self.num_envs):
                    debug_root = _ensure_debug_root(env_idx, "Object")
                    pc_prim = UsdGeom.Points.Define(stage, f"{debug_root}/ObjectPC")
                    pc_prim.CreateWidthsAttr([0.01])
                    pc_prim.GetDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.85, 0.95)])
                    pc_prim.GetPointsAttr().Set(points)

        if getattr(spawn, "show_pca_axes", False) and getattr(spawn, "pca_axes", None):
            try:
                axes = _np.load(spawn.pca_axes).astype(_np.float32)
            except Exception:
                axes = None
            if axes is not None:
                colors = [(1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0)]
                scale = 0.2
                for env_idx in range(self.num_envs):
                    debug_root = _ensure_debug_root(env_idx, "Object")
                    for axis_idx in range(3):
                        curve = UsdGeom.BasisCurves.Define(stage, f"{debug_root}/PCA_Axis_{axis_idx}")
                        curve.CreateTypeAttr("linear")
                        curve.CreateCurveVertexCountsAttr([2])
                        curve.CreateWidthsAttr([0.02])
                        curve.GetDisplayColorAttr().Set([Gf.Vec3f(*colors[axis_idx])])
                        a0 = Gf.Vec3f(0.0, 0.0, 0.0)
                        a1 = Gf.Vec3f(
                            float(scale * axes[axis_idx, 0]),
                            float(scale * axes[axis_idx, 1]),
                            float(scale * axes[axis_idx, 2]),
                        )
                        curve.GetPointsAttr().Set([a0, a1])

        self._update_palm_dir_overlay()

    def _log_obs_block(self, name: str, tensor: torch.Tensor | None) -> None:
        """Record summary statistics for each observation block."""

        if tensor is None or tensor.numel() == 0:
            return
        extras = getattr(self, "extras", None)
        if extras is None:
            extras = {"log": {}}
            self.extras = extras
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            return
        log_dict = extras.setdefault("log", {})
        mean_val = flat.mean().float().cpu()
        std_val = flat.std(unbiased=False).float().cpu()
        min_val = flat.min().float().cpu()
        max_val = flat.max().float().cpu()
        log_dict[f"obs/{name}/mean"] = mean_val
        log_dict[f"obs/{name}/std"] = std_val
        log_dict[f"obs/{name}/min"] = min_val
        log_dict[f"obs/{name}/max"] = max_val

        env_index = min(self._obs_table_env, tensor.shape[0] - 1)
        selected_row = tensor[env_index].reshape(-1)

        if getattr(self, "_print_obs_summary", False):
            print(f"[OBS] {name}: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
            if getattr(self, "_print_obs_entries", False):
                print(f"[OBS] {name} env {env_index}: {selected_row.detach().cpu().numpy()}")

        if self._enable_obs_table and self._obs_table_viewer is not None:
            labels = self._get_obs_labels(name, selected_row.numel())
            self._obs_table_viewer.update(name, selected_row, labels=labels)

    def _get_obs_labels(self, name: str, length: int) -> list[str]:
        """Generate human-readable labels for each observation entry."""

        def _linear(prefix: str, count: int) -> list[str]:
            return [f"{prefix}{i}" for i in range(count)]

        if name == "proprioception" and length == 167:
            labels = _linear("q", 22) + _linear("dq", 22) + _linear("tau", 22)
            finger_names = ["thumb", "index", "middle", "ring", "little"]
            for finger in finger_names:
                labels += [f"{finger}_pos_{axis}" for axis in "xyz"]
                labels += [f"{finger}_quat_{axis}" for axis in ["x", "y", "z", "w"]]
                labels += [f"{finger}_lin_{axis}" for axis in "xyz"]
                labels += [f"{finger}_ang_{axis}" for axis in "xyz"]
                labels += [f"{finger}_force_{axis}" for axis in "xyz"]
                labels += [f"{finger}_torque_{axis}" for axis in "xyz"]
            labels += ["p_x", "p_y", "p_z", "roll", "pitch", "yaw"]
            return labels[:length]
        if name == "previous_action" and length == 24:
            labels = [f"palm_trans_{axis}" for axis in "xyz"]
            labels += [f"palm_rot_{axis}" for axis in "xyz"]
            labels += _linear("joint_act_", length - len(labels))
            return labels
        if name == "object_state":
            labels = [f"obj_pos_{axis}" for axis in "xyz"]
            labels += [f"obj_quat_{axis}" for axis in ["x", "y", "z", "w"]]
            labels += [f"obj_linvel_{axis}" for axis in "xyz"]
            labels += [f"obj_angvel_{axis}" for axis in "xyz"]
            labels += [f"goal_vec_{axis}" for axis in "xyz"]
            return labels[:length]
        if name == "object_visual":
            return _linear("obj_feat_", length)
        if name == "time":
            labels = ["progress"]
            labels += _linear("time_enc_", length - 1)
            return labels
        if name == "hand_object_distance":
            return _linear("body_dist_", length)
        return _linear(f"{name}_", length)

    def _update_reward_table(self, reward_logs: dict[str, torch.Tensor]) -> None:
        if not (self._enable_reward_table and self._reward_table_viewer is not None):
            return
        env_idx = min(self._reward_table_env, self.num_envs - 1)
        rows: list[tuple[str, float]] = []
        for key in sorted(reward_logs.keys()):
            tensor = torch.as_tensor(reward_logs[key])
            if tensor.ndim == 0:
                value = float(tensor.item())
            else:
                idx = min(env_idx, tensor.shape[0] - 1)
                slice_tensor = tensor[idx]
                if slice_tensor.ndim == 0:
                    value = float(slice_tensor.item())
                else:
                    value = float(slice_tensor.reshape(-1)[0].item())
            rows.append((key, value))
        self._reward_table_viewer.update(rows)

    def _update_palm_dir_overlay(self) -> None:
        """Update the palm direction debug markers to follow the hand pose."""
        if not (self._palm_dir_curve_paths and self._palm_dir_center_apis):
            return
        try:
            import omni.usd  # type: ignore
            from pxr import Gf, UsdGeom  # type: ignore
        except Exception:
            return

        dir_local = self._palm_dir_local
        offset_local = self._palm_dir_offset_local
        if dir_local is None or offset_local is None:
            return

        dir_local = dir_local.unsqueeze(0).expand(self.num_envs, -1)
        offset_local = offset_local.unsqueeze(0).expand(self.num_envs, -1)
        hand_rot = self.hand.data.root_quat_w
        hand_pos = self.hand.data.root_pos_w
        dir_world = quat_apply(hand_rot, dir_local) * self._palm_dir_scale
        offset_world = quat_apply(hand_rot, offset_local)
        start_points = hand_pos + offset_world
        end_points = start_points + dir_world

        stage = omni.usd.get_context().get_stage()
        for env_idx, curve_path in enumerate(self._palm_dir_curve_paths):
            center_api = self._palm_dir_center_apis[env_idx]
            start = start_points[env_idx].detach().cpu().tolist()
            end = end_points[env_idx].detach().cpu().tolist()
            try:
                center_api.SetTranslate(Gf.Vec3f(float(start[0]), float(start[1]), float(start[2])))
            except Exception:
                pass
            curve = UsdGeom.BasisCurves.Get(stage, curve_path)
            if not curve:
                continue
            curve.GetPointsAttr().Set(
                [
                    Gf.Vec3f(float(start[0]), float(start[1]), float(start[2])),
                    Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
                ]
            )

    def _init_hand_body_samples(self) -> None:
        """Resolve body indices and offset specs for the 36-point hand distance features."""
        if not hasattr(self, "hand"):
            self._hand_joint_body_indices: list[int] = []
            self._hand_joint_label_lookup = {}
            self._hand_offset_specs = {}
            return

        indices: list[int] = []
        lookup: dict[str, int] = {}
        for label in _HAND_JOINT_LABELS:
            idxs, _ = self.hand.find_bodies(name_keys=[label], preserve_order=True)
            if not idxs:
                raise RuntimeError(
                    f"ShadowHand USD missing body containing '{label}'. "
                    "Ensure the converted shadow_hand_right.urdf preserves upstream link names."
                )
            indices.append(idxs[0])
            lookup[label] = len(indices) - 1

        offset_specs: dict[str, list[torch.Tensor]] = {"palm": list(_PALM_OFFSET_VECS)}
        for label in _DEFAULT_OFFSET_LABELS:
            offset_specs.setdefault(label, []).append(_DEFAULT_OFFSET_VECTOR)
        for label, vectors in _SPECIAL_OFFSET_VECS.items():
            offset_specs.setdefault(label, []).extend(vectors)
        total_offsets = sum(len(v) for v in offset_specs.values())
        if total_offsets != 19:
            raise RuntimeError(f"Hand offset spec must yield 19 points; currently {total_offsets}.")

        self._hand_joint_body_indices = indices
        self._hand_joint_label_lookup = lookup
        self._hand_offset_specs = offset_specs

    def _compute_hand_surface_points(self) -> torch.Tensor:
        """Compute 19 synthetic palm/finger points plus 17 joint centers (36 total)."""
        if not getattr(self, "_hand_joint_body_indices", None):
            raise RuntimeError("Hand body indices not initialized; call _init_hand_body_samples first.")
        body_quat = getattr(self.hand.data, "body_quat_w", None)
        if body_quat is None:
            raise RuntimeError("ShadowHand articulation does not expose body_quat_w.")
        body_pos = self.hand.data.body_pos_w
        joint_pos = body_pos[:, self._hand_joint_body_indices, :]
        joint_rot = body_quat[:, self._hand_joint_body_indices, :]
        num_envs = joint_pos.shape[0]
        samples: list[torch.Tensor] = []
        for label, vectors in self._hand_offset_specs.items():
            slot = self._hand_joint_label_lookup[label]
            base_pos = joint_pos[:, slot, :]
            base_rot = joint_rot[:, slot, :]
            for vec in vectors:
                offset = vec.to(device=base_pos.device, dtype=base_pos.dtype).view(1, 3).expand(num_envs, -1)
                rotated = quat_apply(base_rot, offset)
                samples.append(base_pos + rotated)
        if samples:
            synthetic = torch.stack(samples, dim=1)
        else:
            synthetic = torch.zeros(num_envs, 0, 3, device=joint_pos.device, dtype=joint_pos.dtype)
        return torch.cat([synthetic, joint_pos], dim=1)

    def compute_current_observations(self):
        """Build observations to match the original UniGraspTransformer layout with richer signals."""

        num_envs = self.num_envs
        device = self.device
        dtype = self.robot.data.joint_pos.dtype

        # Object pose and transforms
        if self.obj is not None:
            obj_pos = self.obj.data.root_pos_w
            obj_rot = self.obj.data.root_quat_w
            obj_lin_vel = getattr(self.obj.data, "root_lin_vel_w", torch.zeros_like(obj_pos))
            obj_ang_vel = getattr(self.obj.data, "root_ang_vel_w", torch.zeros_like(obj_pos))
        else:
            obj_pos = torch.zeros(num_envs, 3, device=device, dtype=dtype)
            obj_rot = torch.zeros(num_envs, 4, device=device, dtype=dtype)
            obj_rot[:, 3] = 1.0
            obj_lin_vel = torch.zeros_like(obj_pos)
            obj_ang_vel = torch.zeros_like(obj_pos)
        obj_conj = quat_conjugate(obj_rot)

        # Hand DOFs (66) – normalize around default pose when available
        q = self.robot.data.joint_pos
        dq = self.robot.data.joint_vel
        default_q = getattr(self.robot.data, "default_joint_pos", torch.zeros_like(q))
        hand_dof_pos = _pad_trunc(q - default_q, 22)
        hand_dof_vel = _pad_trunc(dq, 22)
        hand_dof_force = torch.zeros(num_envs, 22, device=device, dtype=dtype)
        hand_dofs = torch.cat([hand_dof_pos, hand_dof_vel, hand_dof_force], dim=-1)

        # Hand fingers (95) – encode object-frame kinematics plus placeholder forces
        tip_pos_w = self.hand.data.body_pos_w[:, self._tip_body_ids, :]
        tip_quat_attr = getattr(self.hand.data, "body_quat_w", None)
        if tip_quat_attr is not None:
            tip_quat_w = tip_quat_attr[:, self._tip_body_ids, :]
        else:
            tip_quat_w = torch.zeros(num_envs, self._num_tips, 4, device=device, dtype=dtype)
            tip_quat_w[..., 3] = 1.0
        tip_lin_attr = getattr(self.hand.data, "body_lin_vel_w", None)
        if tip_lin_attr is not None:
            tip_lin_vel_w = tip_lin_attr[:, self._tip_body_ids, :]
        else:
            tip_lin_vel_w = torch.zeros_like(tip_pos_w)
        tip_ang_attr = getattr(self.hand.data, "body_ang_vel_w", None)
        if tip_ang_attr is not None:
            tip_ang_vel_w = tip_ang_attr[:, self._tip_body_ids, :]
        else:
            tip_ang_vel_w = torch.zeros_like(tip_pos_w)

        tip_pos_obj = _world_to_object_vec(tip_pos_w, obj_pos.unsqueeze(1), obj_conj.unsqueeze(1))
        tip_quat_obj = _world_to_object_quat(tip_quat_w, obj_conj.unsqueeze(1))
        tip_lin_vel_obj = _world_to_object_vec(tip_lin_vel_w, torch.zeros_like(tip_lin_vel_w), obj_conj.unsqueeze(1))
        tip_ang_vel_obj = _world_to_object_vec(tip_ang_vel_w, torch.zeros_like(tip_ang_vel_w), obj_conj.unsqueeze(1))

        finger_kin = torch.cat([tip_pos_obj, tip_quat_obj, tip_lin_vel_obj, tip_ang_vel_obj], dim=-1)
        finger_kin = finger_kin.reshape(num_envs, -1)
        finger_ft = torch.zeros(num_envs, 30, device=device, dtype=dtype)
        hand_fingers = torch.cat([finger_kin, finger_ft], dim=-1)

        # Hand states (6) – palm pose in object frame
        hand_pos_w = self.hand.data.root_pos_w
        hand_rot_w = self.hand.data.root_quat_w
        hand_pos_obj = _world_to_object_vec(hand_pos_w, obj_pos, obj_conj)
        hand_rot_obj = _world_to_object_quat(hand_rot_w, obj_conj)
        hand_euler_obj = _quat_to_euler_xyz(hand_rot_obj)
        hand_states = torch.cat([hand_pos_obj, hand_euler_obj], dim=-1)
        proprioception = torch.cat([hand_dofs, hand_fingers, hand_states], dim=-1)
        self._log_obs_block("proprioception", proprioception)

        # Actions (24) – transform palm motion components into object frame
        last_act = getattr(self, "_last_actions", None)
        if last_act is None or last_act.numel() == 0:
            last_act = torch.zeros(num_envs, self.num_actions, device=device, dtype=dtype)
        act24 = _pad_trunc(last_act, 24)
        if act24.shape[1] >= 6:
            palm_trans = quat_apply(obj_conj, act24[:, :3])
            palm_rot = quat_apply(obj_conj, act24[:, 3:6])
            act24 = torch.cat([palm_trans, palm_rot, act24[:, 6:]], dim=-1)
        self._log_obs_block("previous_action", act24)

        # Object state (16) – match Table 1 definition (center, quat, velocities, goal delta)
        goal_height = self._default_table_surface + 0.20
        goal_pos = torch.tensor(self._default_object_pos, device=device, dtype=dtype).unsqueeze(0).repeat(num_envs, 1)
        goal_pos[:, 2] = goal_height
        goal_vec_obj = _world_to_object_vec(goal_pos, obj_pos, obj_conj)
        obj_lin_vel_obj = quat_apply(obj_conj, obj_lin_vel)
        obj_ang_vel_obj = quat_apply(obj_conj, obj_ang_vel)
        object_state = torch.cat(
            [
                torch.zeros(num_envs, 3, device=device, dtype=dtype),
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).expand(num_envs, 4),
                obj_lin_vel_obj,
                obj_ang_vel_obj,
                goal_vec_obj,
            ],
            dim=-1,
        )
        self._log_obs_block("object_state", object_state)

        # Object visual (128) – use dataset embedding if available, else zeros
        object_points_world = None
        if self._object_pc_local is not None:
            pc_local = self._object_pc_local.to(device, dtype)
            if pc_local.dim() == 2:
                pc_local = pc_local.unsqueeze(0).repeat(num_envs, 1, 1)
            num_points = pc_local.shape[1]
            quat_expanded = obj_rot.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4)
            points_flat = pc_local.reshape(-1, 3)
            rotated = quat_apply(quat_expanded, points_flat).reshape(num_envs, num_points, 3)
            object_points_world = rotated + obj_pos.unsqueeze(1)
            object_centers = torch.mean(object_points_world, dim=1)
            centered = object_points_world - object_centers.unsqueeze(1)
            max_points = min(centered.shape[1], 64)
            flat = centered[:, :max_points].reshape(num_envs, -1)
            object_visual = torch.zeros(num_envs, 128, device=device, dtype=dtype)
            take = min(object_visual.shape[1], flat.shape[1])
            object_visual[:, :take] = flat[:, :take]
        else:
            object_visual = torch.zeros(num_envs, 128, device=device, dtype=dtype)
        self._log_obs_block("object_visual", object_visual)

        # Times (29) – normalized progress + positional encoding
        progress_steps = self.episode_length_buf.to(device=device, dtype=torch.float32)
        time_encoding = _positional_encoding(progress_steps, 28).to(dtype)
        times = torch.cat([progress_steps.unsqueeze(-1).to(dtype), time_encoding], dim=-1)
        self._log_obs_block("time", times)

        # Hand-object distances (36) – min distances from canonical hand points to object cloud
        if object_points_world is not None:
            hand_points = self._compute_hand_surface_points()
            dists = torch.cdist(hand_points, object_points_world).min(dim=-1).values
            hand_object_features = dists
        else:
            hand_object_features = torch.zeros(num_envs, 36, device=device, dtype=dtype)
        self._log_obs_block("hand_object_distance", hand_object_features)

        actor_obs = torch.cat(
            [proprioception, act24, object_state, hand_object_features, times, object_visual],
            dim=-1,
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
        # Reset object pose from YAML-configured defaults or dataset init states when present.
        if len(env_ids) > 0 and getattr(self, "obj", None) is not None:
            ids = env_ids.to(dtype=torch.long, device=self.device)
            try:
                if self._object_init_np is not None:
                    import numpy as _np
                    states = self._object_init_np.get("test") or self._object_init_np.get("train")
                    if states is not None and len(states) > 0:
                        picks = _np.random.randint(0, len(states), size=int(ids.shape[0]))
                        sel = torch.from_numpy(states[picks]).to(device=self.device, dtype=self.robot.data.root_pos_w.dtype)
                        obj_pose = self.obj.data.root_state_w.clone()
                        obj_pose[ids, :3] = self.scene.env_origins[ids] + sel[:, :3]
                        obj_pose[ids, 3:7] = sel[:, 3:7]
                        self.obj.write_root_pose_to_sim(obj_pose[ids, :7], env_ids=ids)
                        zero_vel = torch.zeros((ids.numel(), 6), device=self.device, dtype=obj_pose.dtype)
                        self.obj.write_root_velocity_to_sim(zero_vel, env_ids=ids)
                else:
                    obj_pose = self.obj.data.root_state_w.clone()
                    obj_pose[ids, :3] = origins[ids] + torch.tensor(self._default_object_pos, dtype=obj_pose.dtype, device=self.device)
                    obj_pose[ids, 3:7] = torch.tensor(self._default_object_rot, dtype=obj_pose.dtype, device=self.device)
                    self.obj.write_root_pose_to_sim(obj_pose[ids, :7], env_ids=ids)
                    zero_vel = torch.zeros((ids.numel(), 6), device=self.device, dtype=obj_pose.dtype)
                    self.obj.write_root_velocity_to_sim(zero_vel, env_ids=ids)
            except Exception:
                pass
        # Optionally warp the hand back to YAML-configured default pose
        if getattr(self.cfg.scene.spawn.hand, "warp_on_reset", True):
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
        self._update_reward_table(reward_logs)
        self._update_palm_dir_overlay()

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
