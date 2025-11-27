import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from legged_lab.envs.base.my_confg import GraspObjectCfg, TableCfg
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import quat_apply
from legged_lab.envs.base.my_confg import GraspObjectCfg, TableCfg
import numpy as np
import omni.usd
from pxr import UsdGeom, Gf, Usd


def compute_time_encoding(time: torch.Tensor, dimension: int) -> torch.Tensor:
    """Sinusoidal time/positional encoding (Transformer style).

    Args:
        time: Tensor of shape (N,) with time steps / progress values.
        dimension: Output embedding dimension (even number recommended).

    Returns:
        Tensor of shape (N, dimension) with sin/cos encodings.
    """
    div_term = torch.arange(0, dimension, 2, dtype=torch.float32, device=time.device)
    div_term = torch.exp(div_term * -(torch.log(torch.tensor(10000.0, device=time.device)) / dimension)).unsqueeze(0)
    encoding = torch.zeros(time.shape[0], dimension, device=time.device)
    encoding[:, 0::2] = torch.sin(time.unsqueeze(1) * div_term)
    encoding[:, 1::2] = torch.cos(time.unsqueeze(1) * div_term)
    return encoding


def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to Euler XYZ angles.

    Args:
        quat: (..., 4) tensor in xyzw order.

    Returns:
        (..., 3) tensor of Euler angles (roll, pitch, yaw).
    """
    x, y, z, w = quat.unbind(-1)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2_clamped = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2_clamped)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack((roll, pitch, yaw), dim=-1)


def batch_sided_distance(sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute sided distance from sources (Nenv, Ns, 3) to nearest target (Nenv, Nt, 3)."""
    pairwise_distances = torch.cdist(sources, targets)
    distances, _ = torch.min(pairwise_distances, dim=-1)
    return distances


def compute_hand_body_pos(hand_joint_pos: torch.Tensor, hand_joint_rot: torch.Tensor) -> torch.Tensor:
    """Compute hand body points from joint positions/rotations (matches UniGrasp offset scheme)."""
    device = hand_joint_pos.device
    num_envs = hand_joint_pos.shape[0]
    hand_body_pos = []
    for n in range(hand_joint_rot.shape[1]):
        if n in [2, 5, 8, 12]:
            continue
        elif n == 0:
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], torch.tensor([1, 0, 0], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], torch.tensor([-1, 0, 0], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)
            
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 0, 1], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 0, 1], device=device).repeat(num_envs, 1) * 0.06) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)

            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], torch.tensor([1, 0, 0], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 0, 1], device=device).repeat(num_envs, 1) * 0.06) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)

            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], torch.tensor([1, 0, 0], device=device).repeat(num_envs, 1) * 0.015) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 0, 1], device=device).repeat(num_envs, 1) * 0.015) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)

            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], torch.tensor([-1, 0, 0], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 0, 1], device=device).repeat(num_envs, 1) * 0.03) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 1, 0], device=device).repeat(num_envs, 1) * -0.005)
            hand_body_pos.append(body_pos)

        elif n == 10:
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 0, 1], device=device).repeat(num_envs, 1) * 0.02) \
                + quat_apply(hand_joint_rot[:, n, :], torch.tensor([-1, 0, 0], device=device).repeat(num_envs, 1) * 0.015)
            hand_body_pos.append(body_pos)
        else:
            body_pos = hand_joint_pos[:, n, :] + quat_apply(hand_joint_rot[:, n, :], torch.tensor([0, 0, 1], device=device).repeat(num_envs, 1) * 0.02)
            hand_body_pos.append(body_pos)
    
    hand_body_pos = torch.stack(hand_body_pos, dim=1)
    hand_body_pos = torch.cat([hand_body_pos, hand_joint_pos], dim=1)
    return hand_body_pos


def get_point_cloud_world(env_index: int = 0, prim_suffix: str = "ObjectPC") -> np.ndarray:
    """Fetch the spawned point cloud overlay for an env in world coordinates."""
    stage = omni.usd.get_context().get_stage()
    prim_path = f"/World/envs/env_{env_index}/Object/{prim_suffix}"
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.zeros((0, 3), dtype=np.float32)
    pc = UsdGeom.Points(prim)
    pts_local = pc.GetPointsAttr().Get()
    if not pts_local:
        return np.zeros((0, 3), dtype=np.float32)
    xf = pc.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pts_world = [xf.Transform(p) for p in pts_local]
    return np.array([[p[0], p[1], p[2]] for p in pts_world], dtype=np.float32)

def _load_yaml_cfg(filename: str) -> Dict[str, Any]:
    # load hyperparameters from yaml
    cfg_path = Path(__file__).resolve().parent / "cfg" / filename
    try:
        import yaml
    except ImportError:
        return {}
    if not cfg_path.is_file():
        return {}
    content = cfg_path.read_text(encoding="utf-8")
    if not content.strip():
        return {}
    try:
        loaded = yaml.safe_load(content)
    except Exception:
        return {}
    return loaded or {}

def _build_table_spawn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # isolate table related hyperparameters from SPAWN_CFG
    if not isinstance(cfg, dict):
        raise ValueError("spawn_cfg must be a dict-like object.")
    table_cfg = cfg.get("table", {})
    if not isinstance(table_cfg, dict):
        raise ValueError("spawn_cfg.yaml must contain a 'table' mapping.")
    required_keys = ["enable", "size", "pos", "rot_xyzw"]
    missing = [key for key in required_keys if key not in table_cfg]
    if missing:
        raise ValueError(f"spawn_cfg.yaml 'table' section missing keys: {missing}. Did you rename them?")
    return {
        "enable": bool(table_cfg.get("enable", False)),
        "size": table_cfg.get("size"),
        "pos": table_cfg.get("pos"),
        "rot_xyzw": table_cfg.get("rot_xyzw"),
    }

def _build_hand_spawn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # isolate hand related hyperparameters from SPAWN_CFG
    if not isinstance(cfg, dict):
        raise ValueError("spawn_cfg must be a dict-like object.")
    hand_cfg = cfg.get("hand", {})
    if not isinstance(hand_cfg, dict):
        raise ValueError("spawn_cfg.yaml must contain a 'hand' mapping.")
    required_keys = [
        "asset_type",
        "asset_path",
        "pos",
        "rot_xyzw",
        # "show_palm_dir",
        # "palm_dir_local",
        # "palm_dir_offset_local",
        # "palm_dir_scale",
    ]
    missing = [key for key in required_keys if key not in hand_cfg]
    if missing:
        raise ValueError(f"spawn_cfg.yaml 'hand' section missing keys: {missing}. Did you rename them?")
    return hand_cfg

def _pick_random_object_from_dir(default_dir: str) -> Dict[str, Any]:
    """Pick a random metadata.json two levels under default_dir and return its contents."""
    dir_path = Path(default_dir).expanduser()
    
    if not dir_path.is_dir():
        raise ValueError(f"default_dir '{default_dir}' is not a valid directory for grasp object selection")
    candidates = list(dir_path.glob("*/*/metadata.json"))
    
    if not candidates:
        raise ValueError(f"default_dir '{default_dir}' contains no files to sample grasp objects from")
    meta_path = random.choice(candidates)
    
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        raise ValueError(f"Failed to read metadata at {meta_path}: {exc}") from exc
    
    meta["metadata_path"] = str(meta_path)
    return meta

def _load_metadata_from_object_path(object_path: str) -> Dict[str, Any]:
    """Load metadata.json sitting next to a specified object usd path, if present."""
    obj_path = Path(object_path).expanduser()
    meta_path = obj_path.parent / "metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        raise ValueError(f"Failed to read metadata at {meta_path}: {exc}") from exc
    meta["metadata_path"] = str(meta_path) # meta_path is a domain of meta dic
    return meta

def _build_object_spawn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # isolate object related hyperparameters from SPAWN_CFG
    if not isinstance(cfg, dict):
        raise ValueError("spawn_cfg must be a dict-like object.")
    obj_cfg = cfg.get("object", {})
    if not isinstance(obj_cfg, dict):
        raise ValueError("spawn_cfg.yaml must contain an 'object' mapping.")
    required_keys = [
        "enable",
        "default_dir",
        "object_path",
        "size",
        "pos",
        "rot_xyzw",
        "show_point_cloud",
        "show_pca_axes",
        "object_init",
    ]
    missing = [key for key in required_keys if key not in obj_cfg]
    if missing:
        raise ValueError(f"spawn_cfg.yaml 'object' section missing keys: {missing}. Did you rename them?")
    
    enable = obj_cfg.get("enable")
    default_dir = obj_cfg.get("default_dir")
    object_path = obj_cfg.get("object_path")
    size = obj_cfg.get("size")
    pos = obj_cfg.get("pos")
    rot_xyzw = obj_cfg.get("rot_xyzw")
    show_point_cloud = bool(obj_cfg.get("show_point_cloud", False))
    show_pca_axes = bool(obj_cfg.get("show_pca_axes", False))
    object_init = obj_cfg.get("object_init")

    #object_path missing flag
    path_missing = object_path is None or (isinstance(object_path, str) and len(object_path.strip()) == 0)
    return {
        "enable": enable,
        "default_dir": default_dir,
        "object_path": object_path,
        "show_point_cloud": show_point_cloud,
        "show_pca_axes": show_pca_axes,
        "size": size,
        "pos": pos,
        "rot_xyzw": rot_xyzw,
        "object_init": object_init,
        "path_missing":path_missing
    }

def _build_table_cfg(table_spawn: Dict[str, Any]) -> Optional[TableCfg]:
    # instantiate table config from TableCfg
    if not table_spawn.get("enable", False):
        return TableCfg(enable=False)
    return TableCfg(
        enable=True,
        size=tuple(table_spawn.get("size") or (0.6, 0.6, 0.03)),
        pos=tuple(table_spawn.get("pos") or (0.0, 0.0, 0.25)),
        rot_xyzw=tuple(table_spawn.get("rot_xyzw") or (0.0, 0.0, 0.0, 1.0)),
    )

def _build_grasp_object_cfg(obj_spawn: Dict[str, Any]) -> Optional[GraspObjectCfg]:
    """Instantiate GraspObjectCfg from spawn dict, including random sampling fallback and overlay validation."""
    required_keys = [
        "enable",
        "default_dir",
        "object_path",
        "show_point_cloud",
        "show_pca_axes",
        "size",
        "pos",
        "rot_xyzw",
        "object_init",
        "path_missing", # the flag to indicate tha the object_path is not given
    ]
    if not isinstance(obj_spawn, dict):
        raise ValueError("obj_spawn must be a dict-like object.")
    
    missing = [key for key in required_keys if key not in obj_spawn]
    if missing:
        raise ValueError(f"obj_spawn is missing keys: {missing}. It should match the structure returned by _build_object_spawn.")
    
    # unpacking upsteaming config domains
    # return false if object spawning disabled
    if not obj_spawn.get("enable", False):
        return GraspObjectCfg(enable=False)

    default_dir = obj_spawn.get("default_dir")
    object_path = obj_spawn.get("object_path") # can be explicitly given, if not, then random choose one object
    size = obj_spawn.get("size")
    pos = obj_spawn.get("pos")
    rot_xyzw = obj_spawn.get("rot_xyzw")
    object_init = obj_spawn.get("object_init")
    path_missing = obj_spawn.get("path_missing")
    show_point_cloud = bool(obj_spawn.get("show_point_cloud", False))
    show_pca_axes = bool(obj_spawn.get("show_pca_axes", False))

    # If an explicit object_path exists, try loading adjacent metadata.json to fill optional fields.
    if object_path and not path_missing:
        # load metadata
        meta = _load_metadata_from_object_path(object_path)
        if meta:
            # extend and override the cfg domains with metadata info
            object_path = meta.get("static_usd", object_path)
            pc_fps_path = meta.get("pc_fps")
            pca_axes_path = meta.get("pca_axes")
            object_init = meta.get("object_init", object_init)
            metadata_path = meta.get("metadata_path", metadata_path) # the meta_path is added to the meta dic in the upstream helper
        else:
            raise ValueError("Something went wrong when parsing metadata.json when extending object spawning cfg")

    # Sample a random object if path is missing and a default_dir is provided.
    if path_missing:
        if default_dir is None or (isinstance(default_dir, str) and len(default_dir.strip()) == 0):
            raise ValueError("object_path is missing and no default_dir provided for grasp object.")
        sampled = _pick_random_object_from_dir(default_dir)
        # print("Sampling object successful")
        # print(sampled)
        # input()
        object_path = sampled.get("static_usd")
        pc_fps_path = sampled.get("pc_fps")
        pca_axes_path = sampled.get("pca_axes")
        object_init = sampled.get("object_init")
        metadata_path = sampled.get("metadata_path")

    # validate if object path is finally get.
    if not object_path:
        raise ValueError("Something happens when reading object metadata.")

    # Validate overlays: if requested, paths must be provided.
    if show_point_cloud and not pc_fps_path:
        raise ValueError("show_point_cloud is True but pc_fps_path is missing (metadata).")
    if show_pca_axes and not pca_axes_path:
        raise ValueError("show_pca_axes is True but pca_axes_path is missing (metadata).")

    return GraspObjectCfg(
        # general
        enable=True,
        default_dir=default_dir,
        object_path=object_path,
        size=size,
        pos=pos,
        rot_xyzw=rot_xyzw,
        object_init=object_init,
        metadata_path=metadata_path,

        # point cloud related
        show_point_cloud=show_point_cloud,
        pc_fps_path=pc_fps_path,

        # pca related
        show_pca_axes=show_pca_axes,
        pca_axes_path=pca_axes_path,
    )


def _build_hand_cfg(hand_spawn: Dict[str, Any], hand_cfg: ArticulationCfg) -> ArticulationCfg:
    """Convenience wrapper to override hand cfg from spawn dict."""
    if not isinstance(hand_spawn, dict):
        return hand_cfg
    hand_pos = tuple(hand_spawn.get("pos"))
    hand_rot = tuple(hand_spawn.get("rot_xyzw"))
    hand_cfg.init_state.pos = hand_pos
    hand_cfg.init_state.rot = hand_rot
    # print("override default hand spawning cfg")
    # print(hand_cfg)
    # input()
    return hand_cfg

def get_point_cloud_world(env_index: int = 0, prim_suffix: str = "ObjectPC") -> np.ndarray:
    """Fetch the point cloud overlay for an env in world coordinates.

    Args:
        env_index: Which environment index to read from (default: 0).
        prim_suffix: Name of the point cloud prim under the object (default: ``ObjectPC``).

    Returns:
        An (N, 3) numpy array of points in world frame. If the prim is missing, returns an empty array.
    """
    import omni.usd
    from pxr import UsdGeom, Usd

    stage = omni.usd.get_context().get_stage()
    prim_path = f"/World/envs/env_{env_index}/Object/{prim_suffix}"
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.zeros((0, 3), dtype=np.float32)

    pc = UsdGeom.Points(prim)
    pts_local = pc.GetPointsAttr().Get()
    if not pts_local:
        return np.zeros((0, 3), dtype=np.float32)

    xf = pc.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pts_world = [xf.Transform(p) for p in pts_local]
    return np.array([[p[0], p[1], p[2]] for p in pts_world], dtype=np.float32)
