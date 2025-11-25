import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

from legged_lab.envs.base.my_confg import GraspObjectCfg, TableCfg
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from legged_lab.envs.base.my_confg import GraspObjectCfg, TableCfg

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
    meta["metadata_path"] = str(meta_path)
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
        "path_missing",
    ]
    if not isinstance(obj_spawn, dict):
        raise ValueError("obj_spawn must be a dict-like object.")
    
    missing = [key for key in required_keys if key not in obj_spawn]
    if missing:
        raise ValueError(f"obj_spawn is missing keys: {missing}. It should match the structure returned by _build_object_spawn.")
    
    # return false if object spawning disabled
    if not obj_spawn.get("enable", False):
        return GraspObjectCfg(enable=False)

    default_dir = obj_spawn.get("default_dir")
    object_path = obj_spawn.get("object_path")
    pc_fps_path = obj_spawn.get("pc_fps_path") or obj_spawn.get("pc_fps")
    pca_axes_path = obj_spawn.get("pca_axes_path") or obj_spawn.get("pca_axes")
    object_init = obj_spawn.get("object_init")
    metadata_path = obj_spawn.get("metadata_path")
    show_point_cloud = bool(obj_spawn.get("show_point_cloud", False))
    show_pca_axes = bool(obj_spawn.get("show_pca_axes", False))

    # If an explicit object_path exists, try loading adjacent metadata.json to fill optional fields.
    if object_path and not obj_spawn.get("path_missing", False):
        meta = _load_metadata_from_object_path(object_path)
        if meta:
            object_path = sampled.get("static_usd")
            pc_fps_path = meta.get("pc_fps_path") or meta.get("pc_fps")
            pca_axes_path = meta.get("pca_axes_path") or meta.get("pca_axes")
            object_init = meta.get("object_init")
            metadata_path = meta.get("metadata_path")

    # Sample a random object if path is missing and a default_dir is provided.
    if obj_spawn.get("path_missing", False):
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
        size=tuple(obj_spawn.get("size") or (0.1, 0.1, 0.1)),
        pos=tuple(obj_spawn.get("pos") or (0.0, 0.0, 0.5)),
        rot_xyzw=tuple(obj_spawn.get("rot_xyzw") or (0.0, 0.0, 0.0, 1.0)),
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
