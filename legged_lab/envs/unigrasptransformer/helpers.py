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
    table_cfg = cfg.get("table", {}) if isinstance(cfg, dict) else {}
    return {
        "enable": bool(table_cfg.get("enable", False)),
        "size": table_cfg.get("size"),
        "pos": table_cfg.get("pos"),
        "rot_xyzw": table_cfg.get("rot_xyzw"),
    }

def _build_hand_spawn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # isolate hand related hyperparameters from SPAWN_CFG
    hand_cfg = cfg.get("hand", {}) if isinstance(cfg, dict) else {}
    return hand_cfg if isinstance(hand_cfg, dict) else {}

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

def _build_object_spawn(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # isolate object related hyperparameters from SPAWN_CFG
    obj_cfg = cfg.get("object", {}) if isinstance(cfg, dict) else {}
    default_dir = obj_cfg.get("default_dir")
    object_path = obj_cfg.get("object_path")
    pc_fps = obj_cfg.get("pc_fps")
    pca_axes = obj_cfg.get("pca_axes")
    object_init = obj_cfg.get("object_init")
    metadata_path = obj_cfg.get("metadata_path")

    path_missing = object_path is None or (isinstance(object_path, str) and len(object_path.strip()) == 0)
    sampled_meta: Dict[str, Any] = {}
    if path_missing:
        if default_dir is None or (isinstance(default_dir, str) and len(default_dir.strip()) == 0):
            raise ValueError("object_path is missing and no default_dir provided in spawn_cfg.yaml")
        try:
            sampled_meta = _pick_random_object_from_dir(default_dir)
            object_path = sampled_meta.get("static_usd") or sampled_meta.get("object_path")
            pc_fps = pc_fps or sampled_meta.get("pc_fps")
            pca_axes = pca_axes or sampled_meta.get("pca_axes")
            object_init = object_init or sampled_meta.get("object_init")
            metadata_path = metadata_path or sampled_meta.get("metadata_path")
        except ValueError as exc:
            print(f"[WARN] {exc}. Disabling object spawn for this run.")
            obj_cfg["enable"] = False
            object_path = None
    return {
        "enable": bool(obj_cfg.get("enable", False) and object_path is not None),
        "default_dir": default_dir,
        "object_path": object_path,
        "show_point_cloud": bool(obj_cfg.get("show_point_cloud", False)),
        "show_pca_axes": bool(obj_cfg.get("show_pca_axes", False)),
        "size": obj_cfg.get("size"),
        "pos": obj_cfg.get("pos"),
        "rot_xyzw": obj_cfg.get("rot_xyzw"),
        "object_init": object_init,
        "pc_fps": pc_fps,
        "pca_axes": pca_axes,
        "metadata_path": metadata_path,
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
    """Instantiate GraspObjectCfg from spawn dict, requiring a USD path and optional point cloud/PCA overlays via explicit cfg paths."""
    if not obj_spawn.get("enable", False):
        return GraspObjectCfg(enable=False)
    object_path = obj_spawn.get("object_path")
    if not object_path:
        default_dir = obj_spawn.get("default_dir")
        if default_dir:
            sampled = _pick_random_object_from_dir(default_dir)
            object_path = sampled.get("static_usd") or sampled.get("object_path")
            # propagate optional metadata
            obj_spawn = {
                **obj_spawn,
                "object_path": object_path,
                "pc_fps_path": obj_spawn.get("pc_fps_path") or sampled.get("pc_fps_path"),
                "pca_axes_path": obj_spawn.get("pca_axes_path") or sampled.get("pca_axes_path"),
                "object_init": obj_spawn.get("object_init") or sampled.get("object_init"),
                "metadata_path": obj_spawn.get("metadata_path") or sampled.get("metadata_path"),
            }
        if not object_path:
            raise ValueError("grasp object must specify a USD path (object_path), and default_dir sampling failed.")

    return GraspObjectCfg(
        # general 
        enable=True,
        default_dir=obj_spawn.get("default_dir"),
        object_path=object_path,
        size=tuple(obj_spawn.get("size") or (0.1, 0.1, 0.1)),
        pos=tuple(obj_spawn.get("pos") or (0.0, 0.0, 0.5)),
        rot_xyzw=tuple(obj_spawn.get("rot_xyzw") or (0.0, 0.0, 0.0, 1.0)),
        object_init=obj_spawn.get("object_init"),
        metadata_path=obj_spawn.get("metadata_path"),

        # point cloud related
        show_point_cloud=bool(obj_spawn.get("show_point_cloud", False)),
        pc_fps_path=obj_spawn.get("pc_fps_path"),
        
        # pca related
        show_pca_axes=bool(obj_spawn.get("show_pca_axes", False)),
        pca_axes_path=obj_spawn.get("pca_axes_path"),
        
    )
