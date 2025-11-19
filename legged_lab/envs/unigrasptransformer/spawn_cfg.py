"""UniGraspTransformer scene spawn configuration (standalone).

This module defines the table/object/hand spawn configuration classes and a
YAML loader for object overrides, without inheriting from the GraspXL variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from isaaclab.utils import configclass

from .logging_utils import log_debug

_DEFAULT_CFG_PATH = Path(__file__).with_name("cfg").joinpath("config.yaml").resolve()


@configclass
class UniGraspTransformerTableSpawnCfg:
    """Table geometry, material, and placement defaults."""

    enable: bool | None = None
    size: tuple[float, float, float] | None = None
    pos: tuple[float, float, float] | None = None
    rot: tuple[float, float, float, float] | None = None
    color: tuple[float, float, float] | None = None
    metallic: float | None = None
    roughness: float | None = None
    disable_gravity: bool | None = None

    def __post_init__(self):
        log_debug(f"TableSpawnCfg ready (enable={self.enable}, size={self.size})")


@configclass
class UniGraspTransformerObjectSpawnCfg:
    """Spawn description for the grasp object placeholder."""

    enable: bool | None = None
    size: tuple[float, float, float] | None = None
    pos: tuple[float, float, float] | None = None
    rot: tuple[float, float, float, float] | None = None
    mass: float | None = None
    disable_gravity: bool | None = None
    color: tuple[float, float, float] | None = None
    metallic: float | None = None
    roughness: float | None = None
    asset_prim_name: str = "Object"
    object_id: Optional[str] = None
    static_usd: Optional[str] = None
    spawn_mesh: bool | None = None
    show_point_cloud: bool | None = None
    show_pca_axes: bool | None = None
    pc_fps: Optional[str] = None
    pca_axes: Optional[str] = None
    object_init: Optional[str] = None

    def __post_init__(self):
        log_debug(
            f"ObjectSpawnCfg ready (static_usd={self.static_usd}, enable={self.enable})"
        )


@configclass
class UniGraspTransformerHandSpawnCfg:
    """Root pose defaults for the Inspire Hand articulation."""

    asset_type: str | None = None
    asset_path: str | None = None
    pos: tuple[float, float, float] | None = None
    orientation_xyzw: tuple[float, float, float, float] | None = None
    disable_gravity: bool | None = None
    show_palm_dir: bool | None = None
    palm_dir_local: tuple[float, float, float] | None = None
    palm_dir_offset_local: tuple[float, float, float] | None = None
    palm_dir_scale: float | None = None
    fingertip_body_exprs: tuple[str, ...] | None = None
    warp_on_reset: bool | None = None

    def __post_init__(self):
        log_debug("HandSpawnCfg ready (fixed base pose)")


@configclass
class UniGraspTransformerSpawnCfg:
    """Aggregate spawn configuration bundling table, object, and hand."""

    table: UniGraspTransformerTableSpawnCfg = UniGraspTransformerTableSpawnCfg()
    grasp_object: UniGraspTransformerObjectSpawnCfg = UniGraspTransformerObjectSpawnCfg()
    hand: UniGraspTransformerHandSpawnCfg = UniGraspTransformerHandSpawnCfg()
    config_path: Optional[str] = str(_DEFAULT_CFG_PATH)
    use_object_library: bool = True

    def __post_init__(self):
        log_debug(f"SpawnCfg ready (config_path={self.config_path})")
        cfg_path = self.config_path or str(_DEFAULT_CFG_PATH)
        yaml_path = Path(cfg_path).expanduser()
        if not yaml_path.exists():
            raise FileNotFoundError(f"UniGraspTransformer spawn config not found: {cfg_path}")
        self.config_path = yaml_path.as_posix()
        load_unigrasp_config(self, yaml_path)
        _maybe_pick_random_dataset_object(self)


def _expand_path(path: str | Path | None) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


__all__ = [
    "UniGraspTransformerTableSpawnCfg",
    "UniGraspTransformerObjectSpawnCfg",
    "UniGraspTransformerHandSpawnCfg",
    "UniGraspTransformerSpawnCfg",
]


def load_unigrasp_config(spawn_cfg: UniGraspTransformerSpawnCfg, yaml_path: Path) -> None:
    """Load unified config.yaml and populate spawn_cfg (table/object/hand)."""
    import yaml

    if not yaml_path.exists():
        raise FileNotFoundError(f"Unified config not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text()) or {}
    ucfg = data.get("unigrasptransformer") or data

    def _as_tuple(seq, n):
        if seq is None:
            return None
        vals = tuple(float(x) for x in seq)
        if len(vals) != n:
            raise ValueError(f"Expected {n} elements, got {len(vals)}: {seq}")
        return vals

    # Table
    t = ucfg.get("table", {})
    spawn_cfg.table.enable = bool(t.get("enable", spawn_cfg.table.enable))
    spawn_cfg.table.size = _as_tuple(t.get("size", spawn_cfg.table.size), 3) or spawn_cfg.table.size
    spawn_cfg.table.pos = _as_tuple(t.get("pos", spawn_cfg.table.pos), 3) or spawn_cfg.table.pos
    spawn_cfg.table.rot = _as_tuple(t.get("rot_xyzw", spawn_cfg.table.rot), 4) or spawn_cfg.table.rot

    # Object
    o = ucfg.get("object", {})
    spawn_cfg.grasp_object.enable = bool(o.get("enable", spawn_cfg.grasp_object.enable))
    spawn_cfg.grasp_object.spawn_mesh = bool(o.get("spawn_mesh", spawn_cfg.grasp_object.spawn_mesh))
    spawn_cfg.grasp_object.show_point_cloud = bool(o.get("show_point_cloud", spawn_cfg.grasp_object.show_point_cloud))
    spawn_cfg.grasp_object.show_pca_axes = bool(o.get("show_pca_axes", spawn_cfg.grasp_object.show_pca_axes))
    spawn_cfg.grasp_object.size = _as_tuple(o.get("size", spawn_cfg.grasp_object.size), 3) or spawn_cfg.grasp_object.size
    spawn_cfg.grasp_object.pos = _as_tuple(o.get("pos", spawn_cfg.grasp_object.pos), 3) or spawn_cfg.grasp_object.pos
    spawn_cfg.grasp_object.rot = _as_tuple(o.get("rot_xyzw", spawn_cfg.grasp_object.rot), 4) or spawn_cfg.grasp_object.rot
    static_usd = o.get("static_usd", None)
    spawn_cfg.grasp_object.static_usd = static_usd if static_usd else spawn_cfg.grasp_object.static_usd
    spawn_cfg.grasp_object.pc_fps = o.get("pc_fps", spawn_cfg.grasp_object.pc_fps)
    spawn_cfg.grasp_object.pca_axes = o.get("pca_axes", spawn_cfg.grasp_object.pca_axes)
    spawn_cfg.grasp_object.object_init = o.get("object_init", spawn_cfg.grasp_object.object_init)

    # Enforce coherence: when object is disabled, all dependent flags and assets are cleared
    if not spawn_cfg.grasp_object.enable:
        spawn_cfg.grasp_object.spawn_mesh = False
        spawn_cfg.grasp_object.show_point_cloud = False
        spawn_cfg.grasp_object.show_pca_axes = False
        spawn_cfg.grasp_object.static_usd = None
        spawn_cfg.grasp_object.pc_fps = None
        spawn_cfg.grasp_object.pca_axes = None
        spawn_cfg.grasp_object.object_init = None
        spawn_cfg.grasp_object.object_id = None

    # Hand
    h = ucfg.get("hand", {})
    spawn_cfg.hand.asset_type = h.get("asset_type", spawn_cfg.hand.asset_type)
    spawn_cfg.hand.asset_path = h.get("asset_path", spawn_cfg.hand.asset_path)
    spawn_cfg.hand.pos = _as_tuple(h.get("pos", spawn_cfg.hand.pos), 3) or spawn_cfg.hand.pos
    spawn_cfg.hand.orientation_xyzw = _as_tuple(h.get("rot_xyzw", spawn_cfg.hand.orientation_xyzw), 4) or spawn_cfg.hand.orientation_xyzw
    spawn_cfg.hand.disable_gravity = h.get("disable_gravity", spawn_cfg.hand.disable_gravity)
    spawn_cfg.hand.show_palm_dir = h.get("show_palm_dir", spawn_cfg.hand.show_palm_dir)
    palm_dir = h.get("palm_dir_local")
    if palm_dir is not None:
        spawn_cfg.hand.palm_dir_local = _as_tuple(palm_dir, 3)
    palm_offset = h.get("palm_dir_offset_local")
    if palm_offset is not None:
        spawn_cfg.hand.palm_dir_offset_local = _as_tuple(palm_offset, 3)
    palm_scale = h.get("palm_dir_scale")
    if palm_scale is not None:
        spawn_cfg.hand.palm_dir_scale = float(palm_scale)
    spawn_cfg.hand.warp_on_reset = h.get("warp_on_reset", spawn_cfg.hand.warp_on_reset)
    exprs = h.get("fingertip_body_exprs")
    if exprs is not None:
        spawn_cfg.hand.fingertip_body_exprs = tuple(exprs)

    log_debug("Unified config loaded from %s" % yaml_path)


def _maybe_pick_random_dataset_object(spawn_cfg: UniGraspTransformerSpawnCfg) -> None:
    """Pick a random dataset object if mesh spawning is requested but no USD path provided."""
    if not (spawn_cfg.grasp_object.enable and spawn_cfg.grasp_object.spawn_mesh):
        return
    if getattr(spawn_cfg.grasp_object, "static_usd", None):
        return

    import json
    import os
    import random

    subset_root_env = os.environ.get(
        "UGTF_SUBSET_ROOT",
        "/home/chizhang/projects/Tiangong_isaac_rl/dataset/unigrasptransformer_asset/meshdatav3_scaled/subset_core10",
    )
    subset_root = Path(subset_root_env).expanduser().resolve()
    if not subset_root.exists():
        return

    candidates: list[tuple[Path, Path]] = []
    for category in sorted(p for p in subset_root.iterdir() if p.is_dir()):
        for obj in sorted(p for p in category.iterdir() if p.is_dir()):
            meta = obj / "metadata.json"
            if meta.exists():
                candidates.append((obj, meta))
    if not candidates:
        return

    obj_dir, meta_path = random.choice(candidates)
    try:
        data = json.loads(meta_path.read_text())
    except Exception:
        return

    usd = data.get("static_usd")
    if not usd:
        return
    usd_path = Path(usd).expanduser().resolve()
    if not usd_path.exists():
        return

    spawn_cfg.grasp_object.static_usd = usd_path.as_posix()
    spawn_cfg.grasp_object.pc_fps = data.get("pc_fps")
    spawn_cfg.grasp_object.pca_axes = data.get("pca_axes")
    spawn_cfg.grasp_object.object_init = data.get("object_init")
    spawn_cfg.grasp_object.object_id = obj_dir.name
    log_debug(f"Picked random dataset object: {obj_dir.name}")
