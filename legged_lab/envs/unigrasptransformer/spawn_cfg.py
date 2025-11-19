"""UniGraspTransformer scene spawn configuration (standalone).

This module defines the table/object/hand spawn configuration classes and a
YAML loader for object overrides, without inheriting from the GraspXL variant.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Optional
from isaaclab.utils import configclass

from .logging_utils import log_debug


@configclass
class UniGraspTransformerTableSpawnCfg:
    """Table geometry, material, and placement defaults."""

    enable: bool = True
    size: tuple[float, float, float] = (0.6, 0.6, 0.03)
    pos: tuple[float, float, float] = (0.00, 0.0, 0.70)
    # Identity quaternion (x, y, z, w) so table is axis-aligned
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    metallic: float = 0.0
    roughness: float = 0.6
    disable_gravity: bool = True

    def __post_init__(self):
        log_debug(f"TableSpawnCfg ready (enable={self.enable}, size={self.size})")


@configclass
class UniGraspTransformerObjectSpawnCfg:
    """Spawn description for the grasp object placeholder."""

    enable: bool = True
    size: tuple[float, float, float] = (0.05, 0.05, 0.10)
    pos: tuple[float, float, float] = (0.00, 0.0, 0.73)
    # Identity quaternion (x, y, z, w) so object is upright by default
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    mass: float = 0.5
    disable_gravity: bool = False
    color: tuple[float, float, float] = (0.8, 0.3, 0.3)
    metallic: float = 0.2
    roughness: float = 0.4
    asset_prim_name: str = "Object"
    object_id: Optional[str] = None
    static_usd: Optional[str] = None
    # Spawn and overlay flags
    spawn_mesh: bool = True
    show_point_cloud: bool = True
    show_pca_axes: bool = True
    # Optional dataset helpers
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

    asset_type: str = "shadowhand"
    asset_path: str | None = None
    pos: tuple[float, float, float] = (0.0, 0.0, 0.75)
    orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.70710678,
        0.0,
        0.70710678,
    )
    disable_gravity: bool = True
    # Debug overlays for hand
    show_palm_dir: bool = True
    palm_dir_local: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    palm_dir_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    palm_dir_scale: float = 0.2
    fingertip_body_exprs: tuple[str, ...] = ("fftip", "mftip", "rftip", "lftip", "thtip")
    # Behavior
    warp_on_reset: bool = True

    def __post_init__(self):
        log_debug("HandSpawnCfg ready (fixed base pose)")


@configclass
class UniGraspTransformerSpawnCfg:
    """Aggregate spawn configuration bundling table, object, and hand."""

    table: UniGraspTransformerTableSpawnCfg = UniGraspTransformerTableSpawnCfg()
    grasp_object: UniGraspTransformerObjectSpawnCfg = UniGraspTransformerObjectSpawnCfg()
    hand: UniGraspTransformerHandSpawnCfg = UniGraspTransformerHandSpawnCfg()
    config_path: Optional[str] = str((Path(__file__).with_name("object_cfg.yaml").resolve()))
    use_object_library: bool = True

    def __post_init__(self):
        log_debug(f"SpawnCfg ready (config_path={self.config_path})")
        # Load unified YAML config (config.yaml next to this file) to override defaults
        try:
            load_unigrasp_config(self, Path(__file__).with_name("cfg").joinpath("config.yaml"))
        except Exception:
            # Keep defaults if config not found or malformed
            pass
        # If mesh object requested but no USD provided, try picking a random object
        try:
            _maybe_pick_random_dataset_object(self)
        except Exception:
            # Keep as-is; test script will validate and raise if inconsistent
            pass


def _expand_path(path: str | Path | None) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def load_spawn_from_yaml(spawn_cfg: UniGraspTransformerSpawnCfg) -> None:
    """Load spawn overrides from a YAML file and attach them to ``spawn_cfg``.

    The YAML must contain ``object_dir`` pointing to a directory that contains a
    pre-converted ``<object>_static.usd`` or a metadata.json that references it.
    """
    import yaml

    if spawn_cfg.config_path is None:
        return None

    yaml_path = _expand_path(spawn_cfg.config_path)
    if yaml_path is None or not yaml_path.exists():
        raise FileNotFoundError(f"Spawn config YAML not found: {spawn_cfg.config_path}")

    data = yaml.safe_load(yaml_path.read_text()) or {}

    object_dir = data.get("object_dir")
    if not object_dir:
        raise ValueError(f"'object_dir' must be provided in {yaml_path}")

    object_dir = _expand_path(object_dir)
    if object_dir is None or not object_dir.exists():
        raise FileNotFoundError(f"Object directory specified in YAML does not exist: {object_dir}")

    metadata_path = object_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover
            raise ValueError(f"Failed to parse metadata.json at {metadata_path}") from exc

    def _metadata_path(key: str) -> Optional[Path]:
        value = metadata.get(key)
        if value is None:
            return None
        return _expand_path(value)

    # Resolve static USD
    static_path = _metadata_path("static_usd")
    if static_path is None:
        candidate = object_dir / f"{object_dir.name}_static.usd"
        if candidate.exists():
            static_path = candidate
    if static_path is None or not static_path.exists():
        raise FileNotFoundError(
            f"Static USD for object '{object_dir.name}' not found. "
            "Ensure the conversion tool generated metadata.json with 'static_usd' or '<name>_static.usd'."
        )

    # Update spawn configuration values
    spawn_cfg.use_object_library = False
    spawn_cfg.grasp_object.static_usd = static_path.as_posix()
    # Optional dataset helpers
    pc_fps = _metadata_path("pc_fps")
    pca_axes = _metadata_path("pca_axes")
    object_init = _metadata_path("object_init")
    spawn_cfg.grasp_object.pc_fps = pc_fps.as_posix() if pc_fps else None
    spawn_cfg.grasp_object.pca_axes = pca_axes.as_posix() if pca_axes else None
    spawn_cfg.grasp_object.object_init = object_init.as_posix() if object_init else None

    # Store object id on the spawn config for downstream logging
    spawn_cfg.grasp_object.object_id = object_dir.name
    log_debug(f"UniGraspTransformer spawn override loaded: {spawn_cfg.grasp_object.object_id}")
    return None


__all__ = [
    "UniGraspTransformerTableSpawnCfg",
    "UniGraspTransformerObjectSpawnCfg",
    "UniGraspTransformerHandSpawnCfg",
    "UniGraspTransformerSpawnCfg",
    "load_spawn_from_yaml",
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
    # Optional hand overlays
    spawn_cfg.hand.show_palm_dir = bool(h.get("show_palm_dir", spawn_cfg.hand.show_palm_dir))
    palm_dir = h.get("palm_dir_local", list(spawn_cfg.hand.palm_dir_local))
    if palm_dir is not None:
        spawn_cfg.hand.palm_dir_local = _as_tuple(palm_dir, 3) or spawn_cfg.hand.palm_dir_local
    palm_offset = h.get("palm_dir_offset_local", list(spawn_cfg.hand.palm_dir_offset_local))
    if palm_offset is not None:
        spawn_cfg.hand.palm_dir_offset_local = _as_tuple(palm_offset, 3) or spawn_cfg.hand.palm_dir_offset_local
    spawn_cfg.hand.palm_dir_scale = float(h.get("palm_dir_scale", spawn_cfg.hand.palm_dir_scale))
    spawn_cfg.hand.warp_on_reset = bool(h.get("warp_on_reset", spawn_cfg.hand.warp_on_reset))
    exprs = h.get("fingertip_body_exprs")
    if exprs:
        spawn_cfg.hand.fingertip_body_exprs = tuple(exprs)

    log_debug("Unified config loaded from %s" % yaml_path)


def _maybe_pick_random_dataset_object(spawn_cfg: UniGraspTransformerSpawnCfg) -> None:
    """If mesh spawning is enabled without a USD, pick a random object from a dataset subset.

    The subset root is taken from environment variable UGTF_SUBSET_ROOT, or falls back to
    the project's default path used during development.
    """
    if not (spawn_cfg.grasp_object.enable and spawn_cfg.grasp_object.spawn_mesh):
        return
    if getattr(spawn_cfg.grasp_object, "static_usd", None):
        return

    subset_root_env = os.environ.get(
        "UGTF_SUBSET_ROOT",
        "/home/chizhang/projects/Tiangong_isaac_rl/dataset/unigrasptransformer_asset/meshdatav3_scaled/subset_core10",
    )
    subset_root = Path(subset_root_env).expanduser().resolve()
    if not subset_root.exists():
        return

    candidates: list[tuple[Path, Path]] = []
    for cat in sorted(p for p in subset_root.iterdir() if p.is_dir()):
        for obj in sorted(p for p in cat.iterdir() if p.is_dir()):
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

    # Set resolved object on spawn cfg
    spawn_cfg.grasp_object.static_usd = usd_path.as_posix()
    spawn_cfg.grasp_object.pc_fps = data.get("pc_fps")
    spawn_cfg.grasp_object.pca_axes = data.get("pca_axes")
    spawn_cfg.grasp_object.object_init = data.get("object_init")
    spawn_cfg.grasp_object.object_id = obj_dir.name
    log_debug(f"Picked random dataset object: {obj_dir.name}")
