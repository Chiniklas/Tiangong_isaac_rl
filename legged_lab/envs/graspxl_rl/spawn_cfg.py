# -------------------------------------
# Mirrors the InspireHand spawn configuration so GraspXL-specific environments
# can build table/object/robot spawn settings in a single namespace.  The
# helpers at the bottom keep the preview/test scripts clean by centralising the
# asset resolution logic.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from isaaclab.utils import configclass
from legged_lab.assets.inspirehand.object_library import GraspObjectInfo
from .logging_utils import log_debug

@configclass
class TableSpawnCfg:
    """Table geometry, material, and placement defaults."""

    enable: bool = True
    size: tuple[float, float, float] = (0.6, 0.6, 0.03)
    pos: tuple[float, float, float] = (0.00, 0.0, 0.70)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    metallic: float = 0.0
    roughness: float = 0.6
    disable_gravity: bool = True

    def __post_init__(self):
        log_debug(f"TableSpawnCfg ready (enable={self.enable}, size={self.size})")


@configclass
class ObjectSpawnCfg:
    """Spawn description for the grasp object placeholder."""

    enable: bool = True
    size: tuple[float, float, float] = (0.05, 0.05, 0.10)
    pos: tuple[float, float, float] = (0.00, 0.0, 0.73)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 0.5
    disable_gravity: bool = False
    color: tuple[float, float, float] = (0.8, 0.3, 0.3)
    metallic: float = 0.2
    roughness: float = 0.4
    asset_prim_name: str = "Object"
    object_id: Optional[str] = None
    static_usd: Optional[str] = None
    affordance_usd: Optional[str] = None
    affordance_color: tuple[float, float, float] = (0.2, 0.9, 0.2)
    affordance_opacity: float = 0.35
    affordance_prim_name: str = "AffordancePreview"
    affordance_sdf: Optional[str] = None
    non_affordance_sdf: Optional[str] = None
    lowest_point: Optional[float] = None
    affordance_sdf_data: Optional[dict[str, np.ndarray]] = None
    non_affordance_sdf_data: Optional[dict[str, np.ndarray]] = None

    def __post_init__(self):
        log_debug(
            f"ObjectSpawnCfg ready (static_usd={self.static_usd}, enable={self.enable})"
        )


@configclass
class HandSpawnCfg:
    """Root pose defaults for the Inspire Hand articulation."""
    pos: tuple[float, float, float] = (0.0, 0.0, 0.75)
    orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.70710678,
        0.0,
        0.70710678,
    )
    disable_gravity: bool = True

    def __post_init__(self):
        log_debug("HandSpawnCfg ready (fixed base pose)")


@configclass
class GraspXLSpawnCfg:
    """Aggregate spawn configuration bundling table, object, and hand."""

    table: TableSpawnCfg = TableSpawnCfg()
    grasp_object: ObjectSpawnCfg = ObjectSpawnCfg()
    hand: HandSpawnCfg = HandSpawnCfg()
    config_path: Optional[str] = str((Path(__file__).parent / "object_cfg.yaml").resolve())
    use_object_library: bool = True
    _override_object_info: Optional[GraspObjectInfo] = None

    def __post_init__(self):
        log_debug(f"GraspXLSpawnCfg ready (config_path={self.config_path})")


def _expand_path(path: str | Path | None) -> Optional[Path]:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def load_spawn_from_yaml(spawn_cfg: GraspXLSpawnCfg) -> Optional[GraspObjectInfo]:
    """Load spawn overrides from a YAML file and attach them to ``spawn_cfg``."""

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
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse metadata.json at {metadata_path}") from exc

    spawn_cfg.use_object_library = False

    def _metadata_path(key: str) -> Optional[Path]:
        value = metadata.get(key)
        if value is None:
            return None
        return _expand_path(value)

    static_path = _metadata_path("static_usd")
    if static_path is None:
        candidate = object_dir / f"{object_dir.name}_static.usd"
        if candidate.exists():
            static_path = candidate
    if static_path is None or not static_path.exists():
        raise FileNotFoundError(
            f"Static USD for object '{object_dir.name}' not found. "
            "Ensure the conversion tool generated metadata.json with 'static_usd'."
        )

    affordance_sdf = _metadata_path("affordance_sdf")
    non_affordance_sdf = _metadata_path("non_affordance_sdf")

    def _load_sdf(path: Optional[Path]) -> Optional[dict[str, np.ndarray]]:
        if path is None or not path.exists():
            return None
        data = np.load(path)
        return {
            "grid": data["sdf"],
            "min_bounds": data["min_bounds"],
            "max_bounds": data["max_bounds"],
        }

    affordance_sdf_data = _load_sdf(affordance_sdf)
    non_affordance_sdf_data = _load_sdf(non_affordance_sdf)
    lowest_point = None
    lowest_path = object_dir / "lowest_point_new.txt"
    if lowest_path.exists():
        try:
            text = lowest_path.read_text().strip()
            if text:
                lowest_point = float(text.split()[0])
        except (OSError, ValueError):
            lowest_point = None

    # Update spawn configuration values.
    spawn_cfg.grasp_object.static_usd = static_path.as_posix()
    spawn_cfg.grasp_object.affordance_sdf = affordance_sdf.as_posix() if affordance_sdf else None
    spawn_cfg.grasp_object.non_affordance_sdf = non_affordance_sdf.as_posix() if non_affordance_sdf else None
    spawn_cfg.grasp_object.lowest_point = lowest_point
    spawn_cfg.grasp_object.affordance_sdf_data = affordance_sdf_data
    spawn_cfg.grasp_object.non_affordance_sdf_data = non_affordance_sdf_data

    override = GraspObjectInfo(
        object_id=object_dir.name,
        category=object_dir.name.split("_", 1)[0],
        root_dir=object_dir,
        urdf=None,
        fixed_base_urdf=None,
        affordance_mesh=None,
        non_affordance_mesh=None,
        lowest_point=lowest_point,
        static_usd=static_path,
        affordance_usd=None,
        non_affordance_usd=None,
        affordance_sdf=affordance_sdf,
        non_affordance_sdf=non_affordance_sdf,
    )
    override.affordance_sdf_data = affordance_sdf_data  # type: ignore[attr-defined]
    override.non_affordance_sdf_data = non_affordance_sdf_data  # type: ignore[attr-defined]

    spawn_cfg._override_object_info = override
    log_debug(f"GraspXL spawn override loaded: {override.object_id}")
    return override
