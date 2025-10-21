# -------------------------------------
# Mirrors the InspireHand spawn configuration so GraspXL-specific environments
# can build table/object/robot spawn settings in a single namespace.  The
# helpers at the bottom keep the preview/test scripts clean by centralising the
# asset resolution logic.

from __future__ import annotations

from pathlib import Path
from typing import Optional

from isaaclab.utils import configclass


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


@configclass
class HandSpawnCfg:
    """Root pose defaults for the Inspire Hand articulation."""

    align_to_object: bool = True
    offset_from_object: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hover_above_table: float = 0.1
    # Translation applied by the URDF's fixed joint from the articulation root to base_link.
    root_link_offset: tuple[float, float, float] = (0.0, 0.0, 0.5)
    orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.70710678,
        0.0,
        0.70710678,
    )
    base_pos: tuple[float, float, float] = (0.55, 0.20, 0.95)


@configclass
class GraspXLSpawnCfg:
    """Aggregate spawn configuration bundling table, object, and hand."""

    table: TableSpawnCfg = TableSpawnCfg()
    grasp_object: ObjectSpawnCfg = ObjectSpawnCfg()
    hand: HandSpawnCfg = HandSpawnCfg()

