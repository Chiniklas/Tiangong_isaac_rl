from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class TableSpawnCfg:
    """Table geometry and placement."""

    enable: bool = True
    size: tuple[float, float, float] = (0.6, 0.6, 0.03)
    pos: tuple[float, float, float] = (0.50, 0.0, 0.70)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    color: tuple[float, float, float] = (0.6, 0.6, 0.6)
    metallic: float = 0.0
    roughness: float = 0.6
    disable_gravity: bool = True


@configclass
class ObjectSpawnCfg:
    """Default object placeholder placement."""

    enable: bool = True
    size: tuple[float, float, float] = (0.05, 0.05, 0.10)
    pos: tuple[float, float, float] = (0.55, 0.0, 0.73)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float = 0.5
    disable_gravity: bool = False
    color: tuple[float, float, float] = (0.8, 0.3, 0.3)
    metallic: float = 0.2
    roughness: float = 0.4


@configclass
class HandSpawnCfg:
    """Root pose configuration for the Inspire Hand."""

    align_to_object: bool = True
    offset_from_object: tuple[float, float, float] = (0.0, 0.20, 0.0)
    hover_above_table: float = 0.25
    orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.70710678,
        0.0,
        0.70710678,
    )
    base_pos: tuple[float, float, float] = (0.55, 0.20, 0.95)


@configclass
class InspireHandSpawnCfg:
    """Aggregate spawn configuration for the Inspire Hand scene."""

    table: TableSpawnCfg = TableSpawnCfg()
    grasp_object: ObjectSpawnCfg = ObjectSpawnCfg()
    hand: HandSpawnCfg = HandSpawnCfg()


__all__ = [
    "TableSpawnCfg",
    "ObjectSpawnCfg",
    "HandSpawnCfg",
    "InspireHandSpawnCfg",
]
