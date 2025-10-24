# -------------------------------------
# UniGraspTransformer spawn configuration mirrors the GraspXL setup while
# providing task-specific class names to keep the configuration hierarchy clear.

from __future__ import annotations

from isaaclab.utils import configclass

from legged_lab.envs.graspxl_rl import spawn_cfg as _base_spawn
from .logging_utils import log_debug


@configclass
class UniGraspTransformerTableSpawnCfg(_base_spawn.TableSpawnCfg):
    """Table geometry, material, and placement defaults for UniGraspTransformer."""

    def __post_init__(self):
        super().__post_init__()
        log_debug(f"TableSpawnCfg ready (enable={self.enable}, size={self.size})")


@configclass
class UniGraspTransformerObjectSpawnCfg(_base_spawn.ObjectSpawnCfg):
    """Spawn description for the grasp object placeholder."""

    def __post_init__(self):
        super().__post_init__()
        log_debug(
            f"ObjectSpawnCfg ready (static_usd={self.static_usd}, enable={self.enable})"
        )


@configclass
class UniGraspTransformerHandSpawnCfg(_base_spawn.HandSpawnCfg):
    """Root pose defaults for the Inspire Hand articulation."""

    def __post_init__(self):
        super().__post_init__()
        log_debug("HandSpawnCfg ready (fixed base pose)")


@configclass
class UniGraspTransformerSpawnCfg(_base_spawn.GraspXLSpawnCfg):
    """Aggregate spawn configuration bundling table, object, and hand."""

    table: UniGraspTransformerTableSpawnCfg = UniGraspTransformerTableSpawnCfg()
    grasp_object: UniGraspTransformerObjectSpawnCfg = UniGraspTransformerObjectSpawnCfg()
    hand: UniGraspTransformerHandSpawnCfg = UniGraspTransformerHandSpawnCfg()

    def __post_init__(self):
        super().__post_init__()
        log_debug(f"SpawnCfg ready (config_path={self.config_path})")


load_spawn_from_yaml = _base_spawn.load_spawn_from_yaml

__all__ = [
    "UniGraspTransformerTableSpawnCfg",
    "UniGraspTransformerObjectSpawnCfg",
    "UniGraspTransformerHandSpawnCfg",
    "UniGraspTransformerSpawnCfg",
    "load_spawn_from_yaml",
]
