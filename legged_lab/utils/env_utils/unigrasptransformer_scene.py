"""UniGraspTransformer-specific scene builder (table + object + hand)."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import UnigraspTransformerSceneCfg


@configclass
class UniGraspSceneCfg(InteractiveSceneCfg):
    """Interactive scene that spawns hand, optional table, and grasp object."""

    def __init__(self, config: UnigraspTransformerSceneCfg, physics_dt: float, step_dt: float):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        # No terrain for grasping by default.
        self.terrain = None

        # Robot (Shadow Hand by default)
        self.robot: ArticulationCfg = config.robot.replace(prim_path="{ENV_REGEX_NS}/Hand")

        # Table from RigidObjectCfg if provided, else from table_spawn
        table_cfg = getattr(config, "table", None)
        if isinstance(table_cfg, RigidObjectCfg):
            self.table = table_cfg.replace(prim_path="{ENV_REGEX_NS}/Table")
        # No cuboid fallback to avoid API mismatch; provide a RigidObjectCfg if a table is needed.

        # Object from RigidObjectCfg if provided, else from object_spawn
        object_cfg = getattr(config, "grasp_object", getattr(config, "object", None))
        if isinstance(object_cfg, RigidObjectCfg):
            self.object = object_cfg.replace(prim_path="{ENV_REGEX_NS}/Object")
        else:
            obj = config.object_spawn
            usd_path = obj.get("object_path") if isinstance(obj, dict) else None
            if usd_path:
                pos = obj.get("pos") or [0.0, 0.0, 0.5]
                rot = obj.get("rot_xyzw") or [0.0, 0.0, 0.0, 1.0]
                self.object = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=usd_path,
                        orientation=rot,
                        translation=pos,
                    ),
                )


__all__ = ["UniGraspSceneCfg"]
