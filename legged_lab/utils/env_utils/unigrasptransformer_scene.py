# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.utils import configclass

from legged_lab.sensors.camera import TiledCameraCfg
from legged_lab.terrains.ray_caster_cfg import RayCasterCfg

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseSceneCfg


@configclass
class UniGraspSceneCfg(InteractiveSceneCfg):
    """Interactive scene with optional grasp props (table + object)."""

    def __init__(self, config: "BaseSceneCfg", physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        self.terrain = None

        if not isinstance(config.robot, ArticulationCfg):
            raise ValueError("UniGraspSceneCfg requires a robot ArticulationCfg (e.g., Shadow Hand).")
        self.robot: ArticulationCfg = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print(f"[UniGraspSceneCfg] Robot prim: {self.robot.prim_path}")

        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )

        table_cfg = getattr(config, "table", None)
        if isinstance(table_cfg, RigidObjectCfg):
            self.table = table_cfg.replace(prim_path="{ENV_REGEX_NS}/Table")
            print(f"[UniGraspSceneCfg] Table prim: {self.table.prim_path}")
        else:
            # Require an explicit RigidObjectCfg table for now to avoid API mismatches.
            raise ValueError("UniGraspSceneCfg requires a table (RigidObjectCfg).")

        object_cfg = getattr(config, "grasp_object", getattr(config, "object", None))
        if isinstance(object_cfg, RigidObjectCfg):
            self.object = object_cfg.replace(prim_path="{ENV_REGEX_NS}/Object")
            print(f"[UniGraspSceneCfg] Object prim: {self.object.prim_path}")
        else:
            obj = getattr(config, "object_spawn", {}) or {}
            usd_path = obj.get("object_path") if isinstance(obj, dict) else None
            if obj.get("enable", False) and usd_path:
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
                print(f"[UniGraspSceneCfg] Object prim: {self.object.prim_path}")
            else:
                raise ValueError("UniGraspSceneCfg requires an object (grasp_object cfg or object_spawn with object_path).")


__all__ = ["UniGraspSceneCfg"]
