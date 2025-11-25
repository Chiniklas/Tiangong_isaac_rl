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
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.utils import configclass

from legged_lab.envs.base.my_confg import GraspObjectCfg, TableCfg
from legged_lab.assets.shadow_hand_with_fingertip.shadow_hand import SHADOW_HAND_CFG
from legged_lab.sensors.camera import TiledCameraCfg
from legged_lab.terrains.ray_caster_cfg import RayCasterCfg

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseSceneCfg


@configclass
class UniGraspSceneCfg(InteractiveSceneCfg):
    """This is the scene constructor that we use to generate scene."""

    def __init__(self, config: "BaseSceneCfg", physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        # Validate robot, table, and grasp object early so we fail fast before spawning anything.
        robot_cfg_in = getattr(config, "robot", None)
        shadow_usd = getattr(getattr(SHADOW_HAND_CFG, "spawn", None), "usd_path", None)
        usd_path_in = getattr(getattr(robot_cfg_in, "spawn", None), "usd_path", None)
        if not (isinstance(robot_cfg_in, ArticulationCfg) and usd_path_in == shadow_usd):
            raise ValueError("UniGraspSceneCfg requires the custom Shadow Hand ArticulationCfg (SHADOW_HAND_CFG).")

        table_cfg_in = getattr(config, "table", None)
        if not isinstance(table_cfg_in, TableCfg):
            raise ValueError("UniGraspSceneCfg requires a table specified via TableCfg.")

        object_cfg_in = getattr(config, "grasp_object", getattr(config, "object", None))
        if not isinstance(object_cfg_in, GraspObjectCfg):
            raise ValueError("UniGraspSceneCfg requires a grasp object specified via GraspObjectCfg.")
        if not object_cfg_in.enable:
            raise ValueError("UniGraspSceneCfg requires an enabled grasp object.")

        robot_cfg = robot_cfg_in.replace(prim_path="{ENV_REGEX_NS}/Robot")


        # now let's spawn the stage
        # spawn light
        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )

        # spawn table
        # unpack hyperparameters from the table config
        table_cfg = getattr(config, "table", None)
        if isinstance(table_cfg, TableCfg):
            self.table = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=table_cfg.size,
                    translation=table_cfg.pos,
                    orientation=table_cfg.rot_xyzw,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=table_cfg.friction,
                        dynamic_friction=table_cfg.friction,
                        restitution=table_cfg.restitution,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                ),
            )
            print(f"[UniGraspSceneCfg] Table prim: {self.table.prim_path}")
        else:
            pass

        # spawn object
        # unpack hyperparameters from the grasp_object config
        object_cfg = getattr(config, "grasp_object", getattr(config, "object", None))
        if isinstance(object_cfg, GraspObjectCfg):
            if not object_cfg.enable:
                raise ValueError("UniGraspSceneCfg requires an enabled grasp object.")
            if not object_cfg.object_path:
                raise ValueError("UniGraspSceneCfg requires grasp_object.object_path to spawn the object USD.")

            self.object = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=object_cfg.object_path,
                    orientation=object_cfg.rot_xyzw,
                    translation=object_cfg.pos,
                ),
            )
            print(f"[UniGraspSceneCfg] Object prim: {self.object.prim_path}")

            if object_cfg.show_point_cloud:
                pc_path = object_cfg.pc_fps_path
                if not pc_path:
                    raise ValueError("Point cloud overlay requested but pc_fps_path is missing in GraspObjectCfg.")
                self.object_point_cloud = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/ObjectPointCloud",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=Path(pc_path).expanduser().as_posix(),
                        orientation=object_cfg.rot_xyzw,
                        translation=object_cfg.pos,
                    ),
                )
                print(f"[UniGraspSceneCfg] Point cloud prim: {self.object_point_cloud.prim_path}")

            if object_cfg.show_pca_axes:
                axes_path = object_cfg.pca_axes_path
                if not axes_path:
                    raise ValueError("PCA axes overlay requested but pca_axes_path is missing in GraspObjectCfg.")
                self.object_pca_axes = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/ObjectPCAAxes",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=Path(axes_path).expanduser().as_posix(),
                        orientation=object_cfg.rot_xyzw,
                        translation=object_cfg.pos,
                    ),
                )
                print(f"[UniGraspSceneCfg] PCA axes prim: {self.object_pca_axes.prim_path}")
        else:
            pass

        # spawn robot
        self.robot: ArticulationCfg = robot_cfg
        print(f"[UniGraspSceneCfg] Robot prim: {self.robot.prim_path}")


__all__ = ["UniGraspSceneCfg"]
