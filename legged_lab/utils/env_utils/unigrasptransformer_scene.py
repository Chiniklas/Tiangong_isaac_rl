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
from isaaclab.sim.utils import clone
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.utils import configclass
import numpy as np

from legged_lab.envs.base.my_confg import GraspObjectCfg, TableCfg
from legged_lab.assets.shadow_hand_with_fingertip.shadow_hand import SHADOW_HAND_CFG
from legged_lab.sensors.camera import TiledCameraCfg
from legged_lab.terrains.ray_caster_cfg import RayCasterCfg


@clone
def _spawn_point_cloud_from_npy(
    prim_path: str,
    cfg: sim_utils.SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a UsdGeom.Points from an NPY file (matches legacy behavior)."""
    import numpy as np
    import omni.usd
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    parent_path = prim_path.rsplit("/", 1)[0]
    if parent_path and not prim_utils.is_prim_path_valid(parent_path):
        prim_utils.create_prim(parent_path, "Xform")
    data = np.load(Path(getattr(cfg, "npy_path")).expanduser().as_posix())
    if data.shape[1] < 3:
        raise ValueError(f"Point cloud numpy file must have at least 3 columns (xyz); got shape {data.shape}.")
    points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in data]
    prim = UsdGeom.Points.Define(stage, prim_path)
    xform = UsdGeom.Xformable(prim)
    if translation is not None:
        ops = xform.GetOrderedXformOps()
        translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        if translate_op is None:
            translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        precision = translate_op.GetPrecision()
        if precision == UsdGeom.XformOp.PrecisionDouble:
            translate_op.Set(Gf.Vec3d(*translation))
        else:
            translate_op.Set(Gf.Vec3f(*translation))
    if orientation is not None:
        quat = Gf.Quatf(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
        ops = xform.GetOrderedXformOps()
        orient_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
        if orient_op is None:
            orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        precision = orient_op.GetPrecision()
        if precision == UsdGeom.XformOp.PrecisionDouble:
            orient_op.Set(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
        else:
            orient_op.Set(quat)
    prim.CreateWidthsAttr([getattr(cfg, "width", 0.01)])
    prim.GetDisplayColorAttr().Set([Gf.Vec3f(*getattr(cfg, "color", (0.15, 0.85, 0.95)))])
    prim.GetPointsAttr().Set(points)
    return prim.GetPrim()


@clone
def _spawn_pca_axes_from_npy(
    prim_path: str,
    cfg: sim_utils.SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn PCA axes as BasisCurves from an NPY file (matches legacy behavior)."""
    import numpy as np
    import omni.usd
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if prim_path and not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, "Xform")
    axes = np.load(Path(getattr(cfg, "npy_path")).expanduser().as_posix())
    if axes.shape[0] < 3 or axes.shape[1] < 3:
        raise ValueError(f"PCA axes numpy file must have shape (3,3) or more; got {axes.shape}.")

    root = UsdGeom.Xform.Define(stage, prim_path)
    xform = UsdGeom.Xformable(root)
    if translation is not None:
        ops = xform.GetOrderedXformOps()
        translate_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        if translate_op is None:
            translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
        precision = translate_op.GetPrecision()
        if precision == UsdGeom.XformOp.PrecisionDouble:
            translate_op.Set(Gf.Vec3d(*translation))
        else:
            translate_op.Set(Gf.Vec3f(*translation))
    if orientation is not None:
        quat = Gf.Quatf(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3]))
        ops = xform.GetOrderedXformOps()
        orient_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
        if orient_op is None:
            orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        precision = orient_op.GetPrecision()
        if precision == UsdGeom.XformOp.PrecisionDouble:
            orient_op.Set(Gf.Quatd(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])))
        else:
            orient_op.Set(quat)

    colors = getattr(cfg, "colors", ((1.0, 0.3, 0.3), (0.3, 1.0, 0.3), (0.3, 0.3, 1.0)))
    scale = getattr(cfg, "scale", 0.2)
    for idx in range(3):
        axis = axes[idx]
        curve = UsdGeom.BasisCurves.Define(stage, f"{prim_path}/PCA_Axis_{idx}")
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreateWidthsAttr([0.02])
        curve.GetDisplayColorAttr().Set([Gf.Vec3f(*colors[idx])])
        a0 = (0.0, 0.0, 0.0)
        a1 = (float(scale * axis[0]), float(scale * axis[1]), float(scale * axis[2]))
        curve.GetPointsAttr().Set([Gf.Vec3f(*a0), Gf.Vec3f(*a1)])
    return root.GetPrim()


if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseSceneCfg


@configclass
class UniGraspSceneCfg(InteractiveSceneCfg):
    """This is the scene constructor that we use to generate scene."""

    def __init__(self, config: "BaseSceneCfg", physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        # now let's spawn the stage
        # spawn light
        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )

        # spawn table
        # unpack hyperparameters from the table config
        table_cfg = getattr(config, "table", None)
        if isinstance(table_cfg, TableCfg) and getattr(table_cfg, "enable", True):
            self.table = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=table_cfg.size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=table_cfg.friction,
                        dynamic_friction=table_cfg.friction,
                        restitution=table_cfg.restitution,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        kinematic_enabled=True,  # keep the table fixed even when other bodies collide
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=table_cfg.pos, rot=table_cfg.rot_xyzw),
            )
            print(f"[UniGraspSceneCfg] Table prim: {self.table.prim_path}")

        # spawn object
        # unpack hyperparameters from the grasp_object config
        object_cfg = getattr(config, "grasp_object", getattr(config, "object", None))
        if isinstance(object_cfg, GraspObjectCfg):
            if not object_cfg.enable:
                print("The Object spawning is disabled.")
            elif not object_cfg.object_path:
                raise ValueError("The object path is not passed down.")

            # spawn object usd
            object_spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=object_cfg.object_path,
            )
            # Keep the object from falling under gravity before grasp; user can enable gravity later if desired.
            object_spawn_cfg.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)
            self.object = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=object_spawn_cfg,
                init_state=AssetBaseCfg.InitialStateCfg(pos=object_cfg.pos, rot=object_cfg.rot_xyzw),
            )
            print(f"[UniGraspSceneCfg] Object prim: {self.object.prim_path}")

            # spawn object point cloud overlay
            if object_cfg.show_point_cloud:
                pc_path = object_cfg.pc_fps_path
                if not pc_path:
                    raise ValueError("Point cloud overlay requested but pc_fps_path is missing in GraspObjectCfg.")
                pc_cfg = sim_utils.SpawnerCfg(
                    func=_spawn_point_cloud_from_npy,
                    copy_from_source=False,
                )
                pc_cfg.npy_path = Path(pc_path).expanduser().as_posix()
                pc_cfg.width = 0.005
                pc_cfg.color = (0.15, 0.85, 0.95)
                # Place overlays under the object prim; keep local pose identity to avoid double transforms.
                self.object_point_cloud = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Object/ObjectPC",
                    spawn=pc_cfg,
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0,0,0), rot=(1,0,0,0)),
                )
                print(f"[UniGraspSceneCfg] Point cloud prim: {self.object_point_cloud.prim_path}")

            # spawn object pca axes overlay
            if object_cfg.show_pca_axes:
                axes_path = object_cfg.pca_axes_path
                if not axes_path:
                    raise ValueError("PCA axes overlay requested but pca_axes_path is missing in GraspObjectCfg.")
                axes_cfg = sim_utils.SpawnerCfg(
                    func=_spawn_pca_axes_from_npy,
                    copy_from_source=False,
                )
                axes_cfg.npy_path = Path(axes_path).expanduser().as_posix()
                axes_cfg.scale = 0.2
                axes_cfg.colors = (
                    (1.0, 0.3, 0.3),
                    (0.3, 1.0, 0.3),
                    (0.3, 0.3, 1.0),
                )
                self.object_pca_axes = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Object/PCAAxes",
                    spawn=axes_cfg,
                    init_state=AssetBaseCfg.InitialStateCfg(pos=(0,0,0), rot=(1,0,0,0)),
                )
                print(f"[UniGraspSceneCfg] PCA axes prim: {self.object_pca_axes.prim_path}")
        else:
            pass

        # spawn robot if enabled
        robot_cfg = getattr(config, "robot", None)
        if robot_cfg is not None:
            self.robot: ArticulationCfg = robot_cfg
            print(f"[UniGraspSceneCfg] Robot prim: {self.robot.prim_path}")


def get_point_cloud_world(env_index: int = 0, prim_suffix: str = "ObjectPC") -> np.ndarray:
    """Fetch the point cloud overlay for an env in world coordinates.

    Args:
        env_index: Which environment index to read from (default: 0).
        prim_suffix: Name of the point cloud prim under the object (default: ``ObjectPC``).

    Returns:
        An (N, 3) numpy array of points in world frame. If the prim is missing, returns an empty array.
    """
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim_path = f"/World/envs/env_{env_index}/Object/{prim_suffix}"
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.zeros((0, 3), dtype=np.float32)

    pc = UsdGeom.Points(prim)
    pts_local = pc.GetPointsAttr().Get()
    if not pts_local:
        return np.zeros((0, 3), dtype=np.float32)

    xf = pc.ComputeLocalToWorldTransform(omni.usd.get_context().get_time_code())
    pts_world = [xf.Transform(p) for p in pts_local]
    return np.array([[p[0], p[1], p[2]] for p in pts_world], dtype=np.float32)


__all__ = ["UniGraspSceneCfg"]
