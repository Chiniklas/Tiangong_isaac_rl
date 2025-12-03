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

from legged_lab.envs.base.my_confg import (
    GraspObjectCfg, 
    GraspObjectGoalCfg, 
    TableCfg)
from legged_lab.envs.unigrasptransformer.helpers import define_hand_points
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

@clone
def _spawn_hand_points_overlay(
    prim_path: str,
    cfg: sim_utils.SpawnerCfg,
    **kwargs,
):
    """Spawn per-body hand point overlays (mirrors compute_hand_body_pos offsets)."""
    import omni.usd
    import isaacsim.core.utils.prims as prim_utils
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    hand_root = prim_path.rsplit("/", 1)[0]  # e.g., /World/envs/env_X/Hand

    # Offsets are defined in the local frame of each body so they follow articulation motion.
    valid_names, offset_map, skip_indices = define_hand_points()
    color = getattr(cfg, "color", (0.95, 0.2, 0.6))
    width = getattr(cfg, "width", 0.01)

    # Group root to keep overlay discoverable in the stage tree.
    prim_utils.create_prim(prim_path, "Xform")

    for idx, name in enumerate(valid_names):
        body_path = f"{hand_root}/{name}"
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            # Body not present; skip instead of creating conflicting prims.
            continue

        offsets: list[tuple[float, float, float]] = []
        if idx not in skip_indices:
            offsets.extend(offset_map.get(name, [(0.0, 0.0, 0.02)]))

        # Always include the body origin (matches final concat in compute_hand_body_pos).
        offsets.append((0.0, 0.0, 0.0))

        points_prim_path = f"{body_path}/HandPoints"
        points_prim = UsdGeom.Points.Define(stage, points_prim_path)
        points_prim.CreateWidthsAttr([width] * len(offsets))
        points_prim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        points_prim.GetPointsAttr().Set([Gf.Vec3f(*o) for o in offsets])

    return stage.GetPrimAtPath(prim_path)


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
        # add sky dome for ambient lighting
        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=750.0),
        )
        # add a simple ground plane below the scene
        self.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
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

            # spawn object as a rigid body so pose/vel data is available
            object_spawn_cfg = sim_utils.UsdFileCfg(
                usd_path=object_cfg.object_path,
            )
            object_spawn_cfg.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)
            self.object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                spawn=object_spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=object_cfg.pos, rot=object_cfg.rot_xyzw),
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

        # spawn goal marker relative to object start pose
        goal_cfg = getattr(config, "object_goal", None)
        if isinstance(goal_cfg, GraspObjectGoalCfg) and getattr(goal_cfg, "enable", False):
            displacement = getattr(goal_cfg, "displacement", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
            base_pos = getattr(object_cfg, "pos", (0.0, 0.0, 0.0))
            goal_pos = tuple(base_pos[i] + displacement[i] for i in range(3))
            goal_spawn_cfg = sim_utils.SphereCfg(
                radius=goal_cfg.radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=goal_cfg.color),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            )
            self.object_goal = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Goal",
                spawn=goal_spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=goal_pos, rot=goal_cfg.rot_xyzw),
            )
            print(f"[UniGraspSceneCfg] Goal prim: {self.object_goal.prim_path} (pos {goal_pos}, disp {displacement})")
        else:
            pass

        # spawn robot if enabled
        robot_cfg = getattr(config, "robot", None)
        if robot_cfg is not None:
            self.robot: ArticulationCfg = robot_cfg
            print(f"[UniGraspSceneCfg] Robot prim: {self.robot.prim_path}")

            # Spawn debug overlay showing hand point definitions (body origins + offset points).
            hand_points_cfg = sim_utils.SpawnerCfg(
                func=_spawn_hand_points_overlay,
                copy_from_source=False,
            )
            hand_points_cfg.width = 0.01
            hand_points_cfg.color = (0.95, 0.2, 0.6)
            # the hand points are spawned under each link of the hand so it follows each link movement in runtime
            self.hand_points_overlay = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Hand/HandPointsOverlay",
                spawn=hand_points_cfg,
            )





__all__ = ["UniGraspSceneCfg"]
