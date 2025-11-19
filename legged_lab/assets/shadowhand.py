from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSET_DIR = Path(__file__).resolve().parent / "shadow_hand_unigrasptransformer"
_URDF_PATH = _ASSET_DIR / "urdf" / "shadow_hand_description" / "shadowhand_with_fingertips.urdf"
_USD_CACHE_DIR = _ASSET_DIR / "urdf" / "shadow_hand_description" / "usd_cache"
_USD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Map the 22 UniGrasp DOFs (no wrist joints) onto the URDF names.
_TARGET_INIT_JOINT_ORDER = (
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "FFJ1",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "MFJ1",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "RFJ1",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
)
_TARGET_INIT_QPOS = (
    0.1,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.6,
    0.0,
    -0.1,
    0.0,
    0.6,
    0.0,
    0.0,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    1.2,
    0.0,
    -0.2,
    0.0,
)
TARGET_INIT_JOINT_POS = dict(zip(_TARGET_INIT_JOINT_ORDER, _TARGET_INIT_QPOS))


def _default_joint_drive():
    gains = sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=5000.0, damping=25.0)
    return sim_utils.UrdfConverterCfg.JointDriveCfg(drive_type="force", target_type="position", gains=gains)


SHADOW_HAND_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH.as_posix(),
        usd_dir=_USD_CACHE_DIR.as_posix(),
        make_instanceable=False,
        fix_base=False,
        link_density=800.0,
        force_usd_conversion=True,
        joint_drive=_default_joint_drive(),
        self_collision=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=TARGET_INIT_JOINT_POS,
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=150.0,
            velocity_limit_sim=50.0,
            stiffness=4000.0,
            damping=80.0,
        )
    },
)

__all__ = ["SHADOW_HAND_CFG"]
