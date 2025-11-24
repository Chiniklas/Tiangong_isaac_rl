from __future__ import annotations

import os
from pathlib import Path
import yaml

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


def _resolve_usd_path() -> Path:
    """Resolve a Shadow Hand USD; prefer config.yaml override, then env, then default."""

    candidates = []
    # 1) config.yaml override (unigrasptransformer/cfg/config.yaml: unigrasptransformer.hand.asset_path)
    try:
        cfg_yaml = (
            Path(__file__).resolve()
            .parents[1]
            .joinpath("envs", "unigrasptransformer", "cfg", "config.yaml")
        )
        if cfg_yaml.exists():
            data = yaml.safe_load(cfg_yaml.read_text()) or {}
            cfg_path = (
                data.get("unigrasptransformer", {})
                .get("hand", {})
                .get("asset_path")
            )
            if cfg_path:
                candidates.append(Path(cfg_path).expanduser().resolve())
    except Exception:
        pass

    # 2) explicit env var
    env_path = os.environ.get("SHADOW_HAND_USD_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser().resolve())

    # Fallback to the checked-in USD if available.
    candidates.append(
        Path(__file__).resolve().parent / "shadow_hand_with_fingertip" / "shadow_hand_right" / "shadow_hand_right.usd"
    )

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Shadow hand USD not found. Set SHADOW_HAND_USD_PATH or place a USD at "
        "`legged_lab/assets/shadow_hand_with_fingertip/shadow_hand_right/shadow_hand_right.usd`."
    )

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


_SHADOW_USD_PATH = _resolve_usd_path()

# Approximate the reference UniGraspTransformer MJCF actuator gains/limits.
# The original MJCF sets kp=1.0 on all actuated joints with low force limits;
# we mirror that here while keeping all 22 joints independently actuated.
_REFERENCE_PD_SPECS = {
    # forcerange magnitudes pulled from dexgrasp/hand_assets/open_ai_assets/hand/shared.xml
    "FFJ3": {"stiffness": 1.0, "damping": 0.1, "effort": 0.9},
    "FFJ2": {"stiffness": 1.0, "damping": 0.1, "effort": 0.9},
    "FFJ1": {"stiffness": 1.0, "damping": 0.1, "effort": 0.7245},
    "MFJ3": {"stiffness": 1.0, "damping": 0.1, "effort": 0.9},
    "MFJ2": {"stiffness": 1.0, "damping": 0.1, "effort": 0.9},
    "MFJ1": {"stiffness": 1.0, "damping": 0.1, "effort": 0.7245},
    "RFJ3": {"stiffness": 1.0, "damping": 0.1, "effort": 0.9},
    "RFJ2": {"stiffness": 1.0, "damping": 0.1, "effort": 0.9},
    "RFJ1": {"stiffness": 1.0, "damping": 0.1, "effort": 0.7245},
    "LFJ4": {"stiffness": 1.0, "damping": 0.1, "effort": 0.9},
    "LFJ3": {"stiffness": 1.0, "damping": 0.1, "effort": 0.9},
    "LFJ2": {"stiffness": 1.0, "damping": 0.1, "effort": 0.9},
    "LFJ1": {"stiffness": 1.0, "damping": 0.1, "effort": 0.7245},
    # Thumb joints are shifted by +1 in the USD (palm→5→4→3→2→1).
    # Map reference THJ4→USD THJ5, THJ3→THJ4, THJ2→THJ3, THJ1→THJ2, THJ0→THJ1.
    "THJ5": {"stiffness": 1.0, "damping": 0.1, "effort": 2.3722},  # ref THJ4
    "THJ4": {"stiffness": 1.0, "damping": 0.1, "effort": 1.45},    # ref THJ3
    "THJ3": {"stiffness": 1.0, "damping": 0.1, "effort": 0.99},    # ref THJ2
    "THJ2": {"stiffness": 1.0, "damping": 0.1, "effort": 0.99},    # ref THJ1
    "THJ1": {"stiffness": 1.0, "damping": 0.1, "effort": 0.81},    # ref THJ0
}
# For joints not listed in the MJCF actuator section (e.g., spread joints), fall back to a small effort.
_DEFAULT_PD = {"stiffness": 1.0, "damping": 0.1, "effort": 0.9}

SHADOW_HAND_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_SHADOW_USD_PATH.as_posix(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=TARGET_INIT_JOINT_POS,
    ),
    actuators={
        name: ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=spec.get("effort", _DEFAULT_PD["effort"]),
            velocity_limit_sim=50.0,
            stiffness=spec.get("stiffness", _DEFAULT_PD["stiffness"]),
            damping=spec.get("damping", _DEFAULT_PD["damping"]),
        )
        # Limit to the joints present in the USD; THJ0 is not in this asset, and wrist joints (WRJ1/2) are absent in the palm-only USD.
        for name, spec in {
            **{j: _DEFAULT_PD for j in _TARGET_INIT_JOINT_ORDER},
            **_REFERENCE_PD_SPECS,
        }.items()
        if name in {
            "FFJ4",
            "LFJ5",
            "MFJ4",
            "RFJ4",
            "THJ5",
            "FFJ3",
            "LFJ4",
            "MFJ3",
            "RFJ3",
            "THJ4",
            "FFJ2",
            "LFJ3",
            "MFJ2",
            "RFJ2",
            "THJ3",
            "FFJ1",
            "LFJ2",
            "MFJ1",
            "RFJ1",
            "THJ2",
            "LFJ1",
            "THJ1",
        }
    },
)

__all__ = ["SHADOW_HAND_CFG"]
