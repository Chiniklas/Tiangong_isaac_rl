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

"""Configuration for the Shadow Hand with fingertips.

This mirrors the structure of :mod:`legged_lab.assets.tienkung2_lite.tienkung`
and provides a ready-to-use :data:`SHADOW_HAND_CFG` articulation config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
import yaml

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from legged_lab.assets import ISAAC_ASSET_DIR


def _resolve_usd_path() -> Path:
    """Resolve the Shadow Hand USD path, allowing overrides via config or env."""
    candidates = []

    # 1) config.yaml override (envs/unigrasptransformer/cfg/config.yaml: unigrasptransformer.hand.asset_path)
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

    # 3) fallback to checked-in USD
    candidates.append(
        Path(ISAAC_ASSET_DIR)
        / "shadow_hand_with_fingertip"
        / "shadow_hand_right_for_conversion"
        / "shadow_hand_right_for_conversion.usd"
    )

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Shadow hand USD not found. Set SHADOW_HAND_USD_PATH or place a USD at "
        "`legged_lab/assets/shadow_hand_with_fingertip/shadow_hand_right_for_conversion/shadow_hand_right_for_conversion.usd`."
    )


TARGET_INIT_JOINT_POS: Dict[str, float] = {
    # Finger flex joints
    "FFJ4": 0.1,
    "FFJ3": 0.0,
    "FFJ2": 0.6,
    "FFJ1": 0.0,
    "MFJ4": 0.0,
    "MFJ3": 0.0,
    "MFJ2": 0.6,
    "MFJ1": 0.0,
    "RFJ4": -0.1,
    "RFJ3": 0.0,
    "RFJ2": 0.6,
    "RFJ1": 0.0,
    "LFJ5": 0.0,
    "LFJ4": -0.2,
    "LFJ3": 0.0,
    "LFJ2": 0.6,
    "LFJ1": 0.0,
    # Thumb joints
    "THJ5": 0.0,
    "THJ4": 1.2,
    "THJ3": 0.0,
    "THJ2": -0.2,
    "THJ1": 0.0,
}

REFERENCE_PD = {
    # values roughly aligned with UniGrasp dex grasp specs
    "stiffness": 1.0,
    "damping": 0.1,
    "effort": 1.0,
}

_SHADOW_USD_PATH = _resolve_usd_path()

SHADOW_HAND_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Hand",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_SHADOW_USD_PATH.as_posix(),
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
        joint_vel={".*": 0.0},
    ),
    actuators={
        name: ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=REFERENCE_PD["effort"],
            velocity_limit_sim=50.0,
            stiffness=REFERENCE_PD["stiffness"],
            damping=REFERENCE_PD["damping"],
        )
        for name in TARGET_INIT_JOINT_POS.keys()
    },
)

__all__ = ["SHADOW_HAND_CFG"]
