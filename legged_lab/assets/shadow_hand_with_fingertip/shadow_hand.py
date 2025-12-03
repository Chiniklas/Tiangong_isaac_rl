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

VALID_LINKS = [1,7,8,9,12,13,14,16,18,19,21,22,24,25,27,29,31]
# 1: palm
# 7: ffproximal
# 8: ffmiddle
# 9: ffdistal
# 12: mfproximal
# 13: mfmiddle
# 14: mfdistal
# 16: rfknuckle
# 18: rfmiddle
# 19: rfdistal
# 21: lfmetacarpal
# 22: lfknuckle
# 24: lfmiddle
# 25: lfdistal
# 27: thbase
# 29: thhub
# 31: thdistal

REFERENCE_PD = {
    # values roughly aligned with UniGrasp dex grasp specs
    "stiffness": 1.0,
    "damping": 0.1,
    "effort": 1.0,
}

# Mimic joints (FFJ1/MFJ1/RFJ1/LFJ1) follow their proximal counterparts in the URDF,
# so actuators should target only the 18 actively driven joints.
ACTIVE_JOINTS = [
    # Fingers (exclude distal mimic joints)
    "FFJ4",
    "FFJ3",
    "FFJ2",
    "MFJ4",
    "MFJ3",
    "MFJ2",
    "RFJ4",
    "RFJ3",
    "RFJ2",
    "LFJ5",
    "LFJ4",
    "LFJ3",
    "LFJ2",
    # Thumb
    "THJ5",
    "THJ4",
    "THJ3",
    "THJ2",
    "THJ1",
]

SHADOW_HAND_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Hand",
    spawn=sim_utils.UsdFileCfg(
        usd_path= f"{ISAAC_ASSET_DIR}/shadow_hand_with_fingertip/shadow_hand_right_for_conversion/shadow_hand_right_for_conversion.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            # retain_accelerations=False,
            # linear_damping=0.0,
            # angular_damping=0.0,
            # max_linear_velocity=1000.0,
            # max_angular_velocity=1000.0,
            # max_depenetration_velocity=1.0,
            ),
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
        for name in ACTIVE_JOINTS
    },
)

__all__ = ["SHADOW_HAND_CFG"]
