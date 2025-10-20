from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSET_DIR = Path(__file__).resolve().parent
INSPIRE_HAND_USD = str(_ASSET_DIR / "urdf" / "handright9253_simplified" / "handright9253_simplified.usd")

INSPIRE_HAND_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",   # <-- match scene key "robot"
    spawn=sim_utils.UsdFileCfg(
        usd_path=INSPIRE_HAND_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,  # <-- REQUIRED for ContactSensor
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={},  # use USD defaults
        pos=(0.50, 0.0, 0.75),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        )
    },
)
