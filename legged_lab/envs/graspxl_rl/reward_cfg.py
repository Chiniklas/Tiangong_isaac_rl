from isaaclab.utils import configclass


@configclass
class GraspXLRewardCfg:
    """Tunable coefficients for GraspXL grasp rewards."""

    reach: float = 1.0
    reach_exponent: float = 3.0
    lift: float = 1.0
    lift_height_buffer: float = 0.05
    ground_lift_height: float = 0.75
    hold: float = 0.5
    hold_duration: int = 10
    smooth: float = -0.001
    affordance_sdf: float = 0.3
    affordance_sdf_decay: float = 10.0
    non_affordance_sdf: float = 0.4
