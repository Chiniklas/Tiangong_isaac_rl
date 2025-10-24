from isaaclab.utils import configclass

from legged_lab.envs.base.base_config import ActionDelayCfg, DomainRandCfg
from legged_lab.envs.graspxl_rl.graspxl_cfg import (
    GraspXLRewardCfg,
    GraspXLResetCfg,
    GraspXLEventCfg,
    GraspXLGraspSceneCfg,
    GraspXLEnvCfg,
    GraspXLAgentCfg,
)

from .spawn_cfg import UniGraspTransformerSpawnCfg
from .logging_utils import log_debug


@configclass
class UniGraspTransformerRewardCfg(GraspXLRewardCfg):
    """Reward scaling configuration for UniGraspTransformer."""


@configclass
class UniGraspTransformerResetCfg(GraspXLResetCfg):
    """Reset tolerances for UniGraspTransformer."""


@configclass
class UniGraspTransformerEventCfg(GraspXLEventCfg):
    """Event randomization configuration."""


@configclass
class UniGraspTransformerGraspSceneCfg(GraspXLGraspSceneCfg):
    """Scene configuration using the UniGraspTransformer spawn layout."""

    spawn: UniGraspTransformerSpawnCfg = UniGraspTransformerSpawnCfg()

    def __post_init__(self):
        super().__post_init__()
        log_debug(
            "GraspSceneCfg ready (object=%s)"
            % getattr(self.spawn.grasp_object, "object_id", None)
        )


@configclass
class UniGraspTransformerEnvCfg(GraspXLEnvCfg):
    """Environment configuration for the UniGraspTransformer task."""

    scene: UniGraspTransformerGraspSceneCfg = UniGraspTransformerGraspSceneCfg()
    reward_scales: UniGraspTransformerRewardCfg = UniGraspTransformerRewardCfg()
    reset_cfg: UniGraspTransformerResetCfg = UniGraspTransformerResetCfg()
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=UniGraspTransformerEventCfg(),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 1, "min_delay": 0}),
    )

    def __post_init__(self):
        super().__post_init__()
        log_debug(f"EnvCfg ready (device={self.device})")


@configclass
class UniGraspTransformerAgentCfg(GraspXLAgentCfg):
    """Training configuration for UniGraspTransformer agents."""

    experiment_name = "unigrasptransformer_grasp"

    def __post_init__(self):
        super().__post_init__()
        log_debug(
            "AgentCfg ready (steps_per_env=%d, max_iterations=%d)"
            % (self.num_steps_per_env, self.max_iterations)
        )
