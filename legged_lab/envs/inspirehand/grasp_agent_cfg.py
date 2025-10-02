from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass                         # <<—— add this
class InspireHandGraspAgentCfg(RslRlOnPolicyRunnerCfg):
    # ----- runner -----
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 32
    max_iterations = 40000
    empirical_normalization = False
    save_interval = 50
    clip_actions = None
    runner_class_name = "OnPolicyRunner"
    experiment_name = "inspirehand_grasp"   # now becomes a real string, not MISSING
    run_name = ""
    logger = "tensorboard"
    neptune_project = "inspirehand_grasp"
    wandb_project = "inspirehand_grasp"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
