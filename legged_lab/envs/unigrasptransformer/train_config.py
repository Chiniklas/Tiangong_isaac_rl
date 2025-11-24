from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from legged_lab.mdp.rewards_unigrasptransformer import RewardWeights


_CFG_DIR = Path(__file__).with_name("cfg")
_DEFAULT_WEIGHTS_YAML = _CFG_DIR / "train.yaml"
_DEFAULT_PPO_YAML = _CFG_DIR / "ppo_config.yaml"


def _load_yaml(default_path: Path, override: Path | None = None) -> Dict[str, Any]:
    yaml_path = default_path if override is None else Path(override).expanduser()
    if not yaml_path.exists():
        return {}
    data = yaml.safe_load(yaml_path.read_text()) or {}
    return data


def load_reward_weights_from_train_yaml(path: Path | None = None) -> RewardWeights:
    """Construct ``RewardWeights`` using the current train.yaml contents."""

    data = _load_yaml(_DEFAULT_WEIGHTS_YAML, path)
    weights_cfg: Mapping[str, Any] = data.get("Weights", {})
    kwargs: Dict[str, Any] = {}
    for spec in fields(RewardWeights):
        key = spec.name
        if key in weights_cfg and weights_cfg[key] is not None:
            kwargs[key] = weights_cfg[key]
    return RewardWeights(**kwargs)


def apply_agent_overrides_from_ppo_config(agent_cfg, path: Path | None = None) -> None:
    """Override PPO hyperparameters using the reference-compatible PPO config."""

    data = _load_yaml(_DEFAULT_PPO_YAML, path)
    if not data:
        return

    policy_cfg: Mapping[str, Any] = data.get("policy", {})
    policy = getattr(agent_cfg, "policy", None)
    if policy and policy_cfg:
        if "pi_hid_sizes" in policy_cfg:
            policy.actor_hidden_dims = list(policy_cfg["pi_hid_sizes"])
        if "vf_hid_sizes" in policy_cfg:
            policy.critic_hidden_dims = list(policy_cfg["vf_hid_sizes"])
        if "activation" in policy_cfg:
            policy.activation = policy_cfg["activation"]
        if "init_noise_std" in policy_cfg:
            policy.init_noise_std = policy_cfg["init_noise_std"]

    learn_cfg: Mapping[str, Any] = data.get("learn", {})
    if learn_cfg:
        # Runner-level parameters
        if "save_interval" in learn_cfg:
            agent_cfg.save_interval = learn_cfg["save_interval"]
        if "max_iterations" in learn_cfg:
            agent_cfg.max_iterations = learn_cfg["max_iterations"]
        if "nsteps" in learn_cfg:
            agent_cfg.num_steps_per_env = learn_cfg["nsteps"]
        if "cliprange" in learn_cfg:
            agent_cfg.algorithm.clip_param = learn_cfg["cliprange"]
        if "ent_coef" in learn_cfg:
            agent_cfg.algorithm.entropy_coef = learn_cfg["ent_coef"]
        if "noptepochs" in learn_cfg:
            agent_cfg.algorithm.num_learning_epochs = learn_cfg["noptepochs"]
        if "nminibatches" in learn_cfg:
            agent_cfg.algorithm.num_mini_batches = learn_cfg["nminibatches"]
        if "optim_stepsize" in learn_cfg:
            agent_cfg.algorithm.learning_rate = learn_cfg["optim_stepsize"]
        if "schedule" in learn_cfg:
            agent_cfg.algorithm.schedule = learn_cfg["schedule"]
        if "desired_kl" in learn_cfg:
            agent_cfg.algorithm.desired_kl = learn_cfg["desired_kl"]
        if "gamma" in learn_cfg:
            agent_cfg.algorithm.gamma = learn_cfg["gamma"]
        if "lam" in learn_cfg:
            agent_cfg.algorithm.lam = learn_cfg["lam"]
        if "max_grad_norm" in learn_cfg:
            agent_cfg.algorithm.max_grad_norm = learn_cfg["max_grad_norm"]

    if "clip_actions" in data:
        agent_cfg.clip_actions = data["clip_actions"]
    if "seed" in data and data["seed"] >= 0:
        agent_cfg.seed = data["seed"]


__all__ = [
    "apply_agent_overrides_from_ppo_config",
    "load_reward_weights_from_train_yaml",
]