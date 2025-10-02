# Minimal reward / event helpers used by RewardManager & EventManager

import torch

TABLE_Z = 0.70

# ---------- Reward terms ----------

def reach_object(env, **kwargs):
    """Reward getting palm close to object."""
    palm_p = _palm_pos(env)                    # (N,3)
    obj_p = env.object.data.root_pos_w         # (N,3)
    dist = torch.linalg.norm(obj_p - palm_p, dim=1)
    return torch.exp(-3.0 * dist)              # (N,)

def lift_object(env, **kwargs):
    """Bonus when object lifted above table."""
    z = env.object.data.root_pos_w[:, 2]
    return (z > (TABLE_Z + 0.04)).float() * 1.0

def hold_still(env, **kwargs):
    """Small bonus when object is steady while lifted."""
    lin_ok = env.object.data.root_lin_vel_w.norm(dim=1) < 0.1
    ang_ok = env.object.data.root_ang_vel_w.norm(dim=1) < 0.8
    lifted = env.object.data.root_pos_w[:, 2] > (TABLE_Z + 0.04)
    return (lifted & lin_ok & ang_ok).float() * 0.5

def action_smoothness(env, **kwargs):
    """Small action penalty."""
    if not hasattr(env, "action"):
        return torch.zeros(env.num_envs, device=env.device)
    return -0.002 * (env.action**2).sum(dim=1)

# ---------- Event terms (optional minimal) ----------

def reset_hand_and_object(env, env_ids=None, **kwargs):
    """Put object back on table; hand to default joint targets."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    # reset object pose
    obj = env.object
    root = obj.data.default_root_state[env_ids].clone()
    root[:, :3] = env.scene.env_origins[env_ids] + torch.tensor([0.55, 0.0, TABLE_Z + 0.03], device=env.device)
    root[:, 3:7] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=env.device)
    obj.write_root_pose_to_sim(root[:, :7], env_ids)
    obj.write_root_velocity_to_sim(root[:, 7:], env_ids)
    # hand back to default targets
    q = env.hand.data.joint_pos[env_ids]
    env.hand.set_joint_position_target(q)
    return {}
    
# ---------- helpers ----------

def _palm_pos(env):
    tf = getattr(env.hand.data, "link_tf_w", None)
    if tf is not None and tf.ndim == 3 and tf.shape[1] > 0:
        return tf[:, 0, :3]
    return env.hand.data.root_pos_w
