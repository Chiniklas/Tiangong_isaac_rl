# legged_lab/envs/inspirehand/grasp_env.py

import torch
from legged_lab.envs.base.base_env import BaseEnv
from .grasp_cfg import InspireHandGraspEnvCfg


class InspireHandGraspEnv(BaseEnv):
    cfg: InspireHandGraspEnvCfg

    def __init__(self, cfg: InspireHandGraspEnvCfg, headless: bool):
        super().__init__(cfg, headless)

        # Always present
        self.hand  = self.scene["robot"]
        self.table = self.scene["table"]
        self.obj   = self.scene["object"]


        self._hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # # ---------- helpers ----------
    # def _try_get_scene_entity(self, name: str):
    #     """Return scene[name] if it exists, else None."""
    #     try:
    #         return self.scene[name]
    #     except KeyError:
    #         return None

    # ---------- observations ----------
    def compute_current_observations(self):
        q = self.hand.data.joint_pos
        dq = self.hand.data.joint_vel

        # rel pos hand->object if object exists; otherwise zeros(3)
        if self.obj is not None:
            obj_p = self.obj.data.root_pos_w
            palm_p = self.hand.data.root_pos_w     # using root as palm proxy for now
            rel_p = obj_p - palm_p
        else:
            rel_p = torch.zeros(self.num_envs, 3, device=self.device, dtype=q.dtype)

        actor_obs = torch.cat([q, dq, rel_p], dim=-1)
        critic_obs = actor_obs
        return actor_obs, critic_obs

    def compute_observations(self):
        actor_obs, critic_obs = self.compute_current_observations()
        return actor_obs, critic_obs

    # ---------- step / rewards ----------
    def step(self, actions: torch.Tensor):
        obs, reward_buf, reset_buf, extras = super().step(actions)
        reward_buf = self._get_rewards()
        # RSL-RL 2.x expects this for critic input
        extras["observations"]["critic"] = obs
        return obs, reward_buf, reset_buf, extras

    def _get_rewards(self) -> torch.Tensor:
        # Smoothness (tiny penalty) based on last action from BaseEnv’s delay buffer
        last_action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        r_smooth = -0.001 * (last_action**2).sum(dim=1)

        # If we don't have an object yet, only smoothness applies
        if self.obj is None:
            return r_smooth

        palm_p = self.hand.data.root_pos_w
        obj_p  = self.obj.data.root_pos_w

        # reach reward
        dist = torch.linalg.norm(obj_p - palm_p, dim=1)
        r_reach = torch.exp(-3.0 * dist)

        # lift reward: uses table if present, otherwise a heuristic z-threshold
        if self.table is not None:
            table_z = self.table.data.root_pos_w[:, 2]
            lifted = obj_p[:, 2] > (table_z + 0.05)
        else:
            lifted = obj_p[:, 2] > 0.75  # world-z heuristic without table

        r_lift = lifted.float() * 1.0

        # hold bonus for sustained lift
        self._hold_counter = torch.where(lifted, self._hold_counter + 1, torch.zeros_like(self._hold_counter))
        r_hold = 0.5 * (self._hold_counter > 10).float()

        return r_reach + r_lift + r_hold + r_smooth

    def check_reset(self):
        # simple time-limit termination
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf = time_out_buf.clone()
        return reset_buf, time_out_buf
