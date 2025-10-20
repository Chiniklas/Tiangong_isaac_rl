# legged_lab/envs/inspirehand/grasp_env.py

import torch
from isaaclab.utils.buffers import DelayBuffer
from legged_lab.envs.base.base_env import BaseEnv
from .grasp_cfg import InspireHandGraspEnvCfg


class InspireHandGraspEnv(BaseEnv):
    cfg: InspireHandGraspEnvCfg

    def __init__(
        self,
        cfg: InspireHandGraspEnvCfg,
        headless: bool | None = None,
        *,
        render_mode=None,
        **kwargs,
    ):
        if headless is None:
            if isinstance(render_mode, bool):
                headless = not render_mode
            else:
                headless = render_mode is None

        # store render_mode for callers that rely on it (e.g. Gymnasium)
        self.render_mode = render_mode

        super().__init__(cfg, headless)

        # Always present
        self.hand  = self.scene["robot"]
        self.table = self.scene["table"]
        self.obj   = self.scene["object"]

        # action split: finger joints + palm translation
        self._joint_action_dim = self.robot.data.default_joint_pos.shape[1]
        self._palm_action_dim = 3
        self.num_actions = self._joint_action_dim + self._palm_action_dim
        self._palm_action_scale = 0.03  # meters per env step for unit command

        # rebuild delay buffer to handle extended action dimension
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        # refresh observation buffers with updated action dimension
        self.init_obs_buffer()

        self._zero_root_vel = torch.zeros(self.num_envs, 6, device=self.device)

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
        delayed_actions = self.action_buffer.compute(actions)

        joint_actions = delayed_actions[:, : self._joint_action_dim]
        palm_actions = delayed_actions[:, self._joint_action_dim :]

        joint_actions = torch.clip(joint_actions, -self.clip_actions, self.clip_actions).to(self.device)
        joint_targets = joint_actions * self.action_scale + self.robot.data.default_joint_pos

        self._apply_palm_translation(palm_actions)

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(joint_targets)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        _ = self.reward_manager.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        reward_buf = self._get_rewards()
        extras = self.extras
        extras["observations"]["critic"] = actor_obs
        return actor_obs, reward_buf, self.reset_buf, extras

    def _get_rewards(self) -> torch.Tensor:
        # Smoothness (tiny penalty) based on last action from BaseEnv’s delay buffer
        last_joint_action = self.action_buffer._circular_buffer.buffer[:, -1, : self._joint_action_dim]
        r_smooth = -0.001 * (last_joint_action**2).sum(dim=1)

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

    def _apply_palm_translation(self, palm_actions: torch.Tensor):
        if palm_actions.numel() == 0:
            return

        clamped = torch.clip(palm_actions, -1.0, 1.0)
        root_state = self.robot.data.root_state_w.clone()
        root_state[:, :3] += clamped * self._palm_action_scale

        if self.table is not None:
            min_height = self.table.data.root_pos_w[:, 2] + 0.02
            root_state[:, 2].clamp_(min=min_height)

        self.robot.write_root_pose_to_sim(root_state[:, :7])
        self.robot.write_root_velocity_to_sim(self._zero_root_vel)
