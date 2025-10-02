# legged_lab/envs/inspirehand/grasp_env.py

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from legged_lab.envs.base.base_env import BaseEnv
from .grasp_cfg import InspireHandGraspEnvCfg


class InspireHandGraspEnv(BaseEnv):
    cfg: InspireHandGraspEnvCfg

    def __init__(self, cfg: InspireHandGraspEnvCfg, headless: bool):
        super().__init__(cfg, headless)

        # Always present
        self.hand = self.scene["robot"]

        # Spawn table + object if not already in the scene
        self._spawn_table_and_object()

        # Now they exist; keep handles
        self.table = self.scene["table"]
        self.obj = self.scene["object"]

        self._hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ---------- NEW: spawn helpers ----------
    def _spawn_table_and_object(self):
        """Add a kinematic table and a dynamic cube to every env."""
        # If already present (e.g., you add them later via config), skip.
        names = set(self.scene.entity_names)
        need_table = "table" not in names
        need_object = "object" not in names
        if not (need_table or need_object):
            return

        # Table (kinematic)
        table_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(0.60, 0.60, 0.03),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    kinematic_enabled=True,
                    max_depenetration_velocity=3.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.6, 0.6), metallic=0.0, roughness=0.6
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.50, 0.0, 0.70),  # z = table height
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Grasp object (dynamic cube)
        obj_cfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.10),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=3.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.4),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.3, 0.3), metallic=0.2, roughness=0.4
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.55, 0.0, 0.73),  # slightly above table
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # Add to scene and (re)initialize their replicas across envs
        if need_table:
            self.scene.add("table", table_cfg)
        if need_object:
            self.scene.add("object", obj_cfg)

        # Write to sim & do a forward so data handles become valid
        self.scene.reset()
        self.scene.write_data_to_sim()
        self.sim.forward()
    # ---------- end new ----------

    # ------------------------ Observations ------------------------
    def compute_current_observations(self):
        q = self.hand.data.joint_pos
        dq = self.hand.data.joint_vel
        palm_p = self.hand.data.root_pos_w  # simple palm proxy

        obj_p = self.obj.data.root_pos_w
        rel_p = obj_p - palm_p

        actor_obs = torch.cat([q, dq, rel_p], dim=-1)
        critic_obs = actor_obs
        return actor_obs, critic_obs

    def compute_observations(self):
        actor_obs, critic_obs = self.compute_current_observations()
        return actor_obs, critic_obs

    # ------------------------ Step / Rewards ------------------------
    def step(self, actions: torch.Tensor):
        obs, reward_buf, reset_buf, extras = super().step(actions)
        reward_buf = self._get_rewards()
        extras["observations"]["critic"] = obs
        return obs, reward_buf, reset_buf, extras

    def _get_rewards(self) -> torch.Tensor:
        last_action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        r_smooth = -0.001 * (last_action**2).sum(dim=1)

        palm_p = self.hand.data.root_pos_w
        obj_p = self.obj.data.root_pos_w
        dist = torch.linalg.norm(obj_p - palm_p, dim=1)
        r_reach = torch.exp(-3.0 * dist)

        table_z = self.table.data.root_pos_w[:, 2]
        lifted = obj_p[:, 2] > (table_z + 0.05)
        r_lift = lifted.float() * 1.0

        self._hold_counter = torch.where(lifted, self._hold_counter + 1, torch.zeros_like(self._hold_counter))
        r_hold = 0.5 * (self._hold_counter > 10).float()

        return r_reach + r_lift + r_hold + r_smooth

    def check_reset(self):
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf = time_out_buf.clone()
        return reset_buf, time_out_buf
