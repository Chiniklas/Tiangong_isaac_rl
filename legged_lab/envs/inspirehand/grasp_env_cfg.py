from __future__ import annotations

import torch
from torch import Tensor

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Point to your local USD on disk
INSPIRE_HAND_USD = "/home/chizhang/projects/Tiangong_isaac_rl/legged_lab/assets/handright9253/urdf/handright9253_simplified/handright9253_simplified.usd"
TABLE_Z = 0.70
OBJ_Z   = TABLE_Z + 0.03

INSPIRE_HAND_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=INSPIRE_HAND_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    # optional initial joint pose and world pose
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # put initial joint angles here if you have names, or omit entirely
            # e.g. "thumb_joint1": 0.0, "index_joint1": 0.1, ...
        },
        pos=(0.0, 0.0, 0.0),   # spawn location in world
        rot=(1.0, 0.0, 0.0, 0.0),  # quaternion (w, x, y, z)
    ),
    # actuate all joints by default (adjust regex if needed)
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],          # or a tighter regex per finger
            effort_limit_sim=100.0,
            velocity_limit_sim=100.0,
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)

class InspireSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6), metallic=0.0, roughness=0.6),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, TABLE_Z), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=3.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3), metallic=0.2, roughness=0.4),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, OBJ_Z), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # ✅ replicate the hand per env by replacing prim_path and init_state here
    InspireHand = INSPIRE_HAND_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/InspireHand",
        # prim_path=INSPIRE_HAND_USD,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.50, 0.0, 0.75),
            rot=(1.0, 0.0, 0.0, 0.0),
            # joint_pos={...}  # optional hand-open pose
        ),
    )


@configclass
class InspireHandEnvCfg(DirectRLEnvCfg):
    episode_length_s: float = 8.0
    decimation: int = 2
    action_scale: float = 1.0
    action_space: int = 12      # infer
    observation_space: int = 17 # infer
    state_space: int = 0
    sim: SimulationCfg = SimulationCfg(dt=1/120.0, render_interval=decimation)

    # put your scene **here**
    scene: InspireSceneCfg = InspireSceneCfg(
        num_envs=1, env_spacing=2.0, replicate_physics=True, clone_in_fabric= False
    )


class InspireHandEnv(DirectRLEnv):
    cfg: InspireHandEnvCfg

    def __init__(self, cfg: InspireHandEnvCfg, render_mode=None, **kwargs):
        # Build sim + scene first (parent ctor)
        super().__init__(cfg, render_mode, **kwargs)

        # 1) Camera (purely for visualization)
        self.sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

        # 2) Pointers/shortcuts we will use everywhere
        #    It's safe to grab the articulation now: the scene is initialized by super().__init__
        self.hand = self.scene["InspireHand"]      # articulation (your hand)
        self.obj  = self.scene["object"]           # the red cuboid
        self.table = self.scene["table"]

        # 3) Constants used by rewards/dones
        self.table_z = float(TABLE_Z)
        self.obj_z   = float(OBJ_Z)

        # # 4) Palm link index (choose a reasonable default if name not found)
        # self.palm_link_idx = self._resolve_palm_link_idx(
        #     candidates=("palm", "base", "hand_base")
        # )

        # 5) Episode-state buffers
        #    Boolean: did we ever have >=K contacts? (you can wire this later)
        self._had_contact  = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        #    Counter: consecutive steps of a “good hold”
        self._hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 6) Cache joint limits as (N, J) on the right device (used in _apply_action)
        self._cache_joint_limits()
    
    def _get_palm_pose(self):
        """
        Returns (palm_p, palm_q) as (N,3) and (N,4).
        If link transforms are available, uses link 0.
        Otherwise falls back to articulation root.
        """
        tf = getattr(self.hand.data, "link_tf_w", None)
        if tf is not None and tf.ndim == 3 and tf.shape[1] > 0:
            palm_p = tf[:, 0, :3]
            palm_q = tf[:, 0, 3:]
        else:
            palm_p = self.hand.data.root_pos_w
            palm_q = self.hand.data.root_quat_w
        return palm_p, palm_q


    def _resolve_palm_link_idx(self, candidates=("palm",), default_idx=0) -> int:
        """
        Try to find a palm/base link by name. Falls back to default_idx if not found.
        Prints available link names once if nothing matches (useful for debugging).
        """
        names = list(self.hand.data.link_names)  # sequence[str]
        for nm in candidates:
            if nm in names:
                return names.index(nm)
        # If nothing matched, print once to help you pick the right name from USD
        if not hasattr(self, "_printed_link_names"):
            print("[InspireHandEnv] Palm link not found. Available link names:")
            for i, n in enumerate(names):
                print(f"  [{i:02d}] {n}")
            self._printed_link_names = True
        return default_idx

    def _cache_joint_limits(self):
        """
        Normalize joint_pos_limits into (N, J) tensors: self.jmin, self.jmax (on self.device).
        Handles shapes: (2,J), (J,2), (N,2,J), (N,J,2).
        """
        lims = self.hand.data.joint_pos_limits
        dev  = self.device

        # Normalize to (..., 2, J)
        if lims.ndim == 2:
            if lims.shape == (2, lims.shape[1]):         # (2, J)
                pair = lims
            elif lims.shape[-1] == 2:                    # (J, 2) -> (2, J)
                pair = lims.movedim(-1, -2)
            else:
                raise RuntimeError(f"Unexpected joint_pos_limits shape={tuple(lims.shape)}")
            jmin, jmax = pair[0], pair[1]                # (J,)
            self.jmin = jmin.unsqueeze(0).expand(self.num_envs, -1).to(dev).contiguous()
            self.jmax = jmax.unsqueeze(0).expand(self.num_envs, -1).to(dev).contiguous()

        elif lims.ndim == 3:
            if lims.shape[-2] == 2:                      # (N, 2, J)
                pair = lims
            elif lims.shape[-1] == 2:                    # (N, J, 2) -> (N, 2, J)
                pair = lims.movedim(-1, -2)
            else:
                raise RuntimeError(f"Unexpected joint_pos_limits shape={tuple(lims.shape)}")
            self.jmin = pair[:, 0, :].to(dev).contiguous()   # (N, J)
            self.jmax = pair[:, 1, :].to(dev).contiguous()   # (N, J)
        else:
            raise RuntimeError(f"Unexpected joint_pos_limits ndim={lims.ndim}")



    def _configure_gym_env_spaces(self):
        self.hand = self.scene["InspireHand"]
        J = self.hand.data.joint_pos.shape[1]
        self.cfg.action_space = J if self.cfg.action_space < 0 else self.cfg.action_space
        self.cfg.observation_space = J if self.cfg.observation_space < 0 else self.cfg.observation_space
        return super()._configure_gym_env_spaces()


    def _reset_idx(self, env_ids: torch.Tensor):
        # reset root pose (optional)
        root = self.hand.data.default_root_state[env_ids]
        root[:, :3] += self.scene.env_origins[env_ids]
        self.hand.write_root_pose_to_sim(root[:, :7], env_ids)
        self.hand.write_root_velocity_to_sim(root[:, 7:], env_ids)

        # let the scene push default joint states (avoids size mismatches)
        self.scene.reset(env_ids=env_ids)

        # (optional) zero targets/vels if you want a clean start
        q = self.hand.data.joint_pos[env_ids].clone()
        dq = torch.zeros_like(q)
        self.hand.set_joint_position_target(q)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Cache the latest action; do only lightweight checks here."""
        # Ensure device/dtype/shape are correct
        if actions.device != self.device:
            actions = actions.to(self.device)
        if actions.dtype != torch.float32:
            actions = actions.float()

        # Optionally clip here (you can also clip inside _apply_action)
        actions = torch.clamp(actions, -1.0, 1.0)

        # Cache for _apply_action()
        self._action = actions


    def _apply_action(self):
        """Map normalized actions to PD joint targets and write to sim."""
        a = getattr(self, "_action", None)
        if a is None:
            return  # first frame before any action arrived

        # clamp again for safety (harmless if already clipped)
        a = torch.clamp(a, -1.0, 1.0)            # (N, J)
        q = self.hand.data.joint_pos             # (N, J)

        # Ensure limits are (N,J)
        if self.jmin.ndim == 1:
            jmin = self.jmin.unsqueeze(0).expand_as(q)
            jmax = self.jmax.unsqueeze(0).expand_as(q)
        else:
            jmin, jmax = self.jmin, self.jmax

        # Small delta control and clamp to limits
        q_t = torch.clamp(q + 0.05 * a, min=jmin, max=jmax)
        self.hand.set_joint_position_target(q_t)

        self.action = a  # for diagnostics/dones

    # # mininal version: observations = joint positions only
    # def _get_observations(self) -> torch.Tensor:
    #     return self.hand.data.joint_pos

    def _get_observations(self) -> torch.Tensor:
        q  = self.hand.data.joint_pos
        dq = self.hand.data.joint_vel

        obj = self.scene["object"].data
        obj_p, obj_v, obj_w = obj.root_pos_w, obj.root_lin_vel_w, obj.root_ang_vel_w

        palm_p, palm_q = self._get_palm_pose()   # <<— new

        rel_p = obj_p - palm_p

        # Minimal obs: [q, dq, rel_p, obj_v, obj_w]
        obs = torch.cat([q, dq, rel_p, obj_v, obj_w], dim=1)
        return obs



    # # reward place holder: all zeros
    # def _get_rewards(self) -> torch.Tensor:
    #     return torch.zeros(self.num_envs, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        palm_p, _ = self._get_palm_pose()        # <<— new
        obj = self.scene["object"].data
        obj_p = obj.root_pos_w

        dist   = torch.linalg.norm(obj_p - palm_p, dim=1)
        r_reach = torch.exp(-3.0 * dist)

        r_smooth = -0.002 * (self.action**2).sum(dim=1)

        z = obj.root_pos_w[:, 2]
        r_lift = (z > (self.table_z + 0.04)).float() * 1.0

        slow = (obj.root_lin_vel_w.norm(dim=1) < 0.1) & (obj.root_ang_vel_w.norm(dim=1) < 0.8)
        hold = (z > (self.table_z + 0.04)) & slow

        self._hold_counter = torch.where(hold, self._hold_counter + 1, torch.zeros_like(self._hold_counter))
        r_hold = hold.float() * 0.5

        r = r_reach + r_lift + r_hold + r_smooth
        return r

    
    # # basic version
    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     # NaN/Inf guards (your existing checks)
    #     q_ok = torch.isfinite(self.hand.data.joint_pos).all(dim=1)
    #     dq_ok = torch.isfinite(self.hand.data.joint_vel).all(dim=1)
    #     a_ok = torch.isfinite(self.action).all(dim=1) if hasattr(self, "action") else torch.ones(
    #         self.num_envs, dtype=torch.bool, device=self.device
    #     )
    #     terminated = ~(q_ok & dq_ok & a_ok)  # shape: (num_envs,)

    #     # no explicit time-limit handling here; base can manage counters,
    #     # but step() still needs a second tensor -> return all False
    #     time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    #     return terminated, time_outs
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj = self.scene["object"].data
        z = obj.root_pos_w[:, 2]
        fell = z < (self.table_z - 0.02)
        nan  = ~torch.isfinite(self.hand.data.joint_pos).all(dim=1)
        steps_for_success = int(0.5 / (self.cfg.decimation * self.cfg.sim.dt))
        success = self._hold_counter > steps_for_success
        terminated = fell | nan | success
        time_outs = torch.zeros_like(terminated)
        return terminated, time_outs

    def get_observations(self):
        """Return (obs, extras) tuple."""
        return self._get_observations(), self.extras

    def get_privileged_observations(self):
        """Return None if not using privileged states."""
        return None

    def get_rewards(self):
        """Return reward tensor (N,)."""
        return self._get_rewards()

    def get_dones(self):
        """Return (terminated, time_outs) boolean tensors."""
        return self._get_dones()

    def get_extras(self):
        """Some runners fetch extras separately."""
        return self.extras


