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
        super().__init__(cfg, render_mode, **kwargs)
        self.sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

        # ---- cache joint limits robustly across versions/shapes ----
        lims = self.hand.data.joint_pos_limits  # could be (2,J), (J,2), (N,2,J), (N,J,2)

        if lims.ndim == 2:
            # No env dimension → broadcast across envs
            if lims.shape[0] == 2:          # (2, J)
                jmin = lims[0]              # (J,)
                jmax = lims[1]              # (J,)
            elif lims.shape[1] == 2:        # (J, 2)
                jmin = lims[:, 0]           # (J,)
                jmax = lims[:, 1]           # (J,)
            else:
                raise RuntimeError(f"Unexpected joint_pos_limits shape={tuple(lims.shape)}")
            # expand to (N, J)
            self.jmin = jmin.unsqueeze(0).expand(self.num_envs, -1).to(self.device).contiguous()
            self.jmax = jmax.unsqueeze(0).expand(self.num_envs, -1).to(self.device).contiguous()

        elif lims.ndim == 3:
            # Has env dimension
            if lims.shape[1] == 2:          # (N, 2, J)
                self.jmin = lims[:, 0, :].to(self.device).contiguous()  # (N,J)
                self.jmax = lims[:, 1, :].to(self.device).contiguous()
            elif lims.shape[2] == 2:        # (N, J, 2)
                self.jmin = lims[:, :, 0].to(self.device).contiguous()  # (N,J)
                self.jmax = lims[:, :, 1].to(self.device).contiguous()
            else:
                raise RuntimeError(f"Unexpected joint_pos_limits shape={tuple(lims.shape)}")
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


    def _get_observations(self) -> torch.Tensor:
        return self.hand.data.joint_pos

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # NaN/Inf guards (your existing checks)
        q_ok = torch.isfinite(self.hand.data.joint_pos).all(dim=1)
        dq_ok = torch.isfinite(self.hand.data.joint_vel).all(dim=1)
        a_ok = torch.isfinite(self.action).all(dim=1) if hasattr(self, "action") else torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        terminated = ~(q_ok & dq_ok & a_ok)  # shape: (num_envs,)

        # no explicit time-limit handling here; base can manage counters,
        # but step() still needs a second tensor -> return all False
        time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return terminated, time_outs

