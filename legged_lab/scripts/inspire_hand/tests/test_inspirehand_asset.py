# legged_lab/scripts/tests/test_inspirehand_asset.py

## to test:
#  - basic loading of the InspireHand asset
#  - basic articulation data access
#  - basic stepping with joint position targets

from isaaclab.app import AppLauncher

# CLI passthrough (optional, but nice for --headless, --renderer, etc.)
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

# Start Omniverse Kit/Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ----------------- Test code below -----------------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext, PhysxCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from legged_lab.assets.handright9253.inspirehand import INSPIRE_HAND_CFG

# 1) Build a minimal scene config
class _SceneCfg(InteractiveSceneCfg):
    # simple ground so the stage is valid
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # our hand asset (key name "hand" in the scene dict)
    hand = INSPIRE_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/InspireHand"
    )

def main():
    # 2) Simulation settings
    sim_cfg = sim_utils.SimulationCfg(
        device="cuda:0",
        dt=1.0 / 120.0,
        render_interval=2,
        physx=PhysxCfg(gpu_max_rigid_patch_count=2**16),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )
    sim = SimulationContext(sim_cfg)

    # 3) Create the scene
    scene_cfg = _SceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)

    # 4) Start/Reset
    sim.reset()
    scene.reset()

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # 5) Fetch the articulation and print some info
    hand = scene["hand"]  # key is the cfg attribute name ("hand")

    data = hand.data

    print("\n[JOINT INVENTORY]")
    names = list(data.joint_names) if hasattr(data, "joint_names") else []
    dof_counts = list(getattr(data, "joint_dof_count", [])) or list(getattr(data, "joint_dof_counts", []))

    # joint_pos_limits shape can be (2, J) or (N,2,J) etc. Normalize to (2, J) for printing env 0.
    lims = getattr(data, "joint_pos_limits", None)
    if lims is not None:
        if lims.ndim == 2:                # (2, J)
            jmin = lims[0]
            jmax = lims[1]
        elif lims.ndim == 3:              # (N, 2, J) or (N, J, 2)
            if lims.shape[-2] == 2:
                jmin = lims[0, 0]
                jmax = lims[0, 1]
            else:
                pair = lims[0].movedim(-1, -2)
                jmin, jmax = pair[0], pair[1]
    else:
        jmin = jmax = None

    J = data.joint_pos.shape[1]
    print(f"Total DoFs = {J}\n")
    for i in range(J):
        nm = names[i] if i < len(names) else f"joint_{i}"
        dc = dof_counts[i] if i < len(dof_counts) else 1
        if jmin is not None:
            print(f"[{i:02d}] {nm:32s}  dof={dc}  limits=({float(jmin[i]):+.3f}, {float(jmax[i]):+.3f})")
        else:
            print(f"[{i:02d}] {nm:32s}  dof={dc}  limits=(unknown)")

    # print("\n[LINKS]")
    # for i, ln in enumerate(list(data.link_names)):
    #     print(f"  [{i:02d}] {ln}")

#########################################################################
    sim.forward()         # push to sim once so data is valid

    # Data sanity checks
    q = hand.data.joint_pos              # (N, J)
    qd = hand.data.joint_vel             # (N, J)
    limits = getattr(hand.data, "joint_pos_limits", None)

    print("[INFO] Hand articulation ready.")
    print(f"  Number of envs (N): {q.shape[0] if q is not None else 'unknown'}")
    print(f"  Number of DOFs (J): {q.shape[1] if q is not None else 'unknown'}")
    if limits is not None:
        # limits can be (2, J) or (N, 2, J), print first 5
        if limits.ndim == 2:      # (2, J)
            jmin, jmax = limits[0], limits[1]
        elif limits.ndim == 3:    # (N, 2, J)
            jmin, jmax = limits[0, 0], limits[0, 1]
        else:
            jmin = jmax = None
        if jmin is not None:
            print("  First 5 joint limits:")
            print("    jmin:", jmin[:5].tolist())
            print("    jmax:", jmax[:5].tolist())

    # 6) Step a few frames
    for i in range(10):
        # keep current joint positions as targets (no motion yet)
        if q is not None:
            hand.set_joint_position_target(hand.data.joint_pos)
        scene.write_data_to_sim()
        sim.step(render=not args.headless)
        scene.update(dt=sim_cfg.dt)

    print("[SUCCESS] InspireHand asset basic test completed.")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
