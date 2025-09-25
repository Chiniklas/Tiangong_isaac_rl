import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# point to your local USD on disk
INSPIRE_HAND_USD = "/home/chizhang/projects/Tiangong_isaac_rl/legged_lab/assets/handright9253/urdf/handright9253_simplified/handright9253_simplified.usd"

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

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robots
    InspireHand = INSPIRE_HAND_CONFIG.replace(
        prim_path="{ENV_REGEX_NS}/InspireHand",
        # Optionally set initial world pose (position + orientation) or joint states here:
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.50, 0.0, 0.75),            # place it on a table, for example
            rot=(1.0, 0.0, 0.0, 0.0),         # w,x,y,z
            # joint_pos={"thumb_joint1": 0.0, "index_joint1": 0.0, ...}  # if you have named joints
        ),
    )



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # ------------------------ reset ---------------------------------
        if count % 500 == 0:
            count = 0

            # root state
            root_hand_state = scene["InspireHand"].data.default_root_state.clone()
            root_hand_state[:, :3] += scene.env_origins
            scene["InspireHand"].write_root_pose_to_sim(root_hand_state[:, :7])
            scene["InspireHand"].write_root_velocity_to_sim(root_hand_state[:, 7:])

            # joints
            hand_q, hand_dq = (
                scene["InspireHand"].data.default_joint_pos.clone(),
                scene["InspireHand"].data.default_joint_vel.clone(),
            )
            scene["InspireHand"].write_joint_state_to_sim(hand_q, hand_dq)

            # clear buffers
            scene.reset()
            print("[INFO]: Reset InspireHand state...")

        # ------------------------ control --------------------------------
        # sinusoidal open/close of all joints
        hand_q = scene["InspireHand"].data.default_joint_pos.clone()
        amp = 0.4   # radians; tune per joint limit
        freq = 0.6  # Hz
        hand_q += amp * np.sin(2 * np.pi * freq * sim_time)
        scene["InspireHand"].set_joint_position_target(hand_q)

        # ------------------------ step -----------------------------------
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()