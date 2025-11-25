#!/usr/bin/env python3
"""Minimal smoke test that exercises UniGrasp scene construction.

Builds a UniGraspTransformer config, creates the SimulationContext, then
constructs the InteractiveScene via UniGraspSceneCfg. Intended to quickly
validate that the spawn wiring (robot/table/object) still works after edits.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _ensure_isaaclab_on_path():
    """Ensure the local Isaac Lab source tree is importable."""
    import os
    import sys

    if "isaaclab" in sys.modules:
        return

    source_hint = os.environ.get("ISAACLAB_SOURCE")
    candidates = []
    if source_hint:
        candidates.append(Path(source_hint))
        candidates.append(Path(source_hint) / "isaaclab")
    home_root = Path.home() / "IsaacLab" / "source"
    candidates.append(home_root)
    candidates.append(home_root / "isaaclab")

    for path in candidates:
        if path.exists() and path.as_posix() not in sys.path:
            sys.path.append(path.as_posix())

    try:
        import toml  # noqa: F401
    except ModuleNotFoundError:
        try:
            import tomllib as toml  # type: ignore  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "Missing 'toml' dependency required by Isaac Lab. Install it via 'pip install toml'."
            ) from exc


_ensure_isaaclab_on_path()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Could not import isaaclab.app.AppLauncher. Run inside the Isaac Lab kit shell (./isaaclab.sh --run)."
        ) from exc

    app = AppLauncher(headless=args.headless)
    simulation_app = app.app

    # Local imports after the app is created.
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim import PhysxCfg, SimulationContext
    from isaaclab.scene import InteractiveScene
    from rsl_rl.env import VecEnv

    from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import UnigraspTransformerGraspEnv
    from legged_lab.utils.env_utils.unigrasptransformer_scene import UniGraspSceneCfg

    class MinimalTestEnv(VecEnv):
        """Minimal VecEnv just to exercise scene construction."""

        def __init__(self, cfg, headless):
            self.cfg = cfg
            self.headless = headless
            self.device = cfg.device
            self.num_envs = cfg.scene.num_envs
            self.physics_dt = cfg.sim.dt
            self.step_dt = cfg.sim.decimation * cfg.sim.dt

            sim_cfg = sim_utils.SimulationCfg(
                device=cfg.device,
                dt=cfg.sim.dt,
                render_interval=cfg.sim.decimation,
                physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
            )
            self.sim = SimulationContext(sim_cfg)
            scene_cfg = UniGraspSceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
            self.scene = InteractiveScene(scene_cfg)
            self.sim.reset()

        # Not used in this smoke test; required by VecEnv interface.
        def get_observations(self):
            raise NotImplementedError("Not implemented for scene-construction smoke test.")

        def reset(self):
            raise NotImplementedError("Not implemented for scene-construction smoke test.")

        def step(self, actions):
            raise NotImplementedError("Not implemented for scene-construction smoke test.")

    grasp_cfg = UnigraspTransformerGraspEnv()
    grasp_cfg.scene.num_envs = args.num_envs

    # Minimal spawn assets so UniGraspSceneCfg receives RigidObjectCfgs.
    table_spawn = grasp_cfg.scene.table_spawn
    object_spawn = grasp_cfg.scene.object_spawn

    grasp_cfg.scene.table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=table_spawn.get("size") or (0.6, 0.6, 0.03),
            translation=table_spawn.get("pos") or (0.0, 0.0, 0.25),
            orientation=table_spawn.get("rot_xyzw") or (0.0, 0.0, 0.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )
    # Use a simple cuboid for the grasp object to avoid dataset dependency.
    grasp_cfg.scene.grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=object_spawn.get("size") or (0.1, 0.1, 0.1),
            translation=object_spawn.get("pos") or (0.0, 0.0, 0.5),
            orientation=object_spawn.get("rot_xyzw") or (0.0, 0.0, 0.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )
    # Disable dataset-based spawn to keep the test self-contained.
    grasp_cfg.scene.object_spawn["enable"] = False

    env = MinimalTestEnv(cfg=grasp_cfg, headless=args.headless)
    scene_cfg = env.scene.cfg
    scene = env.scene

    robot = scene["robot"]
    obj = scene["object"]
    tbl = getattr(scene, "table", None)

    print("[INFO] Scene constructed.")
    print(f"  - robot prim: {scene_cfg.robot.prim_path}")
    if tbl is not None:
        print(f"  - table prim: {scene_cfg.table.prim_path}")
    print(f"  - object prim: {scene_cfg.object.prim_path}")
    print(f"  - num_envs: {scene_cfg.num_envs}")
    print(f"  - robot DOF count: {robot.num_dof}")
    print(f"  - object exists: {obj is not None}")

    simulation_app.close()


if __name__ == "__main__":
    main()
