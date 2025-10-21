#!/usr/bin/env python3
"""Preview a random GraspXL object with the Inspire Hand in a 4-env scene.

The script launches the Isaac Lab simulator, spawns a table, the Inspire Hand
robot, and a grasp object sampled from ``dataset/grasp_usd`` (without any
affordance overlays), and steps the simulation for a short window so you can
verify the assets.
"""

from __future__ import annotations

import argparse
import random
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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="Simulation steps to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for object sampling.",
    )
    parser.add_argument(
        "--dataset-root",
        default=Path("dataset") / "grasp_usd",
        type=Path,
        help="Directory containing converted single-body USD assets.",
    )
    return parser.parse_args()


def _pick_random_usd(dataset_root: Path, seed: int | None) -> Path:
    usd_paths = sorted(
        p
        for p in dataset_root.rglob("*.usd")
        if p.is_file()
    )
    if not usd_paths:
        raise FileNotFoundError(
            f"No .usd assets found in {dataset_root}. Run the GraspXL conversion tool first."
        )
    rng = random.Random(seed)
    return rng.choice(usd_paths)


def _build_env_cfg(usd_path: Path, seed: int | None) -> "GraspXLEnvCfg":
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from legged_lab.envs.graspxl_rl.graspxl_cfg import GraspXLEnvCfg
    from legged_lab.envs.graspxl_rl.spawn_cfg import GraspXLSpawnCfg

    cfg = GraspXLEnvCfg()
    cfg.scene.num_envs = 4
    cfg.scene.env_spacing = 2.0
    if seed is not None:
        cfg.scene.seed = seed

    cfg.scene.spawn = GraspXLSpawnCfg()
    cfg.scene.spawn.hand.align_to_object = True
    cfg.scene.spawn.grasp_object.enable = False
    cfg.scene.spawn.grasp_object.affordance_usd = None
    cfg.scene.spawn.grasp_object.affordance_sdf = None
    cfg.scene.spawn.grasp_object.non_affordance_sdf = None

    cfg.scene.grasp_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path.as_posix(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=3.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=cfg.scene.spawn.grasp_object.pos,
            rot=cfg.scene.spawn.grasp_object.rot,
        ),
    )
    return cfg


def main():
    args = parse_args()

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Could not import isaaclab.app.AppLauncher. Run this script inside the Isaac Lab kit shell (./isaaclab.sh --run)."
        ) from exc

    app = AppLauncher(headless=args.headless)
    simulation_app = app.app

    # The following imports rely on Omniverse plugins, so defer until
    # after the AppLauncher has brought up the Kit app.
    import torch

    from legged_lab.envs.graspxl_rl.graspxl_env import GraspXLEnv
    from legged_lab.envs.graspxl_rl.graspxl_cfg import GraspXLEnvCfg

    usd_path = _pick_random_usd(args.dataset_root.expanduser().resolve(), args.seed)
    cfg = _build_env_cfg(usd_path, args.seed)

    env = GraspXLEnv(cfg, headless=args.headless)
    print(f"[INFO] Spawned GraspXL grasp object: {usd_path.name}")

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(args.steps):
        env.step(actions)

    print("[SUCCESS] Completed GraspXL preview run.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
