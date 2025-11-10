#!/usr/bin/env python3
"""Preview a configured UniGraspTransformer scene with the Inspire Hand."""

from __future__ import annotations

import argparse
from pathlib import Path
import os


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
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments to spawn.",
    )
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=None,
        help="Optional YAML file describing a fixed object spawn configuration.",
    )
    parser.add_argument(
        "--ugtf-root",
        type=Path,
        default=Path("/home/chizhang/projects/Tiangong_isaac_rl/dataset/unigrasptransformer_asset/meshdatav3_scaled"),
        help=(
            "Root of UniDexGrasp++ meshes (meshdatav3_scaled)."
        ),
    )
    parser.add_argument(
        "--ugtf-object",
        type=str,
        default=None,
        help=(
            "Optional object relative path under --ugtf-root, e.g. 'core/bottle-xxxx'."
            " If omitted, a random object is chosen."
        ),
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default="subset_core10",
        help="Subset directory name created by the conversion tool (under --ugtf-root)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="core",
        help="Category inside the subset to sample from (e.g., core/sem/ddg/mujoco)",
    )
    # No separate USD root needed when USDs are written into subset object folders
    return parser.parse_args()


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

    import torch

    from legged_lab.envs.unigrasptransformer.unigrasptransformer_env import UniGraspTransformerEnv
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_cfg import (
        UniGraspTransformerEnvCfg,
        UniGraspTransformerGraspSceneCfg,
    )
    from legged_lab.envs.unigrasptransformer.spawn_cfg import UniGraspTransformerSpawnCfg
    from legged_lab.assets.inspirehand.object_library import GraspObjectInfo

    spawn_cfg = UniGraspTransformerSpawnCfg(
        config_path=(args.config_yaml.expanduser().resolve().as_posix() if args.config_yaml else None)
    )

    # If a UniGraspTransformer subset is present, randomly pick an object and load its pre-converted USD via metadata.json
    assets_root = args.ugtf_root.expanduser().resolve()
    subset_dir = assets_root / args.subset_name / args.category


    if subset_dir.exists() and subset_dir.is_dir():
        import json as _json
        import random as _random

        if args.ugtf_object is not None:
            chosen_dir = (subset_dir / args.ugtf_object).resolve()
            if not chosen_dir.exists():
                raise FileNotFoundError(f"UGTF subset object dir not found: {chosen_dir}")
        else:
            obj_dirs = [p for p in subset_dir.iterdir() if p.is_dir()]
            if not obj_dirs:
                raise RuntimeError(f"No objects under subset: {subset_dir}")
            chosen_dir = _random.choice(obj_dirs)

        # Prefer metadata next to the subset object folder (new converter writes here)
        meta_path = chosen_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing metadata.json next to subset object: {meta_path}.\n"
                f"Re-run the converter without --usd-output so USDs are written into subset folders."
            )
        metadata = _json.loads(meta_path.read_text())
        static_usd = metadata.get("static_usd")
        if not static_usd:
            raise RuntimeError(f"metadata.json for {chosen_dir.name} lacks 'static_usd'")
        static_usd_path = Path(static_usd).expanduser().resolve()
        if not static_usd_path.exists():
            raise FileNotFoundError(f"static_usd path does not exist: {static_usd_path}")

        # Override spawn to use the pre-converted USD (no SDFs)
        spawn_cfg.use_object_library = False
        spawn_cfg.grasp_object.static_usd = static_usd_path.as_posix()
        spawn_cfg.grasp_object.affordance_sdf = None
        spawn_cfg.grasp_object.non_affordance_sdf = None
        spawn_cfg.grasp_object.affordance_sdf_data = None
        spawn_cfg.grasp_object.non_affordance_sdf_data = None

        override = GraspObjectInfo(
            object_id=chosen_dir.name,
            category=chosen_dir.parent.name,
            root_dir=chosen_dir,
            urdf=None,
            fixed_base_urdf=None,
            affordance_mesh=None,
            non_affordance_mesh=None,
            lowest_point=None,
            static_usd=static_usd_path,
            affordance_usd=None,
            non_affordance_usd=None,
            affordance_sdf=None,
            non_affordance_sdf=None,
        )
        spawn_cfg._override_object_info = override

    scene_cfg = UniGraspTransformerGraspSceneCfg(spawn=spawn_cfg, num_envs=args.num_envs)
    env_cfg = UniGraspTransformerEnvCfg(scene=scene_cfg)

    env = UniGraspTransformerEnv(env_cfg, headless=args.headless)

    spawned_name = getattr(env._current_object, "object_id", None)
    if spawned_name is None:
        spawned_name = spawn_cfg.grasp_object.object_id or "unknown"

    print(f"[INFO] Spawned UniGraspTransformer object: {spawned_name}")

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    try:
        while True:
            env.step(actions)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Shutting down...")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
