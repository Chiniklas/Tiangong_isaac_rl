#!/usr/bin/env python3
"""Preview random dataset objects placed on the Inspire Hand table.

The script samples objects from ``dataset/mixed_train`` using the grasp object
library, spawns each one into the InspireHand scene (one at a time), and steps
the simulator so you can verify the meshes visually. By default it previews four
objects chosen uniformly at random. Run
`python legged_lab/scripts/inspire_hand/tools/convert_dataset_to_usd.py`
first to generate single-body USDs for each mesh.
"""

import argparse
import random
from pathlib import Path
from typing import Optional

def _ensure_isaaclab_on_path():
    import os
    import sys
    from pathlib import Path

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
        import toml  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        try:
            import tomllib as toml  # type: ignore  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "Missing 'toml' dependency required by Isaac Lab. Install it via 'pip install toml'."
            ) from exc


_ensure_isaaclab_on_path()

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Could not import isaaclab.app.AppLauncher. Run this script inside the Isaac Lab kit shell (./isaaclab.sh --run)."
    ) from exc

from legged_lab.assets.inspirehand import GraspObjectLibrary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering. Useful for CI smoke tests.",
    )
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Explicit object ids to spawn. Overrides random sampling.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=180,
        help="Simulation steps to run per object.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of objects to preview when sampling randomly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )
    return parser.parse_args()


def _colorize_mesh(prim, color, opacity):
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Gprim):
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([color])
        gprim.CreateDisplayOpacityAttr([opacity])
    for child in prim.GetChildren():
        _colorize_mesh(child, color, opacity)


def _add_affordance_markers(env, obj_info):
    stage = env.scene.stage  # type: ignore[attr-defined]
    object_root = "/World/envs/env_0/Object"

    def _spawn_marker(name: str, usd_path: Path, color_tuple: tuple[float, float, float], opacity: float):
        marker_path = f"{object_root}/{name}"
        if stage.GetPrimAtPath(marker_path):
            stage.RemovePrim(marker_path)
        marker_prim = stage.DefinePrim(marker_path, "Xform")
        marker_prim.GetReferences().AddReference(usd_path.as_posix())
        _colorize_mesh(marker_prim, color_tuple, opacity)

    if obj_info.affordance_usd and obj_info.affordance_usd.exists():
        _spawn_marker("AffordancePreview", obj_info.affordance_usd, (0.2, 0.9, 0.2), 0.35)
    if obj_info.non_affordance_usd and obj_info.non_affordance_usd.exists():
        _spawn_marker("NonAffordancePreview", obj_info.non_affordance_usd, (0.9, 0.2, 0.2), 0.25)


def build_env_cfg(asset_path: Path):
    from legged_lab.envs.inspirehand.grasp_cfg import InspireHandGraspEnvCfg
    import isaaclab.sim as sim_utils

    cfg = InspireHandGraspEnvCfg()
    cfg.scene.num_envs = 4
    cfg.scene.env_spacing = 1.5

    if asset_path.suffix.lower() in {".urdf"}:
        cfg.scene.grasp_object.spawn = sim_utils.UrdfFileCfg(
            asset_path=asset_path.as_posix(),
            fix_base=True,
            link_density=500.0,
            collision_from_visuals=True,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                target_type="none",
                gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=3.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    elif asset_path.suffix.lower() in {".usd", ".usda"}:
        cfg.scene.grasp_object.spawn = sim_utils.UsdFileCfg(
            usd_path=asset_path.as_posix(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=3.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    else:
        raise ValueError(f"Unsupported asset format: {asset_path}")

    cfg.scene.grasp_object.init_state.pos = (0.55, 0.0, 0.73)
    cfg.scene.grasp_object.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    return cfg


def _resolve_asset_path(info) -> Optional[Path]:
    """Return converted USD if available; otherwise skip the object."""

    if info.static_usd is not None and info.static_usd.exists():
        return info.static_usd
    return None


def main():
    args = parse_args()

    library = GraspObjectLibrary()
    available_infos = [info for info in library.all_objects() if info.static_usd]
    if not available_infos:
        raise RuntimeError("No converted USD assets found. Run the conversion tool first.")

    rng = random.Random(args.seed)
    if args.objects:
        object_ids = args.objects
    else:
        sample_count = min(args.num_envs, len(available_infos))
        object_ids = [info.object_id for info in rng.sample(available_infos, k=sample_count)]

    if not object_ids:
        raise RuntimeError("No grasp objects selected for preview.")

    app = AppLauncher(headless=args.headless)
    simulation_app = app.app

    from legged_lab.envs.inspirehand.grasp_env import InspireHandGraspEnv
    import torch

    for obj_id in object_ids:
        obj_info = library.get(obj_id)
        asset_path = _resolve_asset_path(obj_info)
        if asset_path is None:
            print(f"[WARN] Skipping '{obj_id}' (no converted USD found). Run the conversion tool first.")
            continue

        try:
            cfg = build_env_cfg(asset_path)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue

        env = InspireHandGraspEnv(cfg, headless=args.headless)
        _add_affordance_markers(env, obj_info)

        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        for _ in range(args.steps):
            env.step(actions)

        if obj_info.affordance_usd:
            print(f"[INFO] Affordance overlay active ({obj_info.affordance_usd.name})")
        if obj_info.non_affordance_usd:
            print(f"[INFO] Non-affordance overlay active ({obj_info.non_affordance_usd.name})")
        print(f"[INFO] Previewed {obj_id}")
        env.close()

    print("[SUCCESS] Completed preview.")
    simulation_app.close()
    for obj_id in object_ids:
        obj_info = library.get(obj_id)
        asset_path = _resolve_asset_path(obj_info)
        if asset_path is None:
            print(f"[WARN] Skipping '{obj_id}' (no converted USD found). Run the conversion tool first.")
            continue
        asset_pairs.append((obj_id, asset_path))

    if not asset_pairs:
        print("[ERROR] No usable objects found. Aborting preview.")
        simulation_app.close()
        return

    cfg = build_env_cfg(asset_pairs[0][1], len(asset_pairs))
    env = InspireHandGraspEnv(cfg, headless=args.headless)

    stage: Usd.Stage = env.scene.stage  # type: ignore[attr-defined]
    for env_idx, (obj_id, asset_path) in enumerate(asset_pairs):
        prim_path = f"/World/envs/env_{env_idx}/Object"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"[WARN] Could not locate prim '{prim_path}' for env {env_idx}.")
            continue
        refs = prim.GetReferences()
        refs.ClearReferences()
        refs.AddReference(asset_path.as_posix())
        print(f"[INFO] env_{env_idx}: {obj_id} -> {asset_path.name}")

    env.reset(torch.arange(env.num_envs, device=env.device))

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    for _ in range(args.steps):
        env.step(actions)

    print("[SUCCESS] Completed preview.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
