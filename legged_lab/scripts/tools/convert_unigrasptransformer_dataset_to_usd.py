#!/usr/bin/env python3
"""Create a small UniGraspTransformer subset and convert selected meshes to USD.

This tool:
- Scans a UniDexGrasp++ mesh root (e.g., meshdatav3_scaled/core)
- Randomly picks N object folders that contain `coacd/decomposed_<scale>.obj`
- Creates a subset directory under the same root using symlinks (fallback to copy)
- Converts one OBJ per selected object to USD for Isaac Sim previews
- Writes a minimal metadata.json with `static_usd` for each object

Run inside the Isaac Lab kit shell so converters are available, e.g.:

  ./isaaclab.sh --run python legged_lab/scripts/tools/convert_unigrasptransformer_dataset_to_usd.py \
    --ugtf-root /home/chizhang/projects/Tiangong_isaac_rl/dataset/unigrasptransformer_asset/meshdatav3_scaled \
    --count 10 --subset-name subset_core10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional


def _ensure_isaaclab_on_path() -> None:
    if "isaaclab" in sys.modules:
        return
    source_hint = os.environ.get("ISAACLAB_SOURCE")
    candidate_paths = []
    if source_hint:
        candidate_paths.append(Path(source_hint))
        candidate_paths.append(Path(source_hint) / "isaaclab")
    home_root = Path.home() / "IsaacLab" / "source"
    candidate_paths.append(home_root)
    candidate_paths.append(home_root / "isaaclab")
    for path in candidate_paths:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ugtf-root",
        type=Path,
        default=Path("/home/chizhang/projects/Tiangong_isaac_rl/dataset/unigrasptransformer_asset/meshdatav3_scaled"),
        help="Root of UniGraspTransformer mesh dataset (meshdatav3_scaled)",
    )
    parser.add_argument("--category", type=str, default="core", help="Category folder to sample from (core/sem/ddg/mujoco)")
    parser.add_argument("--count", type=int, default=10, help="Number of objects to sample")
    parser.add_argument("--subset-name", type=str, default="subset_core10", help="Name of subset directory to create under the root")
    parser.add_argument(
        "--usd-output",
        type=Path,
        default=None,
        help="Optional output directory for USDs. If omitted, USDs + metadata.json are written into each subset object folder.",
    )
    parser.add_argument("--scale", type=str, default=None, help="Preferred scale tag to convert (e.g., 010, 012). Falls back to any.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--keep-mesh-usd",
        action="store_true",
        help="Keep the intermediate mesh USDs. By default only the final <object>_static.usd is kept.",
    )
    parser.add_argument("--force", action="store_true", help="Re-convert even if USD already exists")
    return parser.parse_args()


def _list_candidates(cat_root: Path) -> list[Path]:
    candidates: list[Path] = []
    if not cat_root.exists():
        return candidates
    for obj_dir in sorted(p for p in cat_root.iterdir() if p.is_dir()):
        coacd = obj_dir / "coacd"
        if not coacd.is_dir():
            continue
        meshes = list(coacd.glob("decomposed_*.obj"))
        if meshes:
            candidates.append(obj_dir)
    return candidates


def _choose_mesh(obj_dir: Path, scale_hint: Optional[str]) -> tuple[Path, str]:
    meshes = sorted((obj_dir / "coacd").glob("decomposed_*.obj"))
    if not meshes:
        raise FileNotFoundError(f"No decomposed_*.obj found under {obj_dir}/coacd")
    if scale_hint:
        for m in meshes:
            if m.stem.endswith(scale_hint):
                return m, scale_hint
    # Prefer common scales
    pref = ["010", "012", "015", "008", "006"]
    for tag in pref:
        for m in meshes:
            if m.stem.endswith(tag):
                return m, tag
    # Fallback to first
    m = meshes[0]
    return m, m.stem.split("_")[-1]


def _copy_subset(src: Path, dst: Path) -> None:
    # Always create a real copy to safely write USD/metadata inside the subset without touching the original dataset
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _convert_obj_to_usd(mesh_path: Path, out_dir: Path, force: bool) -> Path:
    from isaaclab.sim.converters.mesh_converter import MeshConverter
    from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg

    out_dir.mkdir(parents=True, exist_ok=True)
    usd_name = f"{mesh_path.parent.parent.name}_{mesh_path.stem}.usd"
    cfg = MeshConverterCfg(
        asset_path=mesh_path.as_posix(),
        usd_dir=out_dir.as_posix(),
        usd_file_name=usd_name,
        make_instanceable=False,
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        rigid_props=None,
        mass_props=None,
        force_usd_conversion=force,
    )
    converter = MeshConverter(cfg)
    return Path(converter.usd_path)


def _bundle_static_usd(mesh_usd_paths: list[Path], out_dir: Path, object_name: str, force: bool) -> Path:
    """Create a single static USD that references the mesh USD(s) and has RigidBody API applied.

    Returns the path to `<out_dir>/<object_name>_static.usd`.
    """
    from isaaclab.sim.schemas import schemas, schemas_cfg
    from pxr import Usd, UsdGeom

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{object_name}_static.usd"
    if output_path.exists() and not force:
        return output_path

    stage = Usd.Stage.CreateNew(output_path.as_posix())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_xform = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root_xform.GetPrim())

    for idx, mesh_usd in enumerate(mesh_usd_paths):
        part_prim = stage.DefinePrim(f"/Object/part_{idx}", "Xform")
        part_prim.GetReferences().AddReference(mesh_usd.as_posix())

    # Apply RigidBody properties so Isaac Lab's RigidObject can find a rigid body
    schemas.define_rigid_body_properties(
        "/Object",
        schemas_cfg.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
            retain_accelerations=False,
        ),
        stage=stage,
    )

    stage.Save()
    return output_path


def main():
    _ensure_isaaclab_on_path()
    args = parse_args()

    root = args.ugtf_root.expanduser().resolve()
    cat_root = root / args.category
    subset_root = root / args.subset_name / args.category
    usd_root: Optional[Path]
    if args.usd_output is not None:
        usd_root = args.usd_output.expanduser().resolve()
    else:
        usd_root = None

    if not cat_root.exists():
        raise FileNotFoundError(f"Category root not found: {cat_root}")

    random.seed(args.seed)
    all_objs = _list_candidates(cat_root)
    print(f"[INFO] Found {len(all_objs)} candidate objects under: {cat_root}")
    if not all_objs:
        print("[INFO] No candidate objects found. Nothing to do.")
        return
    picks = random.sample(all_objs, k=min(args.count, len(all_objs)))
    print("[INFO] Selected objects:")
    for p in picks:
        print(f"  - {p.name}")

    # Launch Isaac Lab app for any USD operations that need it
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Could not import isaaclab.app.AppLauncher. Run inside the Isaac Lab kit shell (./isaaclab.sh --run)."
        ) from exc

    (root / args.subset_name).mkdir(parents=True, exist_ok=True)
    subset_root.mkdir(parents=True, exist_ok=True)
    if usd_root is not None:
        usd_root.mkdir(parents=True, exist_ok=True)

    app = AppLauncher(headless=True)
    simulation_app = app.app

    converted = 0
    try:
        for obj_dir in picks:
            # Create subset entry (symlink if possible)
            dst = subset_root / obj_dir.name
            if not dst.exists():
                _copy_subset(obj_dir, dst)

            # Convert one mesh to USD for quick preview
            mesh_path, scale_tag = _choose_mesh(obj_dir, args.scale)
            print(f"[INFO] Converting {obj_dir.name} (scale {scale_tag}) -> subset folder: {dst}")
            if usd_root is None:
                out_dir = dst  # write into subset object folder
            else:
                out_dir = usd_root / obj_dir.name

            # 1) Convert mesh to USD
            mesh_usd = _convert_obj_to_usd(mesh_path, out_dir, args.force)
            # 2) Bundle into a static rigid USD that references the mesh
            static_usd = _bundle_static_usd([mesh_usd], out_dir, obj_dir.name, args.force)
            # Optionally remove the intermediate mesh USD to keep only one file per object
            if not args.keep_mesh_usd:
                try:
                    mesh_usd.unlink(missing_ok=True)
                except Exception:
                    pass

            # Write minimal metadata.json next to the USD
            metadata = {"static_usd": static_usd.as_posix(), "affordance_usd": None, "non_affordance_usd": None}
            with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f"[OK] {obj_dir.name} -> {static_usd.name}")
            converted += 1
    finally:
        simulation_app.close()
        print(f"[SUMMARY] Converted {converted} / {len(picks)} objects. Subset at: {subset_root}")


if __name__ == "__main__":
    main()
