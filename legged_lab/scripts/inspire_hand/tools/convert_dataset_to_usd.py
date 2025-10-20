#!/usr/bin/env python3
"""Convert GraspXL meshes into single-body USD assets for Inspire Hand previews.

The original PartNet-derived objects are stored as small URDFs containing separate affordance
(``top_*``) and non-affordance (``bottom_*``) meshes. Isaac Lab's ``RigidObject`` loader expects a
single rigid body, so this script converts both meshes to USD and then bundles them under a single
kinematic rigid prim.

Run this once after copying ``dataset/mixed_train`` (inside the Isaac Lab kit shell, e.g.
``./isaaclab.sh --run python <path>/convert_dataset_to_usd.py``):

```
python legged_lab/scripts/inspire_hand/tools/convert_dataset_to_usd.py
```

Converted USDs are written to ``dataset/grasp_usd/<object>/<object>_static.usd`` by default.
The grasp object library will automatically pick them up once generated.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional
import json

MESH_CANDIDATES = (
    ("top_watertight_tiny", ("obj", "stl")),
    ("top_watertight", ("obj", "stl")),
    ("mesh_top", ("obj", "stl")),
)

NON_AFFORDANCE_CANDIDATES = (
    ("bottom_watertight_tiny", ("obj", "stl")),
    ("bottom_watertight", ("obj", "stl")),
    ("mesh_bottom", ("obj", "stl")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset") / "mixed_train",
        help="Path to the mixed_train dataset (defaults to dataset/mixed_train).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset") / "grasp_usd",
        help="Directory to store converted USD assets (defaults to dataset/grasp_usd).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if USD files already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of objects to convert (useful for smoke tests).",
    )
    return parser.parse_args()


def find_mesh(root: Path, candidates: Iterable[tuple[str, Iterable[str]]]) -> Optional[Path]:
    for stem, exts in candidates:
        for ext in exts:
            candidate = root / f"{stem}.{ext}"
            if candidate.exists():
                return candidate
    return None


def convert_mesh(mesh_path: Path, usd_dir: Path, force: bool) -> Path:
    from isaaclab.sim.converters.mesh_converter import MeshConverter
    from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
    from isaaclab.sim.schemas import schemas_cfg
    from isaaclab.sim.schemas import schemas_cfg

    cfg = MeshConverterCfg(
        asset_path=mesh_path.as_posix(),
        usd_dir=usd_dir.as_posix(),
        usd_file_name=f"{mesh_path.stem}.usd",
        make_instanceable=False,
        collision_props=schemas_cfg.CollisionPropertiesCfg(),
        rigid_props=None,
        mass_props=None,
        force_usd_conversion=force,
    )
    converter = MeshConverter(cfg)
    return Path(converter.usd_path)


def bundle_meshes(object_dir: Path, mesh_usd_paths: list[Path], force: bool) -> Path:
    from isaaclab.sim.schemas import schemas, schemas_cfg
    from pxr import Usd, UsdGeom

    usd_dir = object_dir
    usd_dir.mkdir(parents=True, exist_ok=True)
    output_path = usd_dir / f"{object_dir.name}_static.usd"

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


def convert_object(object_dir: Path, output_root: Path, force: bool) -> Optional[Path]:
    usd_mesh_dir = output_root / object_dir.name / "parts"
    usd_mesh_dir.mkdir(parents=True, exist_ok=True)

    affordance_mesh = find_mesh(object_dir, MESH_CANDIDATES)
    non_afford_mesh = find_mesh(object_dir, NON_AFFORDANCE_CANDIDATES)

    mesh_paths = []
    for mesh in (affordance_mesh, non_afford_mesh):
        if mesh is None:
            continue
        usd_path = convert_mesh(mesh, usd_mesh_dir, force)
        mesh_paths.append(usd_path)
    affordance_usd = None
    non_affordance_usd = None
    for path in mesh_paths:
        stem = path.stem.lower()
        if affordance_usd is None and ("top" in stem or "aff" in stem):
            affordance_usd = path
        elif non_affordance_usd is None and ("bottom" in stem or "non" in stem):
            non_affordance_usd = path

    if not mesh_paths:
        print(f"[WARN] No meshes found for {object_dir.name}; skipping.")
        return None

    output_usd = bundle_meshes(output_root / object_dir.name, mesh_paths, force)
    print(f"[OK] {object_dir.name} -> {output_usd}")

    metadata = {
        "static_usd": output_usd.as_posix(),
        "affordance_usd": affordance_usd.as_posix() if affordance_usd else None,
        "non_affordance_usd": non_affordance_usd.as_posix() if non_affordance_usd else None,
    }
    metadata_path = output_root / object_dir.name / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return output_usd


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
        import toml  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        try:
            import tomllib as toml  # type: ignore  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "Missing 'toml' dependency required by Isaac Lab. Install it via 'pip install toml'."
            ) from exc


def main():
    _ensure_isaaclab_on_path()
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Could not import isaaclab.app.AppLauncher. Run this script inside the Isaac Lab kit shell (./isaaclab.sh --run)."
        ) from exc

    output_root.mkdir(parents=True, exist_ok=True)

    object_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    if args.limit is not None:
        object_dirs = object_dirs[: args.limit]

    if not object_dirs:
        print("[INFO] No object directories found. Nothing to convert.")
        return

    app = AppLauncher(headless=True)
    simulation_app = app.app

    try:
        for obj_dir in object_dirs:
            convert_object(obj_dir, output_root, args.force)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
