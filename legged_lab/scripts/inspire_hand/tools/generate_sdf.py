#!/usr/bin/env python3
"""Generate signed distance fields (SDFs) for GraspXL affordance meshes.

This script walks the PartNet-derived dataset (``dataset/mixed_train`` by default),
loads the affordance/non-affordance meshes for each object, builds a dense signed
distance field, and stores the result under ``dataset/grasp_usd/<object>/`` as
``*_sdf.npz``. The companion ``metadata.json`` file is updated with the new SDF
paths so the runtime can load them alongside the converted USD assets.

Example usage (inside the Isaac Lab kit shell):

```
cd ~/IsaacLab
./isaaclab.sh --run python /path/to/repo/legged_lab/scripts/inspire_hand/tools/generate_sdf.py
```

To validate a specific object after generation:

```
./isaaclab.sh --run python /path/to/repo/legged_lab/scripts/inspire_hand/tools/generate_sdf.py --validate Mug_46ed9dad0440c043d33646b0990bb4a_body
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh

DEFAULT_DATASET_ROOT = Path("dataset") / "mixed_train"
DEFAULT_CONVERTED_ROOT = Path("dataset") / "grasp_usd"

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


@dataclass
class SDFResult:
    grid: np.ndarray
    min_bounds: np.ndarray
    max_bounds: np.ndarray

    @property
    def spacing(self) -> np.ndarray:
        return (self.max_bounds - self.min_bounds) / (np.array(self.grid.shape) - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Directory containing the GraspXL PartNet objects (default: dataset/mixed_train).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CONVERTED_ROOT,
        help="Directory where metadata.json lives (default: dataset/grasp_usd).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=40,
        help="SDF grid resolution along each axis (default: 40).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.01,
        help="Bounding-box padding (in meters) when sampling the mesh (default: 1 cm).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of objects to process (for smoke tests).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate SDF files even if they already exist.",
    )
    parser.add_argument(
        "--validate",
        type=str,
        default=None,
        help="Validate a specific object id by sampling points on the mesh and reporting SDF statistics.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Number of surface samples used during validation (default: 128).",
    )
    return parser.parse_args()


def find_mesh(root: Path, candidates: Iterable[tuple[str, Iterable[str]]]) -> Optional[Path]:
    for stem, exts in candidates:
        for ext in exts:
            candidate = root / f"{stem}.{ext}"
            if candidate.exists():
                return candidate
    return None


def compute_sdf(mesh_path: Path, resolution: int, padding: float) -> SDFResult:
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Mesh at {mesh_path} is empty.")

    bounds = mesh.bounds
    min_bounds = bounds[0] - padding
    max_bounds = bounds[1] + padding

    axes = [np.linspace(min_bounds[i], max_bounds[i], resolution) for i in range(3)]
    grid_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    distances = trimesh.proximity.signed_distance(mesh, grid_points)
    if np.any(np.isnan(distances)):
        raise ValueError(f"Signed distance computation failed for {mesh_path}.")

    sdf_grid = distances.reshape(resolution, resolution, resolution).astype(np.float32)
    return SDFResult(grid=sdf_grid, min_bounds=min_bounds, max_bounds=max_bounds)


def save_sdf(result: SDFResult, output_path: Path, mesh_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sdf=result.grid,
        min_bounds=result.min_bounds,
        max_bounds=result.max_bounds,
        mesh=mesh_path.as_posix(),
    )


def update_metadata(meta_path: Path, **entries):
    data = {}
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass
    data.update(entries)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(data, indent=2))


def process_object(
    object_dir: Path,
    output_root: Path,
    resolution: int,
    padding: float,
    force: bool,
) -> None:
    affordance_mesh = find_mesh(object_dir, MESH_CANDIDATES)
    non_affordance_mesh = find_mesh(object_dir, NON_AFFORDANCE_CANDIDATES)

    if affordance_mesh is None and non_affordance_mesh is None:
        print(f"[WARN] No meshes found for {object_dir.name}; skipping.")
        return

    meta_path = output_root / object_dir.name / "metadata.json"
    meta_dir = meta_path.parent

    if affordance_mesh is not None:
        aff_sdf_path = meta_dir / "affordance_sdf.npz"
        if not aff_sdf_path.exists() or force:
            result = compute_sdf(affordance_mesh, resolution, padding)
            save_sdf(result, aff_sdf_path, affordance_mesh)
            print(f"[OK] Affordance SDF -> {aff_sdf_path}")
        update_metadata(meta_path, affordance_sdf=aff_sdf_path.as_posix())

    if non_affordance_mesh is not None:
        non_sdf_path = meta_dir / "non_affordance_sdf.npz"
        if not non_sdf_path.exists() or force:
            result = compute_sdf(non_affordance_mesh, resolution, padding)
            save_sdf(result, non_sdf_path, non_affordance_mesh)
            print(f"[OK] Non-affordance SDF -> {non_sdf_path}")
        update_metadata(meta_path, non_affordance_sdf=non_sdf_path.as_posix())


def trilinear_interpolate(grid: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray, points: np.ndarray):
    resolution = np.array(grid.shape)
    normalized = (points - min_bounds) / (max_bounds - min_bounds) * (resolution - 1)
    normalized = np.clip(normalized, 0.0, resolution - 1)

    idx0 = np.floor(normalized).astype(int)
    idx1 = np.clip(idx0 + 1, 0, resolution - 1)
    frac = normalized - idx0

    c000 = grid[idx0[:, 0], idx0[:, 1], idx0[:, 2]]
    c100 = grid[idx1[:, 0], idx0[:, 1], idx0[:, 2]]
    c010 = grid[idx0[:, 0], idx1[:, 1], idx0[:, 2]]
    c110 = grid[idx1[:, 0], idx1[:, 1], idx0[:, 2]]
    c001 = grid[idx0[:, 0], idx0[:, 1], idx1[:, 2]]
    c101 = grid[idx1[:, 0], idx0[:, 1], idx1[:, 2]]
    c011 = grid[idx0[:, 0], idx1[:, 1], idx1[:, 2]]
    c111 = grid[idx1[:, 0], idx1[:, 1], idx1[:, 2]]

    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def validate_object(object_id: str, converted_root: Path, samples: int):
    meta_path = converted_root / object_id / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found for '{object_id}'. Run the SDF generator first.")

    metadata = json.loads(meta_path.read_text())
    aff_sdf_path = metadata.get("affordance_sdf")
    if aff_sdf_path is None:
        raise FileNotFoundError(f"No affordance SDF recorded for '{object_id}'.")

    sdf_data = np.load(aff_sdf_path)
    grid = sdf_data["sdf"]
    min_bounds = sdf_data["min_bounds"]
    max_bounds = sdf_data["max_bounds"]
    mesh_path = Path(sdf_data["mesh"].item())

    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    surface_samples, _ = trimesh.sample.sample_surface(mesh, samples)
    sdf_vals = trilinear_interpolate(grid, min_bounds, max_bounds, surface_samples)

    print(f"[VALIDATE] object={object_id}")
    print(f"  mesh : {mesh_path}")
    print(f"  SDF  : {aff_sdf_path}")
    print(f"  samples : {samples}")
    print(
        "  abs(distance) stats:",
        f"min={np.abs(sdf_vals).min():.6f}",
        f"max={np.abs(sdf_vals).max():.6f}",
        f"mean={np.abs(sdf_vals).mean():.6f}",
    )


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if args.validate:
        validate_object(args.validate, output_root, args.samples)
        return

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    object_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir())
    if args.limit is not None:
        object_dirs = object_dirs[: args.limit]

    if not object_dirs:
        print("[INFO] No objects found. Nothing to do.")
        return

    for obj_dir in object_dirs:
        try:
            process_object(obj_dir, output_root, args.resolution, args.padding, args.force)
        except Exception as exc:
            print(f"[ERROR] Failed to process {obj_dir.name}: {exc}")


if __name__ == "__main__":
    main()
