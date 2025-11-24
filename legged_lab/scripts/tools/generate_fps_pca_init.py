#!/usr/bin/env python3
"""Generate FPS point clouds, PCA axes, and init metadata next to USDs for a UGTF subset.

This script does NOT require Isaac Lab. It scans the subset created by the USD
converter and writes the following per-object files into the same folder where
the USD lives:

- pc_fps<points>_<scale>.npy
- pca_axes_<scale>.npy
- object_init_<scale>.pkl
- metadata.json (updated to include the above)

It expects the subset layout created by the converter:
  <ugtf-root>/<subset-name>/<category>/<object>/
or a centralized output via --usd-output:
  <usd-output>/<object>/

Usage examples:
  # After running the USD converter (subset in-place)
  python legged_lab/scripts/tools/convert_unigrasptransformer_dataset_legacy.py \
    --ugtf-root /home/chizhang/projects/Tiangong_isaac_rl/dataset/unigrasptransformer_asset/meshdatav3_scaled \
    --subset-name subset_core10 --points 1024 --force

  # If USDs are centralized under dataset/grasp_usd
  python legged_lab/scripts/tools/convert_unigrasptransformer_dataset_legacy.py \
    --usd-output dataset/grasp_usd --points 1024 --force
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh
import pickle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ugtf-root",
        type=Path,
        default=Path("dataset/unigrasptransformer_asset/meshdatav3_scaled"),
        help="UGTF mesh root (used to locate <subset-name> if --usd-output is not set)",
    )
    p.add_argument("--subset-name", type=str, default="subset_core10", help="Subset folder name under --ugtf-root")
    p.add_argument(
        "--categories",
        type=str,
        default="core,sem,ddg,mujoco",
        help="Comma-separated list of categories under the subset to scan",
    )
    p.add_argument(
        "--usd-output",
        type=Path,
        default=None,
        help="If set, scan this USD output root (<usd-output>/<object>/) instead of the subset path.",
    )
    p.add_argument("--points", type=int, default=1024, help="Number of FPS points to save per object")
    p.add_argument("--force", action="store_true", help="Overwrite existing FPS/PCA/INIT if present")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of objects to process")
    p.add_argument(
        "--scale",
        type=str,
        default="010",
        help="Preferred coacd scale tag to sample (e.g., 010, 012). Falls back if unavailable.",
    )
    return p.parse_args()


def _list_objects_from_subset(root: Path, subset_name: str, categories: Iterable[str]) -> list[Path]:
    base = root / subset_name
    objs: list[Path] = []
    for cat in categories:
        cat_dir = base / cat
        if not cat_dir.exists():
            continue
        for obj_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            objs.append(obj_dir)
    return objs


def _list_objects_from_usd_root(usd_root: Path) -> list[Path]:
    if not usd_root.exists():
        return []
    return [p for p in sorted(usd_root.iterdir()) if p.is_dir()]


def _choose_mesh(obj_dir: Path, scale_hint: Optional[str] = None) -> tuple[Path, str]:
    coacd = obj_dir / "coacd"
    if not coacd.exists():
        # In centralized USD output, there may be no coacd; fallback to original subset copy if present
        raise FileNotFoundError(f"Missing coacd folder under {obj_dir}")
    meshes = sorted(coacd.glob("decomposed_*.obj"))
    if not meshes:
        raise FileNotFoundError(f"No decomposed_*.obj under {coacd}")
    if scale_hint:
        for mesh in meshes:
            if mesh.stem.endswith(scale_hint):
                return mesh, scale_hint
    pref = ["010", "012", "015", "008", "006"]
    for tag in pref:
        for mesh in meshes:
            if mesh.stem.endswith(tag):
                return mesh, tag
    mesh_path = meshes[0]
    scale_tag = mesh_path.stem.split("_")[-1]
    return mesh_path, scale_tag


def _fps(points: np.ndarray, k: int) -> np.ndarray:
    n = points.shape[0]
    k = min(k, n)
    idx = np.zeros(k, dtype=np.int64)
    idx[0] = np.random.randint(0, n)
    dists = np.full(n, np.inf, dtype=np.float64)
    last = points[idx[0]]
    for i in range(1, k):
        diff = points - last
        d = np.einsum("ij,ij->i", diff, diff)
        dists = np.minimum(dists, d)
        idx[i] = int(np.argmax(dists))
        last = points[idx[i]]
    return idx


def process_object(obj_dir: Path, out_dir: Path, points: int, force: bool, scale_hint: Optional[str]) -> None:
    # Find decomposed OBJ in the subset copy (we assume subset holds the original coacd)
    try:
        mesh_path, scale_tag = _choose_mesh(obj_dir, scale_hint)
    except Exception as e:
        print(f"[WARN] Skip {obj_dir.name}: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pc_out = out_dir / f"pc_fps{points}_{scale_tag}.npy"
    pca_out = out_dir / f"pca_axes_{scale_tag}.npy"
    init_out = out_dir / f"object_init_{scale_tag}.pkl"

    if not force and pc_out.exists() and pca_out.exists() and init_out.exists():
        print(f"[OK] {obj_dir.name} already has FPS/PCA/INIT; skipping.")
        return

    try:
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        sample_count = max(points * 5, points + 512)
        pts, _ = trimesh.sample.sample_surface(mesh, sample_count)
    except Exception as e:
        print(f"[WARN] Sampling failed for {obj_dir.name}: {e}")
        return

    # FPS and save
    idx = _fps(pts, points)
    pc_fps = pts[idx, :].astype(np.float32)
    np.save(pc_out.as_posix(), pc_fps)

    # PCA axes
    pts_c = pc_fps - pc_fps.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(pts_c, full_matrices=False)
    axes = vh.astype(np.float32)
    np.save(pca_out.as_posix(), axes)

    # Init states (yaw-only)
    import math

    def yaw_quat(theta: float) -> np.ndarray:
        return np.array([0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)], dtype=np.float32)

    test_yaws = [2 * math.pi * i / 16 for i in range(16)]
    train_yaws = [2 * math.pi * np.random.rand() for _ in range(64)]
    pos0 = np.zeros(3, dtype=np.float32)
    test_states = np.stack([np.concatenate([pos0, yaw_quat(t)]) for t in test_yaws], axis=0)
    train_states = np.stack([np.concatenate([pos0, yaw_quat(t)]) for t in train_yaws], axis=0)
    with open(init_out.as_posix(), "wb") as f:
        pickle.dump({"train": train_states, "test": test_states}, f)

    # Update metadata.json in out_dir
    meta_path = out_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    meta.update({
        "pc_fps": pc_out.as_posix(),
        "pca_axes": pca_out.as_posix(),
        "object_init": init_out.as_posix(),
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[OK] {obj_dir.name} -> FPS({pc_out.name}), PCA({pca_out.name}), INIT({init_out.name})")


def main():
    args = parse_args()

    if args.usd_output is not None:
        # Centralized USD root: each object has a folder under this path
        base_objects = _list_objects_from_usd_root(args.usd_output.expanduser().resolve())
        subset_root = args.usd_output.expanduser().resolve()
    else:
        # Subset path: <ugtf-root>/<subset-name>/<category>/<object>
        cats = [s for s in (args.categories or "").split(",") if s]
        base_objects = _list_objects_from_subset(args.ugtf_root.expanduser().resolve(), args.subset_name, cats)
        subset_root = args.ugtf_root.expanduser().resolve() / args.subset_name

    if args.limit is not None:
        base_objects = base_objects[: args.limit]

    if not base_objects:
        print("[INFO] No objects found to process.")
        return

    for obj_dir in base_objects:
        # out_dir is where USD and metadata.json should live
        out_dir = obj_dir if args.usd_output is None else (args.usd_output.expanduser().resolve() / obj_dir.name)
        process_object(obj_dir, out_dir, args.points, args.force, args.scale)

    print(f"[SUMMARY] Processed {len(base_objects)} objects in {subset_root}")


if __name__ == "__main__":
    main()
