#!/usr/bin/env python3
"""Visualize a random UniGraspTransformer object in MuJoCo.

This utility scans `dataset/unigrasptransformer_asset/meshdatav3_scaled` for an object
that has a `coacd/decomposed_<scale>.obj` mesh, builds a minimal MJCF model on the fly,
and opens the MuJoCo viewer to preview the object.

Requirements:
- `pip install mujoco` (official DeepMind bindings)
- Working OpenGL context (GLFW) for the viewer

Usage examples:
- python legged_lab/scripts/unigrasptransformer/tests/test_visualize_in_mujoco.py
- python legged_lab/scripts/unigrasptransformer/tests/test_visualize_in_mujoco.py --object sem/Airplane-98a84003006d3c7142a4fe95eae396be --scale 010
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Optional, Tuple


DATASET_ROOT = Path("/home/chizhang/projects/Tiangong_isaac_rl/dataset/unigrasptransformer_asset/meshdatav3_scaled")


def find_candidate_objects(root: Path) -> list[Tuple[Path, list[Path]]]:
    """Return a list of (object_dir, mesh_files) where mesh_files are decomposed_*.obj under coacd.

    The expected layout is `<root>/<category>/<object_code>/coacd/decomposed_<scale>.obj`.
    """
    candidates: list[Tuple[Path, list[Path]]] = []
    if not root.exists():
        return candidates
    # categories like: core/, sem/, ddg/, mujoco/
    for category in [p for p in root.iterdir() if p.is_dir()]:
        for obj_dir in category.iterdir():
            if not obj_dir.is_dir():
                continue
            coacd = obj_dir / "coacd"
            if not coacd.is_dir():
                continue
            meshes = sorted(coacd.glob("decomposed_*.obj"))
            if meshes:
                candidates.append((obj_dir, meshes))
    return candidates


def select_mesh(meshes: list[Path], scale_hint: Optional[str]) -> Path:
    """Choose a decomposed_<scale>.obj, preferring the provided scale hint if available."""
    if scale_hint:
        for m in meshes:
            if m.stem.endswith(scale_hint):
                return m
    return random.choice(meshes)


def build_mjcf_with_mesh(mesh_path: Path) -> str:
    """Create a minimal MJCF that references the given OBJ mesh by absolute path."""
    mesh_file = mesh_path.resolve().as_posix()
    xml = f"""
<mujoco model="ugtf_object_preview">
  <option integrator="RK4" timestep="0.002"/>
  <compiler angle="radian" meshdir="" texturedir="" coordinate="local"/>
  <asset>
    <mesh name="objmesh" file="{mesh_file}"/>
  </asset>
  <worldbody>
    <geom name="ground" type="plane" size="2 2 .1" rgba="0.85 0.9 1 1" contype="1" conaffinity="1"/>
    <light pos="2 2 3" dir="-1 -1 -2" diffuse="1 1 1" specular="0.3 0.3 0.3"/>
    <body name="object" pos="0 0 0.1">
      <geom type="mesh" mesh="objmesh" rgba="0.9 0.6 0.2 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""
    return xml


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT, help="Root of meshdatav3_scaled (can be absolute or repo-relative)")
    parser.add_argument("--object", type=str, default=None, help="Optional object path relative to dataset root (e.g., 'core/bottle-xxxx')")
    parser.add_argument("--scale", type=str, default=None, help="Optional scale hint like '010', '012', etc.")
    parser.add_argument("--no-view", action="store_true", help="Only print the selected object and mesh without opening the viewer.")
    return parser.parse_args()


def _repo_root_from_file() -> Path:
    # This file: <repo>/legged_lab/scripts/unigrasptransformer/tests/test_visualize_in_mujoco.py
    p = Path(__file__).resolve()
    # repo root is 5 parents up from this file
    # tests -> unigrasptransformer -> scripts -> legged_lab -> <repo>
    return p.parents[5]


def _resolve_dataset_root(arg_root: Path) -> Tuple[Optional[Path], list[Path]]:
    """Resolve dataset root trying env var, given path, and repo-relative fallback.

    Returns the first existing Path and the list of candidates checked.
    """
    tried: list[Path] = []
    # 1) environment override
    env_root = os.environ.get("UNIGRASP_DATASET_ROOT")
    if env_root:
        cand = Path(env_root)
        tried.append(cand)
        if cand.exists():
            return cand, tried
    # 2) as provided
    tried.append(arg_root)
    if arg_root.exists():
        return arg_root, tried
    # 3) repo-relative if arg_root is relative
    if not arg_root.is_absolute():
        repo_root = _repo_root_from_file()
        cand = (repo_root / arg_root).resolve()
        tried.append(cand)
        if cand.exists():
            return cand, tried
    # 4) explicit default under repo root
    repo_root = _repo_root_from_file()
    cand = (repo_root / "dataset/unigrasptransformer_asset/meshdatav3_scaled").resolve()
    tried.append(cand)
    if cand.exists():
        return cand, tried
    return None, tried


def main():
    args = parse_args()

    root, tried = _resolve_dataset_root(args.dataset_root)
    if root is None:
        tried_str = "\n  - " + "\n  - ".join(p.as_posix() for p in tried)
        raise SystemExit(
            "Dataset root not found. Tried:" + tried_str +
            "\nSet UNIGRASP_DATASET_ROOT or pass --dataset-root with an absolute path."
        )

    obj_mesh: Optional[Path] = None
    chosen_obj_dir: Optional[Path] = None

    if args.object is not None:
        chosen_obj_dir = (root / args.object).resolve()
        if not chosen_obj_dir.is_dir():
            raise SystemExit(f"Object directory not found: {chosen_obj_dir}")
        coacd = chosen_obj_dir / "coacd"
        meshes = sorted(coacd.glob("decomposed_*.obj"))
        if not meshes:
            raise SystemExit(f"No decomposed_*.obj under: {coacd}")
        obj_mesh = select_mesh(meshes, args.scale)
    else:
        candidates = find_candidate_objects(root)
        if not candidates:
            raise SystemExit(f"No candidate objects found under: {root}")
        chosen_obj_dir, meshes = random.choice(candidates)
        obj_mesh = select_mesh(meshes, args.scale)

    rel_obj = chosen_obj_dir.relative_to(root)
    scale_tag = obj_mesh.stem.split("_")[-1]
    print(f"[INFO] Selected object: {rel_obj.as_posix()} | mesh: {obj_mesh.name} (scale {scale_tag})")

    if args.no_view:
        return

    # Force GUI viewer via GLFW. If your environment lacks GLX/GLFW, this will error deliberately
    # because this script is intended to run with a GUI.
    os.environ.setdefault("MUJOCO_GL", "glfw")
    try:
        import mujoco
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Could not import 'mujoco'. Install it via 'pip install mujoco' and ensure you have a working OpenGL environment."
        ) from exc

    xml = build_mjcf_with_mesh(obj_mesh)

    # Load model and launch viewer if available
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Require the built-in viewer (mujoco>=2.3.3). Fail if not available.
    try:
        from mujoco import viewer as mj_viewer  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "mujoco.viewer is unavailable. Ensure 'pip install mujoco' provides the viewer and that a GUI is available.\n"
            f"Detail: {exc}"
        )

    print("[INFO] Launching MuJoCo viewer. Close the window to exit.")
    with mj_viewer.launch_passive(model, data) as v:  # blocks until closed
        while True:
            # Support different Mujoco viewer APIs across versions
            if hasattr(v, "is_alive"):
                if not v.is_alive():
                    break
            elif hasattr(v, "is_running"):
                if not v.is_running():
                    break
            mujoco.mj_step(model, data)
            v.sync()


if __name__ == "__main__":
    main()
