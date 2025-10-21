#!/usr/bin/env python3
"""Convert a URDF hand asset into a USD using Isaac Lab's URDF importer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path, help="Path to the source URDF file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to write the USD. Defaults to the URDF directory.",
    )
    parser.add_argument(
        "--usd-name",
        type=str,
        default=None,
        help="Optional name of the output USD file. Defaults to the URDF stem with .usd extension.",
    )
    parser.add_argument(
        "--stage-path",
        type=str,
        default="/World/hand",
        help="Prim path used when instantiating the articulation (default: /World/hand).",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run the importer without GPU acceleration (sets WARP_FORCE_CPU=1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("WARP_FORCE_CPU", "1")
        os.environ.setdefault("OMNI_KIT_DISABLE_GPU", "1")
        os.environ.setdefault("ISAAC_SIM_USE_CPU", "1")
        os.environ.setdefault("CARB_DISABLE_GPU", "1")
        os.environ.setdefault("PHYSX_USE_GPU", "0")

    from isaaclab.app import AppLauncher

    launcher_kwargs = {"headless": True}
    if args.cpu_only:
        launcher_kwargs["device"] = "cpu"

    launch = AppLauncher(**launcher_kwargs)
    simulation_app = launch.app

    try:
        import carb
        import omni.usd
        from isaaclab.sim.importers import urdf
        from isaaclab.sim.importers.urdf import UrdfImporterCfg
    except Exception as exc:  # pragma: no cover - runtime guard
        simulation_app.close()
        raise RuntimeError(
            "URDF importer dependencies unavailable; run this script inside the Isaac Lab kit shell."
        ) from exc

    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.exists():
        simulation_app.close()
        raise FileNotFoundError(f"URDF path does not exist: {urdf_path}")

    output_dir = args.output.expanduser().resolve() if args.output is not None else urdf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    usd_name = args.usd_name or f"{urdf_path.stem}.usd"
    usd_path = output_dir / usd_name

    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().Clear()
    stage_def = stage.DefinePrim("/World")
    assert stage_def.IsValid()

    cfg = UrdfImporterCfg(
        urdf_path=str(urdf_path),
        import_inertia_tensor=True,
        merge_fixed_joints=False,
        fix_base=False,
        make_instanceable=False,
        articulation_name=args.stage_path,
    )
    importer = urdf.UrdfImporter(cfg)
    prim = importer.import_articulation(stage, args.stage_path)
    if not prim.IsValid():
        simulation_app.close()
        raise RuntimeError("URDF import failed; check the log for details.")

    omni.usd.get_context().save_as_stage(str(usd_path))
    carb.log_info(f"Saved USD to {usd_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
