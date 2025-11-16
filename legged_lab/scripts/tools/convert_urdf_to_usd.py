#!/usr/bin/env python3
"""Convert a URDF articulation to a cached USD using Isaac Lab's UrdfConverter.

Run this inside the Isaac Lab kit shell (``./isaaclab.sh --run``), for example:

```
python legged_lab/scripts/tools/convert_urdf_to_usd.py \
    --urdf-path legged_lab/assets/hand_assets/urdf/shadow_hand_description/shadowhand_with_fingertips.urdf
```

By default the script mirrors the ShadowHand settings used in ``SHADOW_HAND_CFG`` and writes the
USD to ``<urdf_dir>/usd_cache``. Override the CLI flags to reuse it for other hands/robots.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "legged_lab").exists():
            return candidate
    return here.parent


_REPO_ROOT = _find_repo_root()
_DEFAULT_URDF = (
    _REPO_ROOT
    / "legged_lab"
    / "assets"
    / "hand_assets"
    / "urdf"
    / "shadow_hand_description"
    / "shadowhand_with_fingertips.urdf"
)


def _ensure_isaaclab_on_path() -> None:
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


def _resolve_with_repo_fallback(path: Path) -> Path:
    """Resolve ``path`` relative to CWD first, then fall back to repo root."""

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()

    cwd_candidate = (Path.cwd() / expanded).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (_REPO_ROOT / expanded).resolve()
    if repo_candidate.exists():
        return repo_candidate

    return cwd_candidate


def _sanitize_usd_inertias(usd_path: Path, *, min_component: float = 1e-6) -> None:
    """Ensure every rigid body in the USD has strictly positive mass/inertia."""

    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path.as_posix())
    dirty = False
    physx_mass_api_cls = getattr(PhysxSchema, "PhysxMassPropertiesAPI", None)
    missing_physx_api = physx_mass_api_cls is None
    for prim in stage.Traverse():
        if not (UsdPhysics.RigidBodyAPI(prim) or PhysxSchema.PhysxRigidBodyAPI(prim)):
            continue

        mass_api = UsdPhysics.MassAPI.Apply(prim)

        mass_attr = mass_api.GetMassAttr()
        mass = mass_attr.Get()
        if mass is None or mass <= 0.0:
            mass_attr.Set(max(1.0, min_component))
            dirty = True

        inertia_attr = mass_api.GetDiagonalInertiaAttr()
        inertia = inertia_attr.Get()
        if inertia is None:
            inertia = (min_component, min_component, min_component)
            inertia_attr.Set(inertia)
            dirty = True
        else:
            updated = tuple(max(abs(val), min_component) for val in inertia)
            if updated != inertia:
                inertia_attr.Set(updated)
                dirty = True

        if physx_mass_api_cls is not None:
            physx_mass_api = physx_mass_api_cls.Apply(prim)
            tensor_attr = physx_mass_api.GetMassSpaceInertiaTensorAttr()
        else:
            attr_name = "physxMassProperties:massSpaceInertiaTensor"
            tensor_attr = prim.GetAttribute(attr_name)
            if tensor_attr is None or tensor_attr.GetTypeName() in (None, ""):
                tensor_attr = prim.CreateAttribute(
                    attr_name,
                    Sdf.ValueTypeNames.Float3,
                    custom=False,
                )
        if tensor_attr is not None:
            tensor = tensor_attr.Get()
            if tensor is None:
                tensor_attr.Set((min_component, min_component, min_component))
                dirty = True
            else:
                updated = tuple(max(abs(val), min_component) for val in tensor)
                if updated != tensor:
                    tensor_attr.Set(updated)
                    dirty = True

    if dirty:
        stage.Save()
    if missing_physx_api:
        print(
            "[WARN] PhysxSchema.PhysxMassPropertiesAPI not available; sanitized tensors via generic attribute access."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=_DEFAULT_URDF,
        help=f"URDF path to convert (defaults to ShadowHand: {_DEFAULT_URDF}).",
    )
    parser.add_argument(
        "--usd-dir",
        type=Path,
        default=None,
        help="Directory for cached USD assets. Defaults to '<urdf_dir>/usd_cache'.",
    )
    parser.add_argument(
        "--usd-file-name",
        type=str,
        default=None,
        help="Optional output file name ('.usd' is appended if missing). Defaults to the URDF stem.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the USD even if cached files already exist.",
    )
    parser.add_argument(
        "--fix-base",
        action="store_true",
        help="Fix the base/root link during import.",
    )
    parser.add_argument(
        "--root-link-name",
        type=str,
        default=None,
        help="Optional root link override passed to UrdfConverterCfg.",
    )
    parser.add_argument(
        "--link-density",
        type=float,
        default=800.0,
        help="Fallback density used when URDF links are missing inertia tags (kg/m^3).",
    )
    parser.add_argument(
        "--drive-stiffness",
        type=float,
        default=5000.0,
        help="Joint drive stiffness used when --disable-joint-drive is not set.",
    )
    parser.add_argument(
        "--drive-damping",
        type=float,
        default=25.0,
        help="Joint drive damping used when --disable-joint-drive is not set.",
    )
    parser.add_argument(
        "--drive-type",
        choices=("force", "acceleration"),
        default="force",
        help="Drive type applied to every joint.",
    )
    parser.add_argument(
        "--drive-target",
        choices=("none", "position", "velocity"),
        default="position",
        help="Drive target type applied to every joint.",
    )
    parser.add_argument(
        "--disable-joint-drive",
        action="store_true",
        help="Skip joint drive overrides (use the URDF's native drive settings).",
    )
    parser.add_argument(
        "--collider-type",
        choices=("convex_hull", "convex_decomposition"),
        default="convex_hull",
        help="Collision simplification strategy for imported meshes.",
    )
    parser.add_argument(
        "--collision-from-visuals",
        action="store_true",
        help="Generate collision shapes directly from the visual meshes.",
    )
    parser.add_argument(
        "--replace-cylinders-with-capsules",
        action="store_true",
        help="Convert cylinder collisions to capsules during import.",
    )
    parser.add_argument(
        "--convert-mimic-joints",
        action="store_true",
        help="Convert mimic joints to independent joints.",
    )
    parser.add_argument(
        "--instanceable",
        action="store_true",
        help="Generate an instanceable USD (non-instanceable by default for hands).",
    )
    parser.add_argument(
        "--self-collision",
        dest="self_collision",
        action="store_true",
        help="Enable self-collisions in the imported articulation (default).",
    )
    parser.add_argument(
        "--no-self-collision",
        dest="self_collision",
        action="store_false",
        help="Disable self-collisions.",
    )
    parser.set_defaults(self_collision=True)
    parser.add_argument(
        "--merge-fixed-joints",
        dest="merge_fixed_joints",
        action="store_true",
        help="Merge links connected via fixed joints (default).",
    )
    parser.add_argument(
        "--no-merge-fixed-joints",
        dest="merge_fixed_joints",
        action="store_false",
        help="Keep fixed-joint links separate.",
    )
    parser.set_defaults(merge_fixed_joints=True)
    parser.add_argument(
        "--convert-mimic-to-normal",
        dest="convert_mimic_joints_to_normal_joints",
        action="store_true",
        help="Convert mimic joints to normal joints (alias for --convert-mimic-joints).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Isaac Lab with a viewer instead of running headless.",
    )
    parser.add_argument(
        "--min-inertia",
        type=float,
        default=1e-6,
        help="Clamp inertia tensor components to at least this value when post-processing the USD.",
    )
    return parser.parse_args()


def convert(args: argparse.Namespace) -> Path:
    _ensure_isaaclab_on_path()
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Could not import isaaclab.app.AppLauncher. Run inside the Isaac Lab kit shell (./isaaclab.sh --run)."
        ) from exc

    urdf_path = _resolve_with_repo_fallback(args.urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    usd_dir = args.usd_dir.expanduser().resolve() if args.usd_dir else urdf_path.parent / "usd_cache"
    usd_dir.mkdir(parents=True, exist_ok=True)

    app = AppLauncher(headless=not args.gui)
    simulation_app = app.app

    try:
        from isaaclab.sim.converters.urdf_converter import UrdfConverter
        from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg

        joint_drive_cfg = None
        if not args.disable_joint_drive:
            gains = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args.drive_stiffness,
                damping=args.drive_damping,
            )
            joint_drive_cfg = UrdfConverterCfg.JointDriveCfg(
                drive_type=args.drive_type,
                target_type=args.drive_target,
                gains=gains,
            )

        cfg = UrdfConverterCfg(
            asset_path=urdf_path.as_posix(),
            usd_dir=usd_dir.as_posix(),
            usd_file_name=args.usd_file_name,
            force_usd_conversion=args.force,
            make_instanceable=args.instanceable,
            fix_base=args.fix_base,
            root_link_name=args.root_link_name,
            link_density=args.link_density,
            merge_fixed_joints=args.merge_fixed_joints,
            convert_mimic_joints_to_normal_joints=(
                args.convert_mimic_joints or args.convert_mimic_joints_to_normal_joints
            ),
            joint_drive=joint_drive_cfg,
            collision_from_visuals=args.collision_from_visuals,
            collider_type=args.collider_type,
            self_collision=args.self_collision,
            replace_cylinders_with_capsules=args.replace_cylinders_with_capsules,
        )
        converter = UrdfConverter(cfg)
        usd_path = Path(converter.usd_path)
        try:
            _sanitize_usd_inertias(usd_path, min_component=args.min_inertia)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed to sanitize inertia tensors in {usd_path}: {exc}")
        print(f"[OK] Converted {urdf_path} -> {usd_path}")
        return usd_path
    finally:
        simulation_app.close()


def main() -> None:
    args = parse_args()
    convert(args)


if __name__ == "__main__":
    main()
