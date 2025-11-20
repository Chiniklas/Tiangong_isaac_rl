#!/usr/bin/env python3
"""Generate a detailed report of the UniGraspTransformer hand asset (links, joints, sensors)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


def _ensure_isaaclab_on_path() -> None:
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

from isaaclab.app import AppLauncher


@dataclass
class LinkInfo:
    name: str
    path: str
    parent_name: str | None
    parent_path: str | None


@dataclass
class JointInfo:
    name: str
    path: str
    type_name: str
    parent_path: str | None
    child_path: str | None
    axis: tuple[float, float, float] | None
    limits: tuple[float | None, float | None]


@dataclass
class SpecialPrim:
    name: str
    path: str
    type_name: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run Isaac Lab in headless mode.")
    parser.add_argument(
        "--env-index",
        type=int,
        default=0,
        help="Environment slot to inspect (0-based). The scene will allocate enough envs to cover this index.",
    )
    parser.add_argument(
        "--hand-usd",
        type=Path,
        default="/home/chizhang/Tiangong_isaac_rl/legged_lab/assets/shadow_hand_with_fingertip/shadow_hand_right/shadow_hand_right.usd",
        help="Optionally override the hand asset with a specific USD file.",
    )
    parser.add_argument(
        "--list-dofs",
        action="store_true",
        help="Print the robot joint names (and default DOF state) and exit.",
    )
    return parser.parse_args()


def _resolve_stage() -> "Usd.Stage":
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage is not available (is the simulator running?).")
    return stage


def _basename(path: str | None) -> str | None:
    if not path:
        return None
    if path.endswith("/"):
        path = path[:-1]
    if not path:
        return None
    return path.rsplit("/", 1)[-1]


def _format_axis(axis: tuple[float, float, float] | None) -> str:
    if not axis:
        return "-"
    return f"({axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f})"


def _format_limits(limits: tuple[float | None, float | None]) -> str:
    lo, hi = limits
    if lo is None and hi is None:
        return "-"
    lo_str = f"{lo:+.3f}" if lo is not None else "None"
    hi_str = f"{hi:+.3f}" if hi is not None else "None"
    return f"[{lo_str}, {hi_str}]"


def _collect_links(stage: "Usd.Stage", hand_path: str) -> list[LinkInfo]:
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(hand_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Hand prim '{hand_path}' not found on the stage.")

    records: list[LinkInfo] = []

    def _walk(prim, active_parent: str | None) -> None:
        next_parent = active_parent
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            path = prim.GetPath().pathString
            parent_name = _basename(active_parent)
            records.append(LinkInfo(name=prim.GetName(), path=path, parent_name=parent_name, parent_path=active_parent))
            next_parent = path
        for child in prim.GetChildren():
            _walk(child, next_parent)

    _walk(root, None)
    return records


def _collect_joints(stage: "Usd.Stage", hand_path: str) -> list[JointInfo]:
    from pxr import UsdPhysics

    joint_info: list[JointInfo] = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(hand_path):
            continue
        type_name = prim.GetTypeName()
        if not type_name.endswith("Joint"):
            continue
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        parent_path = body0[0].pathString if body0 else None
        child_path = body1[0].pathString if body1 else None
        axis: tuple[float, float, float] | None = None
        limits: tuple[float | None, float | None] = (None, None)
        if type_name == "PhysicsRevoluteJoint":
            schema = UsdPhysics.RevoluteJoint(prim)
            axis_val = schema.GetAxisAttr().Get()
            axis = (float(axis_val[0]), float(axis_val[1]), float(axis_val[2])) if axis_val is not None else None
            lower = schema.GetLowerLimitAttr().Get()
            upper = schema.GetUpperLimitAttr().Get()
            limits = (float(lower) if lower is not None else None, float(upper) if upper is not None else None)
        elif type_name == "PhysicsPrismaticJoint":
            schema = UsdPhysics.PrismaticJoint(prim)
            axis_val = schema.GetAxisAttr().Get()
            axis = (float(axis_val[0]), float(axis_val[1]), float(axis_val[2])) if axis_val is not None else None
            lower = schema.GetLowerLimitAttr().Get()
            upper = schema.GetUpperLimitAttr().Get()
            limits = (float(lower) if lower is not None else None, float(upper) if upper is not None else None)
        joint_info.append(
            JointInfo(
                name=prim.GetName(),
                path=path,
                type_name=type_name,
                parent_path=parent_path,
                child_path=child_path,
                axis=axis,
                limits=limits,
            )
        )
    return joint_info


def _collect_special_prims(stage: "Usd.Stage", hand_path: str) -> tuple[list[SpecialPrim], list[SpecialPrim]]:
    from pxr import PhysxSchema

    sensors: list[SpecialPrim] = []
    markers: list[SpecialPrim] = []
    sensor_keywords = ("sensor", "touch", "tch")
    marker_keywords = ("marker", "site", "tip")

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(hand_path):
            continue
        name_lower = prim.GetName().lower()
        prim_type = prim.GetTypeName()
        if prim.HasAPI(PhysxSchema.PhysxTriggerAPI):
            sensors.append(SpecialPrim(name=prim.GetName(), path=path, type_name=prim_type, reason="PhysxTriggerAPI"))
            continue
        if any(keyword in name_lower for keyword in sensor_keywords):
            sensors.append(SpecialPrim(name=prim.GetName(), path=path, type_name=prim_type, reason="name match"))
            continue
        if any(keyword in name_lower for keyword in marker_keywords):
            markers.append(SpecialPrim(name=prim.GetName(), path=path, type_name=prim_type, reason="name match"))

    return sensors, markers


def _print_link_hierarchy(links: Sequence[LinkInfo], joints: Sequence[JointInfo]) -> None:
    children: dict[str | None, list[str]] = defaultdict(list)
    for link in links:
        children[link.parent_name].append(link.name)
    for value in children.values():
        value.sort()

    joint_map: dict[str, list[JointInfo]] = defaultdict(list)
    for joint in joints:
        child_name = _basename(joint.child_path)
        if child_name:
            joint_map[child_name].append(joint)

    def _describe_joint(infos: Sequence[JointInfo]) -> str:
        parts = []
        for info in infos:
            short_type = info.type_name.replace("Physics", "").replace("Joint", "") or info.type_name
            parts.append(f"{info.name}:{short_type} axis={_format_axis(info.axis)} limits={_format_limits(info.limits)}")
        return "; ".join(parts)

    def _render(node: str, prefix: str, is_last: bool) -> None:
        branch = "└─" if is_last else "├─"
        joint_text = ""
        if node in joint_map:
            joint_text = f" [{_describe_joint(joint_map[node])}]"
        link = next((item for item in links if item.name == node), None)
        path_text = f" ({link.path})" if link else ""
        print(f"{prefix}{branch} {node}{joint_text}{path_text}")
        next_prefix = prefix + ("   " if is_last else "│  ")
        for idx, child in enumerate(children.get(node, [])):
            _render(child, next_prefix, idx == len(children[node]) - 1)

    print("\n[Link Hierarchy]")
    roots = sorted(children.get(None, []))
    if not roots:
        print("  (no rigid bodies were detected under the hand prim)")
        return
    for idx, root in enumerate(roots):
        _render(root, "", idx == len(roots) - 1)


def _print_joint_table(joints: Sequence[JointInfo]) -> None:
    print("\n[Joint Summary]")
    if not joints:
        print("  (no joints detected)")
        return
    for joint in sorted(joints, key=lambda item: item.name):
        short_type = joint.type_name.replace("Physics", "").replace("Joint", "") or joint.type_name
        parent_name = _basename(joint.parent_path) or "<floating>"
        child_name = _basename(joint.child_path) or "<unassigned>"
        print(
            f" - {joint.name}: {short_type} parent={parent_name} -> child={child_name} "
            f"axis={_format_axis(joint.axis)} limits={_format_limits(joint.limits)} path={joint.path}"
        )


def _print_special_section(title: str, entries: Sequence[SpecialPrim]) -> None:
    print(f"\n[{title}]")
    if not entries:
        print("  (none)")
        return
    for item in sorted(entries, key=lambda entry: entry.path):
        print(f" - {item.name} ({item.type_name}) reason={item.reason} path={item.path}")


def _print_usd_joint_names(usd_path: Path) -> None:
    """Print joint names from a USD file without launching the simulator."""

    from pxr import Usd, UsdPhysics

    if not usd_path.exists():
        raise FileNotFoundError(f"Hand USD not found at {usd_path}")
    stage = Usd.Stage.Open(usd_path.as_posix())
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")
    print(f"\n[Joint Names from {usd_path}]")
    joints = []
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if not type_name.endswith("Joint"):
            continue
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        joints.append(prim.GetName())
    if not joints:
        print("  (no joints detected)")
    else:
        for idx, name in enumerate(sorted(joints)):
            print(f" {idx:02d}: {name}")


def _build_spawn_cfg(hand_usd: Path | None) -> "UniGraspTransformerSpawnCfg":
    """Create a spawn config that ignores config.yaml overrides."""

    cfg = UniGraspTransformerSpawnCfg()
    cfg.use_object_library = False
    cfg.table = UniGraspTransformerTableSpawnCfg()
    cfg.grasp_object = UniGraspTransformerObjectSpawnCfg(enable=False, spawn_mesh=False, show_point_cloud=False, show_pca_axes=False)
    cfg.grasp_object.static_usd = None
    cfg.hand = UniGraspTransformerHandSpawnCfg()
    if hand_usd is not None:
        cfg.hand.asset_path = hand_usd.expanduser().resolve().as_posix()
    return cfg


def _print_joint_names(env, env_index: int) -> None:
    names = getattr(env.robot.data, "joint_names", None)
    if not names:
        print("[WARN] Robot articulation does not expose joint_names.")
        return
    joint_pos_tensor = env.robot.data.joint_pos
    if joint_pos_tensor.shape[0] <= env_index:
        env.reset()
    joint_pos = env.robot.data.joint_pos[env_index].detach().cpu().tolist()
    defaults_attr = getattr(env.robot.data, "default_joint_pos", None)
    default_pos = defaults_attr[env_index].detach().cpu().tolist() if defaults_attr is not None else None
    print("\n[Joint Names / DOF State]")
    for idx, name in enumerate(names):
        cur = joint_pos[idx] if idx < len(joint_pos) else float("nan")
        if default_pos is not None and idx < len(default_pos):
            print(f" {idx:02d}: {name:>16s} current={cur:+.6f} default={default_pos[idx]:+.6f}")
        else:
            print(f" {idx:02d}: {name:>16s} current={cur:+.6f}")


def main() -> None:
    args = parse_args()

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    from legged_lab.envs.unigrasptransformer.spawn_cfg import (
        UniGraspTransformerHandSpawnCfg,
        UniGraspTransformerObjectSpawnCfg,
        UniGraspTransformerSpawnCfg,
        UniGraspTransformerTableSpawnCfg,
    )
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_cfg import (
        UniGraspTransformerEnvCfg,
        UniGraspTransformerGraspSceneCfg,
    )
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_env import UniGraspTransformerEnv

    env = None
    try:
        if args.list_dofs:
            usd_path = args.hand_usd.expanduser().resolve() if args.hand_usd else Path(
                "/home/chizhang/Tiangong_isaac_rl/legged_lab/assets/shadow_hand_unigrasptransformer/open_ai_assets/hand/shadow_hand/shadow_hand.usd"
            )
            _print_usd_joint_names(usd_path)
            return

        spawn_cfg = _build_spawn_cfg(args.hand_usd)
        num_envs = max(args.env_index + 1, 1)
        scene_cfg = UniGraspTransformerGraspSceneCfg(spawn=spawn_cfg, num_envs=num_envs)
        env_cfg = UniGraspTransformerEnvCfg(scene=scene_cfg)
        env = UniGraspTransformerEnv(env_cfg, headless=args.headless)

        hand_path = f"/World/envs/env_{args.env_index}/Robot"
        stage = _resolve_stage()
        links = _collect_links(stage, hand_path)
        joints = _collect_joints(stage, hand_path)
        sensors, markers = _collect_special_prims(stage, hand_path)

        print(f"[INFO] Reporting hand asset at prim '{hand_path}'.")
        print(f"[INFO] Detected {len(links)} rigid bodies, {len(joints)} joints, {len(sensors)} sensors, {len(markers)} markers.")
        _print_link_hierarchy(links, joints)
        _print_joint_table(joints)
        _print_special_section("Sensors / Touch Sites", sensors)
        _print_special_section("Markers / Sites", markers)
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
