#!/usr/bin/env python3
"""Remove the exported root_joint body from a USD articulation."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path, help="Path to the USD file to edit.")
    parser.add_argument(
        "--joint-prim",
        default="/World/handright9253/root_joint",
        help="Prim path of the unwanted root joint (default: /World/handright9253/root_joint).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usd_path = args.usd.expanduser().resolve()
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)

    from pxr import Usd, UsdUtils

    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath(args.joint_prim)
    if not prim:
        raise RuntimeError(f"Joint prim not found: {args.joint_prim}")

    UsdUtils.RemovePrims(stage, [prim])
    stage.GetRootLayer().Save()
    print(f"[INFO] Removed {args.joint_prim} from {usd_path}")


if __name__ == "__main__":
    main()
