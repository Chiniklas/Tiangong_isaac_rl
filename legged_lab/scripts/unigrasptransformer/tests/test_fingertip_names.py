#!/usr/bin/env python3
"""Check whether fingertip body names listed in the spawn config exist in the robot articulation."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify fingertip body names in the UniGraspTransformer robot.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Lab in headless mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    try:
        from legged_lab.envs.unigrasptransformer.unigrasptransformer_cfg import UniGraspTransformerEnvCfg
        from legged_lab.envs.unigrasptransformer.unigrasptransformer_env import UniGraspTransformerEnv

        cfg = UniGraspTransformerEnvCfg()
        env = UniGraspTransformerEnv(cfg, headless=args.headless)

        hand_cfg = cfg.scene.spawn.hand
        name_patterns = list(getattr(hand_cfg, "fingertip_body_exprs", ()))
        if not name_patterns:
            print("Spawn config does not list any fingertip body expressions.")
            return

        all_names = list(env.hand.data.body_names)
        missing = []
        for expr in name_patterns:
            matches = [name for name in all_names if expr in name]
            if matches:
                print(f"[OK] pattern '{expr}' matched: {matches}")
            else:
                missing.append(expr)
                print(f"[MISS] pattern '{expr}' not found in robot body names.")

        if missing:
            print("\nSummary: some fingertip patterns were missing.")
            for expr in missing:
                print(f" - {expr}")
        else:
            print("\nSummary: all fingertip patterns were found.")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
