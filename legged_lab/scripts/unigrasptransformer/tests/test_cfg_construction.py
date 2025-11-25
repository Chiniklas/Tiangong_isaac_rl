#!/usr/bin/env python3
"""Smoke test to construct spawn configs and print summaries.

- Loads YAML hyperparameters.
- Builds spawn dicts (table/hand/object).
- Converts spawn dicts into typed cfgs (TableCfg/GraspObjectCfg).
- Instantiates UnigraspTransformerSceneCfg and UnigraspTransformerGraspEnv with those cfgs.
"""

from __future__ import annotations

import sys
from pprint import pprint
from pathlib import Path

def main():
    # Ensure repo root is importable when running via kit python.
    repo_root = Path(__file__).resolve().parents[3]
    if repo_root.as_posix() not in sys.path:
        sys.path.append(repo_root.as_posix())

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import isaaclab.app.AppLauncher. Run inside the Isaac Lab kit shell (isaaclab.sh -p)."
        ) from exc

    app = AppLauncher(headless=True)
    simulation_app = app.app

    from legged_lab.envs.unigrasptransformer.helpers import (
        _build_grasp_object_cfg,
        _build_hand_spawn,
        _build_object_spawn,
        _build_table_cfg,
        _build_table_spawn,
        _load_yaml_cfg,
    )
    from legged_lab.envs.unigrasptransformer.dex_grasp_cfg import (
        UnigraspTransformerGraspEnv,
        UnigraspTransformerSceneCfg,
    )

    spawn_cfg = _load_yaml_cfg("spawn_cfg.yaml")
    weights_cfg = _load_yaml_cfg("weights_cfg.yaml")
    ppo_cfg = _load_yaml_cfg("ppo_cfg.yaml")

    table_spawn = _build_table_spawn(spawn_cfg)
    hand_spawn = _build_hand_spawn(spawn_cfg)
    object_spawn = _build_object_spawn(spawn_cfg)

    print("[INFO] Spawn dicts:")
    pprint({"table": table_spawn, "hand": hand_spawn, "object": object_spawn})

    table_cfg = _build_table_cfg(table_spawn)
    try:
        object_cfg = _build_grasp_object_cfg(object_spawn)
    except ValueError as exc:
        print(f"[ERROR] Failed to build grasp object cfg: {exc}")
        object_cfg = None

    print("[INFO] Typed cfgs:")
    pprint({"table_cfg": table_cfg, "object_cfg": object_cfg})

    scene_cfg = UnigraspTransformerSceneCfg(
        table=table_cfg,
        grasp_object=object_cfg,
    )
    env_cfg = UnigraspTransformerGraspEnv(scene=scene_cfg)

    print("[INFO] Scene cfg summary:")
    pprint(
        {
            "num_envs": scene_cfg.num_envs,
            "env_spacing": scene_cfg.env_spacing,
            "robot_asset": scene_cfg.robot.spawn if hasattr(scene_cfg.robot, "spawn") else None,
            "table_cfg": scene_cfg.table,
            "object_cfg": scene_cfg.grasp_object,
        }
    )
    print("[INFO] Grasp env cfg summary:")
    pprint(
        {
            "device": env_cfg.device,
            "sim_dt": env_cfg.sim.dt,
            "sim_decimation": env_cfg.sim.decimation,
        }
    )
    print("[INFO] Weights cfg keys:", list(weights_cfg.keys()))
    print("[INFO] PPO cfg keys:", list(ppo_cfg.keys()))

    simulation_app.close()


if __name__ == "__main__":
    main()
