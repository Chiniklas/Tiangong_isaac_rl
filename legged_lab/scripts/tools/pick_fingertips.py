#!/usr/bin/env python3
"""Keyboard-driven fingertip marker controller (uses legacy keyboard backend).

Spawns a single Inspire Hand (no table/object). Shows two spheres:
  - Green: fingertip link origin
  - Yellow: adjustable fingertip target (same link)

Keys (viewer focused):
  - 1/2: ±X, 3/4: ±Y, 5/6: ±Z  (hand-root local axes)
  - R: reset yellow to link origin
  - S: save offset to YAML (persists to hand.fingertip_local_offsets for selected finger)

Select finger at startup with --finger (1..5 or name: thumb,index,middle,ring,little).
Continuously prints the yellow marker coordinates in the fingertip link frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _ensure_isaaclab_on_path():
    import os, sys
    if "isaaclab" in sys.modules:
        return
    source_hint = os.environ.get("ISAACLAB_SOURCE")
    candidates = []
    if source_hint:
        candidates += [Path(source_hint), Path(source_hint) / "isaaclab"]
    home_root = Path.home() / "IsaacLab" / "source"
    candidates += [home_root, home_root / "isaaclab"]
    for p in candidates:
        if p.exists() and p.as_posix() not in sys.path:
            sys.path.append(p.as_posix())


_ensure_isaaclab_on_path()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", action="store_true", help="Run without rendering (not recommended).")
    ap.add_argument("--step", type=float, default=0.001, help="Step size per key press (meters)")
    ap.add_argument("--finger", type=str, default="thumb", help="Finger: 1..5 or {thumb,index,middle,ring,little}")
    ap.add_argument("--yaml", type=Path, default=Path("legged_lab/envs/unigrasptransformer/config.yaml"), help="Path to config.yaml")
    return ap.parse_args()


def main():
    args = parse_args()

    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        raise SystemExit("Run inside Isaac Lab kit shell (./isaaclab.sh --run)") from exc

    app = AppLauncher(headless=args.headless)
    simulation_app = app.app

    import numpy as np
    import yaml
    import carb
    import carb.input as cinput
    import omni.appwindow
    import omni.usd
    from pxr import UsdGeom, Gf

    from legged_lab.envs.unigrasptransformer.unigrasptransformer_env import UniGraspTransformerEnv
    from legged_lab.envs.unigrasptransformer.unigrasptransformer_cfg import (
        UniGraspTransformerEnvCfg,
        UniGraspTransformerGraspSceneCfg,
    )
    from legged_lab.envs.unigrasptransformer.spawn_cfg import UniGraspTransformerSpawnCfg

    # Hand-only scene
    spawn_cfg = UniGraspTransformerSpawnCfg()
    spawn_cfg.table.enable = False
    spawn_cfg.grasp_object.enable = False
    scene_cfg = UniGraspTransformerGraspSceneCfg(spawn=spawn_cfg, num_envs=1)
    try:
        scene_cfg.max_episode_length_s = 1e12
    except Exception:
        pass
    env_cfg = UniGraspTransformerEnvCfg(scene=scene_cfg)
    env = UniGraspTransformerEnv(env_cfg, headless=args.headless)

    finger_names = ["thumb", "index", "middle", "ring", "little"]
    sarg = str(args.finger).strip().lower()
    if sarg.isdigit():
        sel = max(1, min(5, int(sarg))) - 1
    else:
        try:
            sel = finger_names.index(sarg)
        except ValueError:
            sel = 0

    print("[INFO] Interactive fingertip tuner loaded.")
    print("  - Move (hand-local): 1=+X, 2=-X, 3=+Y, 4=-Y, 5=+Z, 6=-Z")
    print("  - Reset: 9,  Save & Exit: 0")
    print(f"[INFO] Selected finger: {sel+1} ({finger_names[sel]})")

    offsets = getattr(env.cfg.scene.spawn.hand, "fingertip_local_offsets", []) or []
    if len(offsets) != 5:
        offsets = [(0.0, 0.0, 0.0) for _ in range(5)]

    # Helpers
    def _tip_local() -> np.ndarray:
        tip_pos_w = env.hand.data.body_pos_w[:, env._tip_body_ids, :][0].detach().cpu().numpy()
        hand_pos = env.hand.data.root_pos_w[0].detach().cpu().numpy()
        q = env.hand.data.root_quat_w[0].detach().cpu().numpy()  # xyzw
        ox, oy, oz, ow = q
        R = np.array([
            [1 - 2 * (oy * oy + oz * oz), 2 * (ox * oy - oz * ow), 2 * (ox * oz + oy * ow)],
            [2 * (ox * oy + oz * ow), 1 - 2 * (ox * ox + oz * oz), 2 * (oy * oz - ox * ow)],
            [2 * (ox * oz - oy * ow), 2 * (oy * oz + ox * ow), 1 - 2 * (ox * ox + oy * oy)],
        ], dtype=np.float32)
        return (tip_pos_w - hand_pos[None, :]) @ R

    def _yellow_in_link(marker_local: np.ndarray) -> tuple[float, float, float]:
        hpos = env.hand.data.root_pos_w[0].detach().cpu().numpy()
        qh = env.hand.data.root_quat_w[0].detach().cpu().numpy()  # xyzw
        ox, oy, oz, ow = qh
        R_h = np.array([
            [1 - 2 * (oy * oy + oz * oz), 2 * (ox * oy - oz * ow), 2 * (ox * oz + oy * ow)],
            [2 * (ox * oy + oz * ow), 1 - 2 * (ox * ox + oz * oz), 2 * (oy * oz - ox * ow)],
            [2 * (ox * oz - oy * ow), 2 * (oy * oz + ox * ow), 1 - 2 * (ox * ox + oy * oy)],
        ], dtype=np.float32)
        m_world = hpos + (R_h @ marker_local)
        tip_pos_w = env.hand.data.body_pos_w[0, env._tip_body_ids[sel], :].detach().cpu().numpy()
        tip_q = env.hand.data.body_quat_w[0, env._tip_body_ids[sel], :].detach().cpu().numpy()
        lx, ly, lz, lw = tip_q
        R_l = np.array([
            [1 - 2 * (ly * ly + lz * lz), 2 * (lx * ly - lz * lw), 2 * (lx * lz + ly * lw)],
            [2 * (lx * ly + lz * lw), 1 - 2 * (lx * lx + lz * lz), 2 * (ly * lz - lx * lw)],
            [2 * (lx * lz - ly * lw), 2 * (ly * lz + lx * lw), 1 - 2 * (lx * lx + ly * ly)],
        ], dtype=np.float32)
        p_link = R_l.T @ (m_world - tip_pos_w)
        return float(p_link[0]), float(p_link[1]), float(p_link[2])

    # Build spheres
    stage = omni.usd.get_context().get_stage()
    tip_local = _tip_local()
    debug_root = "/World/envs/env_0/Robot/Debug"
    UsdGeom.Xform.Define(stage, debug_root)
    radius = float(getattr(env.cfg.scene.spawn.hand, "fingertip_marker_size", 0.001))
    green_path = f"{debug_root}/TipLinkOrigin"
    green = UsdGeom.Sphere.Define(stage, green_path)
    green.CreateRadiusAttr(radius * 0.9)
    green.GetDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.95, 0.3)])
    UsdGeom.Xformable(green).AddTranslateOp().Set(Gf.Vec3f(*map(float, tip_local[sel])))

    marker_local = tip_local[sel].copy()
    ox, oy, oz = offsets[sel]
    marker_local += np.array([float(ox), float(oy), float(oz)], dtype=np.float32)
    marker_path = f"{debug_root}/TipMarker"
    sph = UsdGeom.Sphere.Define(stage, marker_path)
    sph.CreateRadiusAttr(radius * 0.6)
    sph.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.85, 0.2)])
    UsdGeom.Xformable(sph).AddTranslateOp().Set(Gf.Vec3f(*map(float, marker_local)))

    def _update_marker():
        prim = UsdGeom.Sphere.Get(stage, marker_path)
        if not prim:
            return
        xfa = UsdGeom.Xformable(prim)
        ops = xfa.GetOrderedXformOps()
        pos = Gf.Vec3f(*map(float, marker_local))
        (ops[0].Set(pos) if ops else xfa.AddTranslateOp().Set(pos))

    # Keyboard subscription (legacy backend)
    input_iface = cinput.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()
    kb = app_window.get_keyboard() if app_window else None
    state = {"dx": 0.0, "dy": 0.0, "dz": 0.0, "save": False, "reset": False}

    KEY_MAP = {
        cinput.KeyboardInput.KEY_1: ("dx", +1.0),
        cinput.KeyboardInput.KEY_2: ("dx", -1.0),
        cinput.KeyboardInput.KEY_3: ("dy", +1.0),
        cinput.KeyboardInput.KEY_4: ("dy", -1.0),
        cinput.KeyboardInput.KEY_5: ("dz", +1.0),
        cinput.KeyboardInput.KEY_6: ("dz", -1.0),
        cinput.KeyboardInput.KEY_0: ("save", 1.0),
        cinput.KeyboardInput.KEY_9: ("reset", 1.0),
    }

    def _on_key(evt, *_args):
        if evt.input not in KEY_MAP:
            return False
        name, sign = KEY_MAP[evt.input]
        if name in ("dx", "dy", "dz"):
            if evt.type in (cinput.KeyboardEventType.KEY_PRESS, cinput.KeyboardEventType.KEY_REPEAT):
                state[name] = args.step * sign
            elif evt.type == cinput.KeyboardEventType.KEY_RELEASE:
                state[name] = 0.0
        else:
            if evt.type == cinput.KeyboardEventType.KEY_PRESS:
                state[name] = True
        return True

    sub = None
    if input_iface and kb:
        sub = input_iface.subscribe_to_keyboard_events(kb, _on_key)
        print("[INFO] Keyboard ready: 1/2 ±X, 3/4 ±Y, 5/6 ±Z, 9 reset, 0 save")
    else:
        print("[WARN] Keyboard unavailable. Movement disabled.")

    # Run
    try:
        import torch
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        running = True
        while running and simulation_app.is_running():
            env.step(actions)

            # apply movement
            if state["dx"] or state["dy"] or state["dz"]:
                marker_local[0] += float(state["dx"])  # x
                marker_local[1] += float(state["dy"])  # y
                marker_local[2] += float(state["dz"])  # z
                _update_marker()
                print(f"[INFO] Yellow (hand frame): [{marker_local[0]:.4f}, {marker_local[1]:.4f}, {marker_local[2]:.4f}]")

            if state["reset"]:
                tip_local = _tip_local()
                marker_local[:] = tip_local[sel]
                _update_marker()
                print(f"[INFO] Yellow (hand frame): [{marker_local[0]:.4f}, {marker_local[1]:.4f}, {marker_local[2]:.4f}]")
                state["reset"] = False

            if state["save"]:
                # save offset and exit
                cfg_path = args.yaml.expanduser()
                # Make relative path resolve from repo root (three parents up from this file)
                if not cfg_path.is_absolute():
                    cfg_path = (Path(__file__).resolve().parents[3] / cfg_path).resolve()
                try:
                    data = yaml.safe_load(cfg_path.read_text()) or {}
                except Exception:
                    data = {}
                root = data.get("unigrasptransformer", data)
                hand = root.setdefault("hand", {})
                tip_local = _tip_local()
                delta = marker_local - tip_local[sel]
                offs = list(offsets)
                offs[sel] = (float(delta[0]), float(delta[1]), float(delta[2]))
                hand["fingertip_local_offsets"] = [[float(a), float(b), float(c)] for (a, b, c) in offs]
                if "unigrasptransformer" in data:
                    data["unigrasptransformer"]["hand"] = hand
                else:
                    data["hand"] = hand
                # Ensure parent dir exists and write
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))
                print(f"[INFO] Saved offsets to {cfg_path}")
                running = False

    except KeyboardInterrupt:
        pass
    finally:
        if sub and input_iface and kb:
            # Different carb builds expose different unsubscribe names
            if hasattr(input_iface, "unsubscribe_from_keyboard_events"):
                input_iface.unsubscribe_from_keyboard_events(kb, sub)
            else:
                input_iface.unsubscribe_to_keyboard_events(kb, sub)
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
