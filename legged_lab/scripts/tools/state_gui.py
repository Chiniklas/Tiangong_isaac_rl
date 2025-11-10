#!/usr/bin/env python3
"""Lightweight state GUI for UniGraspTransformer debugging.

Usage (from a test script):

    from legged_lab.scripts.tools.state_gui import launch_state_gui
    gui = launch_state_gui(title="UGTF Debug")
    ...
    # Inside your simulation loop each step:
    actor_obs, reward_buf, reset_buf, extras = env.step(actions)
    gui.update(
        step=step_i,
        obs=actor_obs,
        actions=actions,
        reward_total=reward_buf,
        reward_logs=extras.get("log", {}),
    )

If omni.ui is available (running inside Isaac Lab), a small GUI window shows the
current step, observation/action shapes and a breakdown of reward terms. If not,
it falls back to concise stdout prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math


def _to_cpu_list(x) -> list[float]:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().to("cpu")
            if x.ndim == 0:
                return [float(x.item())]
            return [float(v) for v in x.view(-1).tolist()]
    except Exception:
        pass
    # Numpy arrays or lists
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return [float(x.item())]
            return [float(v) for v in x.reshape(-1).tolist()]
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    try:
        return [float(x)]
    except Exception:
        return []


@dataclass
class _Snapshot:
    step: int = 0
    obs_shape: str = ""
    act_shape: str = ""
    reward_total: float = 0.0
    reward_terms: Dict[str, float] = None  # type: ignore[assignment]


class _ConsoleFallback:
    def __init__(self, title: str) -> None:
        self._title = title
        self._last_print = -1

    def update(self, snap: _Snapshot) -> None:
        # throttle prints to ~10 Hz if step increments fast
        if snap.step == self._last_print:
            return
        self._last_print = snap.step
        terms = " ".join(
            f"{k.split('/')[-1]}={v:+.3f}"
            for k, v in sorted((snap.reward_terms or {}).items())
            if k.startswith("reward/")
        )
        print(
            f"[{self._title}] step={snap.step} obs={snap.obs_shape} act={snap.act_shape} total={snap.reward_total:+.3f} {terms}"
        )


class _OmniGui:
    def __init__(self, title: str = "State GUI") -> None:
        import omni.ui as ui

        self._ui = ui
        self._win = ui.Window(title, width=420, height=260)
        with self._win.frame:
            with ui.VStack(spacing=4, height=0):
                self._l_step = ui.Label("Step: -")
                self._l_shapes = ui.Label("Obs: - | Act: -")
                self._l_total = ui.Label("Reward (total): -")
                ui.Spacer(height=4)
                ui.Label("Reward terms (env 0):", style={"font_size": 14})
                with ui.VStack(spacing=2, height=0):
                    self._l_init = ui.Label("init: -")
                    self._l_grasp = ui.Label("grasp: -")
                    self._l_act = ui.Label("action_penalty: -")
                ui.Spacer(height=4)
                self._l_hint = ui.Label("Update cadence follows simulation steps.")

    def update(self, snap: _Snapshot) -> None:
        ui = self._ui
        if self._win is None or self._win.frame is None:
            return
        self._l_step.text = f"Step: {snap.step}"
        self._l_shapes.text = f"Obs: {snap.obs_shape} | Act: {snap.act_shape}"
        self._l_total.text = f"Reward (total): {snap.reward_total:+.4f}"
        terms = snap.reward_terms or {}
        def _fmt(key: str) -> str:
            v = terms.get(key)
            if v is None:
                return "-"
            return f"{float(v):+.4f}"
        self._l_init.text = f"init: {_fmt('reward/init')}"
        self._l_grasp.text = f"grasp: {_fmt('reward/grasp')}"
        self._l_act.text = f"action_penalty: {_fmt('reward/action_penalty')}"


class StateGUI:
    """Unified wrapper providing a GUI if available, stdout fallback otherwise.

    Call `update(step=..., obs=..., actions=..., reward_total=..., reward_logs=...)`
    once per simulation step. Tensors are handled on device and reduced for env 0.
    """

    def __init__(self, title: str = "State GUI") -> None:
        try:
            # Test that omni.ui can be imported and a window created
            self._backend = _OmniGui(title)
        except Exception:
            self._backend = _ConsoleFallback(title)

    def update(
        self,
        *,
        step: int,
        obs: Any,
        actions: Any,
        reward_total: Any,
        reward_logs: Optional[Dict[str, Any]] = None,
    ) -> None:
        # shapes
        try:
            obs_shape = tuple(getattr(obs, "shape", ()))
        except Exception:
            obs_shape = ()
        try:
            act_shape = tuple(getattr(actions, "shape", ()))
        except Exception:
            act_shape = ()

        # reward scalars (env 0)
        total_list = _to_cpu_list(reward_total)
        total0 = total_list[0] if total_list else 0.0

        terms: Dict[str, float] = {}
        if isinstance(reward_logs, dict):
            for k in ("reward/init", "reward/grasp", "reward/action_penalty"):
                v = reward_logs.get(k)
                vals = _to_cpu_list(v)
                if vals:
                    terms[k] = vals[0]

        snap = _Snapshot(
            step=int(step),
            obs_shape=str(obs_shape),
            act_shape=str(act_shape),
            reward_total=float(total0),
            reward_terms=terms,
        )
        self._backend.update(snap)


def launch_state_gui(title: str = "State GUI") -> StateGUI:
    """Construct and return a StateGUI backend.

    This is cheap and safe to call from test scripts. If running outside Isaac Lab
    (no omni.ui), it will print lines to stdout instead of opening a window.
    """
    return StateGUI(title=title)


__all__ = ["launch_state_gui", "StateGUI"]

