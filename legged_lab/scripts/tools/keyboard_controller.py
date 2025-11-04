"""Minimal keyboard controller shared by simulator preview scripts.

Controls
- Translation: 1/2 → ±X, 3/4 → ±Y, 5/6 → ±Z
- Rotation (about root): Q/W → ±Rx, E/R → ±Ry, T/Y → ±Rz

Notes
- Values represent per-step offsets (not accumulated). On key release the
  corresponding component returns to 0. Combine with your own scaling in the
  environment (e.g., `_palm_trans_action_scale`, `_palm_rot_action_scale`).

Switches
- Pass `enable_rotation=False` to disable all rotation features: rotation keys
  are ignored and `rotation` reports `(0, 0, 0)`.
"""

from __future__ import annotations

from dataclasses import dataclass
import weakref

import carb
import omni.appwindow


@dataclass
class _AxisState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0


class KeyboardController:
    """Translate numerical key presses into Cartesian wrist motions."""

    # Prefer carb enums for digits (robust across kits); use names for letters.
    try:
        _KEY_MAP_ENUM = {
            carb.input.KeyboardInput.KEY_1: ("x", 1.0),
            carb.input.KeyboardInput.KEY_2: ("x", -1.0),
            carb.input.KeyboardInput.KEY_3: ("y", 1.0),
            carb.input.KeyboardInput.KEY_4: ("y", -1.0),
            carb.input.KeyboardInput.KEY_5: ("z", 1.0),
            carb.input.KeyboardInput.KEY_6: ("z", -1.0),
        }
    except Exception:  # pragma: no cover
        _KEY_MAP_ENUM = {}

    # Fallback by input names for rotation keys (letters may be missing as enums on some builds).
    # event.input.name is expected to be like "Q", "W", etc. Add common variants.
    _KEY_NAME_MAP = {
        # translation
        "1": ("x", 1.0),
        "2": ("x", -1.0),
        "3": ("y", 1.0),
        "4": ("y", -1.0),
        "5": ("z", 1.0),
        "6": ("z", -1.0),
        "NUMPAD1": ("x", 1.0),
        "NUMPAD2": ("x", -1.0),
        "NUMPAD3": ("y", 1.0),
        "NUMPAD4": ("y", -1.0),
        "NUMPAD5": ("z", 1.0),
        "NUMPAD6": ("z", -1.0),
        # rotation (about root)
        "Q": ("rx", 1.0),
        "W": ("rx", -1.0),
        "E": ("ry", 1.0),
        "R": ("ry", -1.0),
        "T": ("rz", 1.0),
        "Y": ("rz", -1.0),
    }

    def __init__(self, step_size: float = 0.2, rot_step_size: float = 0.2, enable_rotation: bool = True) -> None:
        self._step = step_size
        self._rot_step = rot_step_size
        self._enable_rotation = bool(enable_rotation)
        self._state = _AxisState()
        self._input_iface = None
        self._keyboard = None
        try:
            self._input_iface = carb.input.acquire_input_interface()
            app_window = omni.appwindow.get_default_app_window()
            self._keyboard = app_window.get_keyboard() if app_window is not None else None
            if self._input_iface is None or self._keyboard is None:
                raise RuntimeError("Keyboard device unavailable")
            self._subscription = self._input_iface.subscribe_to_keyboard_events(
                self._keyboard,
                lambda evt, *args, obj=weakref.proxy(self): obj._on_keyboard_event(evt, *args),
            )
            self._active = True
            if self._enable_rotation:
                print(
                    "[KeyboardController] Controls: 1/2→±X, 3/4→±Y, 5/6→±Z; Q/W→±Rx, E/R→±Ry, T/Y→±Rz (CTRL+C to exit)."
                )
            else:
                print("[KeyboardController] Controls: 1/2→±X, 3/4→±Y, 5/6→±Z (rotation disabled).")
        except Exception as exc:  # pragma: no cover - fallback for headless
            print(f"[WARN][KeyboardController] Input interface unavailable ({exc}). Keyboard control disabled.")
            self._subscription = None
            self._active = False

    def _on_keyboard_event(self, event, *_args, **_kwargs) -> bool:
        axis = direction = None
        # First, try enum mapping (reliable for digits on most kits).
        if event.input in self._KEY_MAP_ENUM:
            axis, direction = self._KEY_MAP_ENUM[event.input]
        else:
            # Fallback to name-based mapping.
            key_name = getattr(event.input, "name", None)
            if key_name:
                key_name = str(key_name).upper()
                if key_name in self._KEY_NAME_MAP:
                    axis, direction = self._KEY_NAME_MAP[key_name]

        if axis is None:
            return False
        # Ignore rotation axes when disabled
        if not self._enable_rotation and axis in ("rx", "ry", "rz"):
            return True
        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            step = self._rot_step if axis in ("rx", "ry", "rz") else self._step
            setattr(self._state, axis, step * direction)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            setattr(self._state, axis, 0.0)
        return True

    @property
    def translation(self) -> tuple[float, float, float]:
        """Return the active translation offset."""

        if not getattr(self, "_active", False):
            return 0.0, 0.0, 0.0
        return self._state.x, self._state.y, self._state.z

    @property
    def rotation(self) -> tuple[float, float, float]:
        """Return the active rotation vector offsets (rx, ry, rz)."""

        if not getattr(self, "_active", False) or not self._enable_rotation:
            return 0.0, 0.0, 0.0
        return self._state.rx, self._state.ry, self._state.rz

    def shutdown(self) -> None:
        """Detach the keyboard subscription."""

        if getattr(self, "_subscription", None) is not None and self._input_iface is not None and self._keyboard is not None:
            if hasattr(self._input_iface, "unsubscribe_from_keyboard_events"):
                self._input_iface.unsubscribe_from_keyboard_events(self._keyboard, self._subscription)
            elif hasattr(self._input_iface, "unsubscribe_to_keyboard_events"):
                self._input_iface.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None
        self._keyboard = None

    def __del__(self) -> None:  # pragma: no cover
        self.shutdown()


__all__ = ["KeyboardController"]
