"""Minimal keyboard controller shared by simulator preview scripts."""

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


class KeyboardController:
    """Translate numerical key presses into Cartesian wrist motions."""

    _KEY_MAP = {
        carb.input.KeyboardInput.KEY_1: ("x", 1.0),
        carb.input.KeyboardInput.KEY_2: ("x", -1.0),
        carb.input.KeyboardInput.KEY_3: ("y", 1.0),
        carb.input.KeyboardInput.KEY_4: ("y", -1.0),
        carb.input.KeyboardInput.KEY_5: ("z", 1.0),
        carb.input.KeyboardInput.KEY_6: ("z", -1.0),
    }

    def __init__(self, step_size: float = 0.2) -> None:
        self._step = step_size
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
            print("[KeyboardController] Controls active: 1/2→X, 3/4→Y, 5/6→Z (CTRL+C to exit).")
        except Exception as exc:  # pragma: no cover - fallback for headless
            print(f"[WARN][KeyboardController] Input interface unavailable ({exc}). Keyboard control disabled.")
            self._subscription = None
            self._active = False

    def _on_keyboard_event(self, event, *_args, **_kwargs) -> bool:
        if event.input not in self._KEY_MAP:
            return False

        axis, direction = self._KEY_MAP[event.input]
        if event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            setattr(self._state, axis, self._step * direction)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            setattr(self._state, axis, 0.0)
        return True

    @property
    def translation(self) -> tuple[float, float, float]:
        """Return the active translation offset."""

        if not getattr(self, "_active", False):
            return 0.0, 0.0, 0.0
        return self._state.x, self._state.y, self._state.z

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
