# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Keyboard controller for simple in-sim actions.

Usage
- Construct with an initialized `BaseEnv`: `kb = Keyboard(env)`.
- Default binding: press `R` to mark all envs for reset (sets `episode_length_buf` large so the env's reset path triggers).
- Add custom keys: `keyboard.add_callback("F", lambda env: print("F pressed"))`.
  - Callback signature should be `func(env) -> None` (the environment is passed in). If your function takes no arguments, it will be called without parameters.

Notes
- This utility depends on Omniverse input APIs (carb/omni). It is a lightweight alternative to the preview‑script controller in `legged_lab/scripts/tools/keyboard_controller.py` (which maps 1–6 to XYZ translation for demos).
"""

import weakref
from collections.abc import Callable

import carb
import omni
import torch
from isaaclab.devices.device_base import DeviceBase

from legged_lab.envs.base.base_env import BaseEnv


class Keyboard(DeviceBase):
    def __init__(self, env: BaseEnv):
        """Initialize the keyboard layer."""
        self.env = env
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for ManagerBasedRLEnv: {self.__class__.__name__}\n"
        return msg

    """
    Operations
    """

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        """Register a callback for a given key name.

        - `key`: single character or Omniverse key name (e.g. "R", "F").
        - `func`: callable invoked on key press. Preferred signature: `(env) -> None`.
        """
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        # Normalize to upper-case Omniverse names for consistency with `event.input.name`.
        self._additional_callbacks[key.upper()] = func

    def advance(self):
        pass

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = getattr(event.input, "name", None)
            if key_name is None:
                return True
            # default mappings
            if key_name in self._INPUT_KEY_MAPPING:
                if key_name == "R":
                    self.env.episode_length_buf = torch.ones_like(self.env.episode_length_buf) * 1e6
            # user callbacks
            cb = self._additional_callbacks.get(key_name)
            if cb is not None:
                try:
                    cb(self.env)
                except TypeError:
                    cb()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # forward command
            "R": "reset envs",
        }
