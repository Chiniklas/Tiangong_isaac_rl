# legged_lab package

This directory contains the installable Python package that extends Isaac Lab with
locomotion tasks carried over from the upstream TienKung-Lab template and the new
Inspire Hand dexterous manipulation environment.

## Script entry points
- `scripts/train.py`, `scripts/play.py`, `scripts/play_amp_animation.py`, and `scripts/sim2sim.py` remain the upstream locomotion utilities for the TienKung humanoid tasks (`walk`, `run`, and their sensor variants).
- `scripts/inspire_hand/train.py` and `scripts/inspire_hand/play.py` are the equivalents for the `inspirehand_grasp` task, with additional smoke tests under `scripts/inspire_hand/tests/`.

Refer to the top-level project `README.md` for installation steps and a rundown of the full workflow.


# TODO: implement a RL pipeline for grasp tasks for inspire hand