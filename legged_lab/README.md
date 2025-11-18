# legged_lab package

This directory contains the installable Python package that extends Isaac Lab with
locomotion tasks carried over from the upstream TienKung-Lab template and the new
Inspire Hand dexterous manipulation environment.

## Script entry points
- `scripts/train.py`, `scripts/play.py`, `scripts/play_amp_animation.py`, and `scripts/sim2sim.py` remain the upstream locomotion utilities for the TienKung humanoid tasks (`walk`, `run`, and their sensor variants).
- `scripts/inspire_hand/train.py` and `scripts/inspire_hand/play.py` are the equivalents for the `inspirehand_grasp` task, with additional smoke tests under `scripts/inspire_hand/tests/`.
- UniGraspTransformer can be trained and replayed via the shared RSL-RL entrypoints. After registering `unigrasptransformer` in `task_registry`, launch training with:
  ```
  python legged_lab/scripts/train.py \
    --task unigrasptransformer \
    --num_envs 64 \
    --headless True \
    --seed 42
  ```
  and replay a checkpoint (e.g., `logs/unigrasptransformer_grasp/<run>/model_0.pt`) with:
  ```
  python legged_lab/scripts/play.py \
    --task unigrasptransformer \
    --load_run 2025-11-18_15-01-22 \
    --load_checkpoint model_0.pt \
    --num_envs 1 \
    --headless False
  ```

Refer to the top-level project `README.md` for installation steps and a rundown of the full workflow.


# TODO: implement a RL pipeline for grasp tasks for inspire hand

## Inspire Hand dataset helpers

- A standalone loader lives in `legged_lab/assets/inspirehand/object_library.py`. It scans the external `dataset/mixed_train` folder (copied from the GraspXL release) and exposes lightweight `GraspObjectInfo` entries for each of the 58 PartNet objects.
- Existing environments are untouched; the library can be imported from `legged_lab.assets.inspirehand` when you are ready to wire the dataset into the Inspire Hand RL pipeline.
- Optional: convert the meshes to single-rigid USDs for faster previewing via `python legged_lab/scripts/inspire_hand/tools/convert_dataset_to_usd.py`. Generated files are stored under `dataset/grasp_usd/<object>/` and picked up automatically by the loader.
