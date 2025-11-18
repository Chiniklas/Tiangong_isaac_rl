#!/usr/bin/env bash
# Convert GraspXL meshes to USDs and generate SDFs in one go.
# Usage (inside Isaac Lab kit shell):
#   ./isaaclab.sh --run bash legged_lab/scripts/tools/run_graspxl_asset_pipeline.sh [args]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

echo "[GraspXL] Converting meshes to USD..."
python "${SCRIPT_DIR}/convert_graspxl_dataset_to_usd.py" "$@"

echo "[GraspXL] Generating SDF volumes..."
python "${SCRIPT_DIR}/generate_sdf.py" "$@"

echo "[GraspXL] Pipeline complete."
