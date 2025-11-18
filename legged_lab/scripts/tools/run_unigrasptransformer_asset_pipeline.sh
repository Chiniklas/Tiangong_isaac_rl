#!/usr/bin/env bash
# Helper script to build UniGraspTransformer USDs and dataset helpers in one go.
# Run inside the Isaac Lab kit shell so the converters are available:
#   ./isaaclab.sh --run bash legged_lab/scripts/tools/run_unigrasptransformer_asset_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." >/dev/null 2>&1 && pwd)"

echo "[UGTF] Converting selected meshes to USD..."
python "${SCRIPT_DIR}/convert_unigrasptransformer_dataset_to_usd.py" "$@"

echo "[UGTF] Generating FPS point clouds, PCA axes, and initial poses..."
python "${SCRIPT_DIR}/generate_fps_pca_init.py" "$@"

echo "[UGTF] Pipeline complete."
