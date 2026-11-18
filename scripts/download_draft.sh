#!/bin/bash
# Fetch the DFlash2 draft used with this model (Mia-AiLab 5.0bpw EXL3, ~1.4 GB) at a pinned revision.
# usage: bash download_draft.sh [models_dir=/models]
set -euo pipefail
M="${1:-/models}"
hf download Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw --revision 4f0436269bca761b071f05319e8e04a87cc633f9 \
  --local-dir "$M/mia-dflash2-50"
