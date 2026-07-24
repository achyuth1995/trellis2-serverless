#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cold-start entrypoint for the TRELLIS.2 RunPod serverless worker.
#
# Everything heavy (the `trellis2` conda env, repo, weights, HF cache) lives on
# the Network Volume RunPod mounts at /runpod-volume. We symlink it to
# /workspace and then run the SAME activation the working box uses.
# ---------------------------------------------------------------------------
set -euo pipefail

# RunPod serverless always mounts the network volume at /runpod-volume. Expose it
# as /workspace so the /workspace/... paths (conda env, weights, HF cache) match
# the working box. (On a GPU Pod the volume may already be /workspace -> keep it.)
if [ ! -e /workspace ]; then
    ln -s /runpod-volume /workspace
fi

export HF_HOME=/workspace/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/
# shellcheck disable=SC1091
source miniconda3/bin/activate
conda activate conda/envs/trellis2/

# `runpod` is a small pure-python package (NOT torch). It should be installed
# once into the volume env (see RUNPOD_DEPLOY.md); this is a safety net only.
python -c "import runpod" 2>/dev/null || pip install --no-cache-dir runpod

cd /workspace/Trellis_2/TRELLIS.2
exec python -u rp_handler.py
