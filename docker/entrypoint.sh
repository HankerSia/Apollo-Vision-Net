#!/usr/bin/env bash
set -e

# Make mounted source trees importable without requiring pip editable installs.
export PYTHONPATH="/workspace/Apollo-Vision-Net:${PYTHONPATH:-}"

if [[ -d /workspace/MapTR/mmdetection3d ]]; then
  export PYTHONPATH="/workspace/MapTR/mmdetection3d:${PYTHONPATH}"
fi

exec "$@"
