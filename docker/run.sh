#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MAPTR_ROOT_DEFAULT="$(cd "${REPO_ROOT}/.." && pwd)/MapTR"
MAPTR_ROOT="${MAPTR_ROOT:-${MAPTR_ROOT_DEFAULT}}"

if [[ "${DOCKER_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo docker)
else
  DOCKER_CMD=(docker)
fi

DOCKER_RUN_ARGS=(
  run -it
  --gpus all
  --shm-size=16g
  --dns 8.8.8.8
  --dns 114.114.114.114
  -v "${REPO_ROOT}:/workspace/Apollo-Vision-Net"
)

if [[ -d "${MAPTR_ROOT}" ]]; then
  DOCKER_RUN_ARGS+=( -v "${MAPTR_ROOT}:/workspace/MapTR" )
fi

DOCKER_RUN_ARGS+=(
  -w /workspace/Apollo-Vision-Net
  apollo-vnet:cu113-v1
)

# Note: '--rm' is removed to allow 'docker commit' after exit.
if [[ "$#" -gt 0 ]]; then
  "${DOCKER_CMD[@]}" "${DOCKER_RUN_ARGS[@]}" "$@"
else
  "${DOCKER_CMD[@]}" "${DOCKER_RUN_ARGS[@]}" bash
fi

