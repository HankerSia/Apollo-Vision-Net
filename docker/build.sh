#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f docker/apollo_vnet.tar.gz ]]; then
  echo "Missing docker/apollo_vnet.tar.gz. Run: docker/pack_env.sh" >&2
  exit 1
fi

if [[ "${DOCKER_SUDO:-0}" == "1" ]]; then
  DOCKER_CMD=(sudo docker)
else
  DOCKER_CMD=(docker)
fi

BUILD_ARGS=()
if [[ -n "${BASE_IMAGE:-}" ]]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi

"${DOCKER_CMD[@]}" build "${BUILD_ARGS[@]}" -f docker/Dockerfile.apollo_vnet -t apollo-vnet:cu113 .
