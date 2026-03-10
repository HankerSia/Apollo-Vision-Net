#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: expected conda at ${HOME}/anaconda3; update docker/pack_env.sh if yours differs." >&2
  exit 1
fi

source "${HOME}/anaconda3/etc/profile.d/conda.sh"
conda activate apollo_vnet

mkdir -p docker

rm -f docker/apollo_vnet.tar.gz

# Packs the *current* conda env (exact versions) into a relocatable tarball.
# Note: conda-pack can't include editable packages. We ignore them and rely on
# mounting the source repo(s) + PYTHONPATH inside the container.
conda-pack -n apollo_vnet --ignore-editable-packages -o docker/apollo_vnet.tar.gz

echo "Wrote docker/apollo_vnet.tar.gz"
