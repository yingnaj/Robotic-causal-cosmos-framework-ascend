#!/usr/bin/env bash
# Source this file to activate the Cosmos3/RLinf Ascend Conda environment.

_cosmos_rlinf_env="${COSMOS_RLINF_CONDA_ENV:-/root/miniconda3/envs/cosmos3-rlinf-ascend}"
if [[ ! -d "${_cosmos_rlinf_env}/conda-meta" ]]; then
    printf 'Environment not found: %s\n' "${_cosmos_rlinf_env}" >&2
    return 1
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${_cosmos_rlinf_env}"
export COSMOS_FRAMEWORK_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset _cosmos_rlinf_env
