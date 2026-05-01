#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the gtests' install location
script_dir="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
repo_dir="$(realpath "${script_dir}/..")"
export RAPIDS_DATASET_ROOT_DIR="${RAPIDS_DATASET_ROOT_DIR:-${repo_dir}/datasets}"
export CUOPT_HOME="${CUOPT_HOME:-${repo_dir}}"

# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuopt/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="${repo_dir}/cpp/build/latest"

if [[ -d "${installed_test_location}" ]]; then
    cd "${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
    cd "${devcontainers_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

ctest --output-on-failure --no-tests=error "$@"
