#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

export PYTHONDONTWRITEBYTECODE=1
export DEEPFIRM_OFFLINE=1

python scripts/verify_release.py --deny-network
python -m pytest -q -p no:cacheprovider \
  tests/test_offline_smoke.py \
  tests/test_artifacts.py \
  tests/test_frozen_results.py

echo "OFFLINE_SMOKE_STATUS=PASSED"
