#!/usr/bin/env bash
# Build the fraudlake wheel, upload it to the raw volume, deploy the bundle and run the job.
# Needs: the Databricks CLI logged in (databricks auth login), and uv (installed by earlier projects) or Python 3.10+.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf dist
if command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ]; then
  "$(command -v uv || echo "$HOME/.local/bin/uv")" build --wheel -o dist .
else
  python3 -m pip wheel --no-deps -w dist .
fi
WHL=$(ls dist/*.whl)
databricks fs mkdir dbfs:/Volumes/workspace/fraud/raw/libs
databricks fs cp --overwrite "$WHL" dbfs:/Volumes/workspace/fraud/raw/libs/
databricks bundle deploy
databricks bundle run fraud_job
