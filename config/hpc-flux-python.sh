#!/bin/bash
# Compatibility overlay: retain the previously validated CUDA 12.4 dependencies.
# This is invoked INSIDE Apptainer by hpc-python.sh, never directly for inference
# on a login node. The official source is kept separate from site-packages.
set -euo pipefail
curator_root="$(cd "$(dirname "$0")/.." && pwd -P)"
source_root="$curator_root/work/flux3-bfc8889/src"
if [[ ! -f "$source_root/flux3/pipeline.py" ]]; then
    printf 'Missing pinned official FLUX source. See README compatibility setup.\n' >&2
    exit 2
fi
export PYTHONPATH="$source_root:$curator_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$curator_root/.venv-zimage/bin/python" "$@"
