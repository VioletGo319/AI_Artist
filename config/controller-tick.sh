#!/bin/sh
# Run on the private Mac only. Credentials and daily state remain local.
set -eu
CURATOR_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$CURATOR_ROOT"
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PYTHONPYCACHEPREFIX="$CURATOR_ROOT/work/native-controller/pycache"
export PYTHONUNBUFFERED=1
exec "$CURATOR_ROOT/work/control-arm64/bin/python" -m artcurator tick
