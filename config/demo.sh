#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export ART_BACKEND=mock
export ART_CONFIG=config/demo.yaml
export ART_GPU_PYTHON="${ART_DEMO_PYTHON:-python3}"
export ART_STATE_DIR="${ART_DEMO_STATE_DIR:-work/demo/private}"
export ART_PUBLIC_DIR="${ART_DEMO_PUBLIC:-work/demo/public}"
export GMAIL_ENABLED=false GIT_PUSH_ENABLED=false HPC_ENABLED=false
py="$ART_GPU_PYTHON"
"$py" -m artcurator generate --count 8
"$py" -m artcurator preview-email --latest
"$py" -m artcurator send-email --latest --mock
"$py" -m artcurator mock-reply --latest --text 'APPROVE: 01, 02, 03, 04'
"$py" -m artcurator check-replies --mock
"$py" -m artcurator publish --approved-only --mock
