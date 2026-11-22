#!/usr/bin/env bash
set -euo pipefail
web="$(cd "$(dirname "$0")/../web" && pwd)"
mkdir -p "$web/../work"
fixture="$(mktemp -d "$web/../work/web-e2e.XXXXXX")"
trap 'rm -rf -- "$fixture"' EXIT
export ART_DEMO_STATE_DIR="$fixture/private"
export ART_DEMO_PUBLIC="$fixture/public"
bash "$web/../config/demo.sh"
cd "$web"
npx --no-install playwright test "$@"
