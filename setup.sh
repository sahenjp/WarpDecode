#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"

BUILD=build-ui
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DWARPDECODE_BUILD_ENGINE=OFF >&2
cmake --build "$BUILD" -j"$(nproc)" >&2
exec "$BUILD/warpdecode-tui" "$@"
