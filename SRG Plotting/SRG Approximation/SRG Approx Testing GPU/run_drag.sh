#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build \
  --log-level=NOTICE \
  -Wno-deprecated \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.12

cmake --build build

../.venv/bin/python3 drag_listener.py & 
PY_PID0=$!

setsid ./build/drag_sys & 
CPP_PID0=$!

cleanup() {
    kill "$CPP_PID0" "$PY_PID0" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

wait -n "$PY_PID0"

cleanup
wait || true