#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build \
  --log-level=NOTICE \
  -Wno-deprecated \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.12

cmake --build build

# Start both programs in separate sessions so Ctrl+C only reaches this script.
setsid ../.venv/bin/python3 drag_listener.py &
PY_PID=$!

setsid ./build/drag_sys &
CPP_PID=$!

phase="cpp"

handle_ctrl_c() {
    if [[ "$phase" == "cpp" ]]; then
        echo
        echo "Stopping C++..."
        kill "$CPP_PID" 2>/dev/null || true
    elif [[ "$phase" == "python" ]]; then
        echo
        echo "Stopping Python..."
        kill "$PY_PID" 2>/dev/null || true
    fi
}

cleanup() {
    kill "$CPP_PID" "$PY_PID" 2>/dev/null || true
}

trap handle_ctrl_c INT
trap cleanup TERM EXIT


# ── Phase 1 ───────────────────────────────────────────────
# Wait until:
#   - C++ finishes naturally, OR
#   - Ctrl+C causes handle_ctrl_c to kill it.

while kill -0 "$CPP_PID" 2>/dev/null; do
    wait "$CPP_PID" 2>/dev/null || true
done

# Reap C++ if necessary.
wait "$CPP_PID" 2>/dev/null || true


# ── Phase 2 ───────────────────────────────────────────────
phase="python"

# Wait until:
#   - Python finishes naturally, OR
#   - another Ctrl+C causes handle_ctrl_c to kill it.

while kill -0 "$PY_PID" 2>/dev/null; do
    wait "$PY_PID" 2>/dev/null || true
done

wait "$PY_PID" 2>/dev/null || true

phase="done"