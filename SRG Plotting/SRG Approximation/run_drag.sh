#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$SCRIPT_DIR/SRG Approx Testing"
cd "$CPP_DIR"

cmake -S . -B build \
  --log-level=NOTICE \
  -Wno-deprecated \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.12

cmake --build build --target drag_sys

# Start C++ first; Python subscribes before sending its ready message.
setsid "$CPP_DIR/build/drag_sys" &
CPP_PID=$!
cd "$SCRIPT_DIR"
setsid .venv/bin/python3 Python/drag_listener.py "$@" &
PY_PID=$!

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
    # If the plotter exits, also stop the simulator.
    if ! kill -0 "$PY_PID" 2>/dev/null; then
        # No receiver remains for the readiness/completion handshake.
        kill -KILL "$CPP_PID" 2>/dev/null || true
    fi
    sleep 0.1 || true
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
