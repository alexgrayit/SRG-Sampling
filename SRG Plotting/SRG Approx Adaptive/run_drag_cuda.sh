#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$SCRIPT_DIR/SRG Approx Testing GPU"
simulator_args=()
plotter_args=()
show_plot=true
while (( $# )); do
    case "$1" in
        --mode)
            if (( $# < 2 )); then echo "--mode needs a value." >&2; exit 2; fi
            simulator_args+=("$1" "$2")
            plotter_args+=("$1" "$2")
            shift
            ;;
        --mode=*)
            simulator_args+=("$1")
            plotter_args+=("$1")
            ;;
        --no-video)
            plotter_args+=("$1")
            ;;
        --no-plot|--help|--profile-support)
            show_plot=false
            simulator_args+=("$1")
            ;;
        *)
            simulator_args+=("$1")
            ;;
    esac
    shift
done

echo "Building the CUDA simulator..."
cmake -S "$CPP_DIR" -B "$CPP_DIR/build" \
  --log-level=NOTICE \
  -Wno-deprecated \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-89}" \
  -DUSE_LSODA="${USE_LSODA:-OFF}" \
  -DSRG_LINEAR_DRAG="${SRG_LINEAR_DRAG:-OFF}" \
  -DSRG_ENABLE_FRICTION="${SRG_ENABLE_FRICTION:-OFF}"

cmake --build "$CPP_DIR/build" --target drag_sys_cuda
if ! "$show_plot"; then
    exec "$CPP_DIR/build/drag_sys_cuda" "${simulator_args[@]}"
fi

python_executable="${SRG_PYTHON:-$SCRIPT_DIR/.venv/bin/python3}"
if ! command -v "$python_executable" >/dev/null 2>&1; then
    echo "Install the Python dependencies in .venv or set SRG_PYTHON." >&2
    exit 2
fi
setsid "$CPP_DIR/build/drag_sys_cuda" "${simulator_args[@]}" &
CPP_PID=$!
cd "$SCRIPT_DIR"
echo "Starting the SRG plotter..."
SRG_SIM_PID="$CPP_PID" setsid "$python_executable" -u Python/drag_listener.py "${plotter_args[@]}" &
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

# CLI, LCM startup and abnormal exits may occur before the completion message.
cpp_status=0
wait "$CPP_PID" 2>/dev/null || cpp_status=$?
if (( cpp_status == 2 || cpp_status >= 128 )); then
    exit "$cpp_status"
fi


# ── Phase 2 ───────────────────────────────────────────────
phase="python"

# Wait until:
#   - Python finishes naturally, OR
#   - another Ctrl+C causes handle_ctrl_c to kill it.

while kill -0 "$PY_PID" 2>/dev/null; do
    wait "$PY_PID" 2>/dev/null || true
done

plotter_status=0
wait "$PY_PID" 2>/dev/null || plotter_status=$?

phase="done"
if (( cpp_status != 0 )); then exit "$cpp_status"; fi
exit "$plotter_status"
