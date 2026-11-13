#!/usr/bin/env bash
# Put this beside run_drag_cuda.sh in the SRG Approximation directory.
# Requires NVIDIA Nsight Compute (ncu) and the project's existing Python venv.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$SCRIPT_DIR/SRG Approx Testing GPU"
PYTHON="$SCRIPT_DIR/.venv/bin/python3"
NCU="${NCU:-ncu}"
NCU_SET="${NCU_SET:-full}"
NCU_SKIP="${NCU_SKIP:-0}"
NCU_COUNT="${NCU_COUNT:-1}"
NCU_REPLAY="${NCU_REPLAY:-kernel}"
PROFILE_ONLY_CHECK="${PROFILE_ONLY_CHECK:-0}"
PROFILE_BATCH_SIZE="${PROFILE_BATCH_SIZE:-10240}"

# Reject malformed environment values before invoking Nsight. Canonicalise
# decimal integers so leading zeroes cannot be interpreted as another base.
normalise_launch_number() {
    local name="$1" minimum="$2" value
    value="${!name}"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        printf 'Invalid %s=%q: use a decimal integer (minimum %s).\n' \
            "$name" "$value" "$minimum" >&2
        exit 2
    fi
    value="${value#"${value%%[!0]*}"}"
    value="${value:-0}"
    if [[ ${#value} -gt 10 || ( ${#value} -eq 10 && "$value" > 2147483647 ) ]]; then
        printf '%s is too large: use a value no greater than 2147483647.\n' "$name" >&2
        exit 2
    fi
    if [[ "$minimum" == 1 && "$value" == 0 ]]; then
        printf '%s must be at least 1.\n' "$name" >&2
        exit 2
    fi
    printf -v "$name" '%s' "$value"
}
normalise_launch_number NCU_SKIP 0
normalise_launch_number NCU_COUNT 1
normalise_launch_number PROFILE_BATCH_SIZE 1
if (( PROFILE_BATCH_SIZE > 81920 )); then
    echo "PROFILE_BATCH_SIZE must not exceed the normal batch size of 81920." >&2
    exit 2
fi
if [[ "$NCU_REPLAY" != kernel && "$NCU_REPLAY" != application ]]; then
    echo "NCU_REPLAY must be kernel or application." >&2
    exit 2
fi
if [[ "$PROFILE_ONLY_CHECK" != 0 && "$PROFILE_ONLY_CHECK" != 1 ]]; then
    echo "PROFILE_ONLY_CHECK must be 0 or 1." >&2
    exit 2
fi
export SRG_PROFILE_BATCHES="$((NCU_SKIP + NCU_COUNT))"
export SRG_PROFILE_BATCH_SIZE="$PROFILE_BATCH_SIZE"
if (( SRG_PROFILE_BATCHES > 2147483647 )); then
    echo "NCU_SKIP + NCU_COUNT must not exceed 2147483647." >&2
    exit 2
fi

if [[ ! -f "$CPP_DIR/CMakeLists.txt" ]]; then
    echo "Place this script beside run_drag_cuda.sh in SRG Approximation." >&2
    exit 2
fi
if [[ "$PROFILE_ONLY_CHECK" != 1 ]] && ! command -v "$NCU" >/dev/null 2>&1; then
    echo "Nsight Compute CLI (ncu) was not found. Set NCU to its full path." >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Project Python was not found: $PYTHON" >&2
    exit 2
fi
PROFILE_DIR="$(mktemp -d "$SCRIPT_DIR/profiles_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
REPORT="$PROFILE_DIR/srg_cuda.ncu-rep"
export PYTHONPATH="$SCRIPT_DIR/Python${PYTHONPATH:+:$PYTHONPATH}"

# A partial .ncu-rep can exist after a profiling failure. Keep the complete
# console output so the underlying CUDA/profiler error is not lost.
exec > >(tee "$PROFILE_DIR/run.log") 2>&1
printf 'Capture directory: %s\n' "$PROFILE_DIR"
printf 'Requested metric set: %s\n' "$NCU_SET"
printf 'Profiling batch size: %s simulations (normal runs use 81920).\n' "$PROFILE_BATCH_SIZE"

# Check readiness-message imports before starting a profiler that would wait.
"$PYTHON" -c 'import lcm; from lcm_messages.simulation_messages import startSim'

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
    cmake -S "$CPP_DIR" -B "$CPP_DIR/build" \
        --log-level=NOTICE -Wno-deprecated \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-89}" \
        2>&1 | tee "$PROFILE_DIR/configure.log"
    cmake --build "$CPP_DIR/build" --target drag_sys_cuda \
        2>&1 | tee "$PROFILE_DIR/build.log"
fi

if [[ ! -x "$CPP_DIR/build/drag_sys_cuda" ]]; then
    echo "CUDA executable was not found: $CPP_DIR/build/drag_sys_cuda" >&2
    exit 2
fi

if command -v "$NCU" >/dev/null 2>&1; then
    "$NCU" --version > "$PROFILE_DIR/ncu_version.txt"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit \
        --format=csv > "$PROFILE_DIR/gpu_info.csv" || true
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version > "$PROFILE_DIR/nvcc_version.txt"
fi

# Keep the exact configuration/source with the profile for reproducibility.
cp "$CPP_DIR/src/drag_sys_cuda.cu" "$PROFILE_DIR/drag_sys_cuda.cu"
cp "$CPP_DIR/CMakeLists.txt" "$PROFILE_DIR/CMakeLists.txt"
if [[ -f "$CPP_DIR/include/integrators/LSODA/lsoda_cuda_bridge.cuh" ]]; then
    cp "$CPP_DIR/include/integrators/LSODA/lsoda_cuda_bridge.cuh" "$PROFILE_DIR/lsoda_cuda_bridge.cuh"
fi

PROFILE_PID=""
READY_PID=""
cleanup() {
    if [[ -n "$READY_PID" ]]; then
        kill "$READY_PID" 2>/dev/null || true
        wait "$READY_PID" 2>/dev/null || true
    fi
    if [[ -n "$PROFILE_PID" ]]; then
        # setsid gives the profiler and target their own process group.
        kill -- "-$PROFILE_PID" 2>/dev/null || true
        wait "$PROFILE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Profiling %s matching simulate_batch launch(es), skipping %s.\n' "$NCU_COUNT" "$NCU_SKIP"
printf 'Reports and source configuration: %s\n' "$PROFILE_DIR"
printf 'Nsight Compute may replay the batch many times to collect the full metric set.\n'

# The target stops itself after SRG_PROFILE_BATCHES and checks its results.
ncu_command=(
    "$NCU"
    --set "$NCU_SET"
    --kernel-name-base function
    --kernel-name simulate_batch
    --launch-count "$NCU_COUNT"
    --kill no
    --check-exit-code yes
    --replay-mode "$NCU_REPLAY"
    --import-source yes
    --export "$PROFILE_DIR/srg_cuda"
)
# Zero is Nsight's default. Omit the option entirely for the first batch.
if [[ "$NCU_SKIP" != 0 ]]; then
    ncu_command+=(--launch-skip "$NCU_SKIP")
fi
ncu_command+=("$CPP_DIR/build/drag_sys_cuda")
{
    printf 'Profiler command:'
    printf ' %q' "${ncu_command[@]}"
    printf '\n'
} > "$PROFILE_DIR/profiler_command.txt"
cat "$PROFILE_DIR/profiler_command.txt"
run_with_start_message() {
    local log_path="$1"
    shift
    local target_output tee_pid status=0
    exec {target_output}> >(tee "$log_path")
    tee_pid=$!
    setsid "$@" >&"$target_output" 2>&1 &
    PROFILE_PID=$!

# The simulator waits for startSim. Send throughout the bounded job so every
# process started by application replay receives a startup message as well.
"$PYTHON" - <<'PY' &
import lcm
from lcm_messages.simulation_messages import startSim

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0&recv_buf_size=8388608")
message = startSim()
message.sim_name = "drag_sim"
# Keep sending during the job: application replay can start a fresh simulator.
while True:
    lc.publish("startSim", message.encode())
    lc.handle_timeout(200)
PY
READY_PID=$!
    wait "$PROFILE_PID" || status=$?
    PROFILE_PID=""
    kill "$READY_PID" 2>/dev/null || true
    wait "$READY_PID" 2>/dev/null || true
    READY_PID=""
    exec {target_output}>&-
    wait "$tee_pid" || true
    return "$status"
}

# Refuse a stale executable instead of allowing an unbounded full sweep.
if ! timeout --kill-after=2s 5s "$CPP_DIR/build/drag_sys_cuda" --profile-support \
        > "$PROFILE_DIR/profile_support.txt" 2>&1; then
    echo "Rebuild drag_sys_cuda from this diagnostic update before running this script." >&2
    exit 2
fi
read -r profile_support < "$PROFILE_DIR/profile_support.txt"
if [[ "$profile_support" != SRG_PROFILE_BOUNDED_V2 ]]; then
    echo "This executable does not support profiling batch-size control. Rebuild it." >&2
    exit 2
fi

printf '\nRunning %s batch(es) without Nsight to check CUDA and solver results.\n' "$SRG_PROFILE_BATCHES"
preflight_status=0
run_with_start_message "$PROFILE_DIR/preflight.log" "$CPP_DIR/build/drag_sys_cuda" \
    || preflight_status=$?
printf '%s\n' "$preflight_status" > "$PROFILE_DIR/preflight_exit_status.txt"
if [[ "$preflight_status" != 0 ]]; then
    printf 'The unprofiled run failed (status %s). Share this folder:\n%s\n' \
        "$preflight_status" "$PROFILE_DIR" >&2
    exit 1
fi
preflight_validation_status=0
"$PYTHON" - "$PROFILE_DIR/preflight.log" "$SRG_PROFILE_BATCHES" "$PROFILE_BATCH_SIZE" <<'PY' \
    > "$PROFILE_DIR/preflight_validation.txt" 2>&1 || preflight_validation_status=$?
import math
import re
import sys
text = open(sys.argv[1]).read()
rows = re.findall(r"PROFILE_BATCH_OK batch=(\d+) simulations=(\d+) kernel_event_seconds=(\S+)", text)
requested = int(sys.argv[2])
batch_size = int(sys.argv[3])
if f"PROFILE_BATCH_SIZE simulations={batch_size} normal=81920" not in text:
    raise SystemExit("The simulator did not confirm the requested profiling batch size.")
if len(rows) != requested or "PROFILE_COMPLETE" not in text:
    raise SystemExit(f"Expected {requested} healthy batches and normal completion; found {len(rows)}.")
for index, (batch, count, seconds) in enumerate(rows, 1):
    if int(batch) != index or not 0 < int(count) <= batch_size or not math.isfinite(float(seconds)) or float(seconds) <= 0:
        raise SystemExit("Invalid unprofiled batch count or event duration.")
    print(f"Unprofiled batch {batch}: {count} simulations, GPU event duration {seconds} s")
print("PASS: unprofiled CUDA execution, solver results, and timing checks passed.")
PY
cat "$PROFILE_DIR/preflight_validation.txt"
if [[ "$preflight_validation_status" != 0 ]]; then
    printf 'The unprofiled run did not confirm valid completed batches. Share:\n%s\n' "$PROFILE_DIR" >&2
    exit 1
fi
if [[ "$PROFILE_ONLY_CHECK" == 1 ]]; then
    printf '\nGPU execution check completed. Share this folder:\n%s\n' "$PROFILE_DIR"
    exit 0
fi

printf '\nStarting Nsight capture with %s replay. CUDA-event times in this stage include profiler overhead.\n' "$NCU_REPLAY"
profile_status=0
run_with_start_message "$PROFILE_DIR/capture.log" "${ncu_command[@]}" || profile_status=$?
printf '%s\n' "$profile_status" > "$PROFILE_DIR/profiler_exit_status.txt"
printf 'Nsight Compute exit status: %s\n' "$profile_status"

if [[ ! -s "$REPORT" ]]; then
    if [[ "$profile_status" != 0 ]] && grep -Eq 'code=702|cudaErrorLaunchTimeout' "$PROFILE_DIR/capture.log"; then
        echo "The profiled kernel hit the GPU execution timeout. Retry with a smaller PROFILE_BATCH_SIZE (for example 1280)." >&2
    fi
    echo "No profiling report was produced. Share this folder, including run.log:" >&2
    printf '%s\n' "$PROFILE_DIR" >&2
    exit 1
fi

"$NCU" --import "$REPORT" --page details --print-details all \
    > "$PROFILE_DIR/details.txt"
"$NCU" --import "$REPORT" --page raw --csv \
    > "$PROFILE_DIR/metrics.csv"
"$NCU" --import "$REPORT" --page session \
    > "$PROFILE_DIR/session.txt"

# Having a report file does not prove that a kernel was successfully measured.
# Validate the raw CSV before announcing success. Retain all diagnostics on
# failure, including when the report contains only device attributes/zeros.
validation_status=0
"$PYTHON" - "$PROFILE_DIR/metrics.csv" "$NCU_COUNT" "$PROFILE_DIR/capture.log" "$SRG_PROFILE_BATCHES" "$PROFILE_BATCH_SIZE" <<'PY' \
    > "$PROFILE_DIR/validation.txt" 2>&1 || validation_status=$?
import csv
import math
import re
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))

# Nsight's CSV contains a units row and scales long durations to seconds.
# Keep its actual unit instead of assuming every duration is in nanoseconds.
units = next((row for row in rows if not (row.get("ID") or "").isdigit()), {})
duration_unit = units.get("gpu__time_duration.sum") or "(unit not recorded)"
rows = [row for row in rows if (row.get("ID") or "").isdigit()]
errors = []
requested = int(sys.argv[2])
with open(sys.argv[3]) as stream:
    capture_log = stream.read()
completed = re.findall(r"PROFILE_COMPLETE batches=(\d+) simulations=(\d+)", capture_log)
batch_size = int(sys.argv[5])
configured_sizes = re.findall(r"PROFILE_BATCH_SIZE simulations=(\d+) normal=81920", capture_log)
if not configured_sizes or any(int(size) != batch_size for size in configured_sizes):
    errors.append("The profiled application did not confirm the requested profiling batch size.")
if not completed or any(int(batches) != int(sys.argv[4]) or not 0 < int(count) <= int(batches) * batch_size for batches, count in completed):
    errors.append("The profiled application did not confirm normal completion of the requested batches.")
if len(rows) != requested:
    errors.append(f"Expected {requested} kernel result(s), found {len(rows)}.")

def number(value):
    try:
        return float(value.replace(",", ""))
    except (AttributeError, ValueError):
        return math.nan

for row in rows:
    name = row.get("Kernel Name", "unknown kernel")
    duration = number(row.get("gpu__time_duration.sum"))
    print(f"{name}: duration = {duration} {duration_unit}")
    if not math.isfinite(duration) or duration <= 0:
        errors.append("Kernel duration is missing, zero or non-finite.")
    for label in ("Block Size", "Grid Size"):
        dimensions = re.findall(r"\d+", row.get(label, ""))
        if not dimensions or any(int(value) <= 0 for value in dimensions):
            errors.append(f"Invalid {label}: {row.get(label)!r}.")
    if "smsp__inst_executed.sum" in row:
        instructions = number(row["smsp__inst_executed.sum"])
        print(f"Executed instructions = {instructions}")
        if not math.isfinite(instructions) or instructions <= 0:
            errors.append("Executed instruction count is zero or non-finite.")

if errors:
    print("INVALID CAPTURE: no reliable performance diagnosis can be made.")
    for error in errors:
        print("- " + error)
    sys.exit(1)
print("PASS: requested kernel measurements have positive duration and launch dimensions.")
PY
cat "$PROFILE_DIR/validation.txt"

if [[ "$validation_status" != 0 ]]; then
    printf '\nProfiling produced an invalid or incomplete capture.\n' >&2
    printf 'The separate unprofiled run passed; its timing is in preflight.log.\n' >&2
    printf 'Share the complete folder, especially run.log and validation.txt:\n%s\n' "$PROFILE_DIR" >&2
    exit 1
fi
if [[ "$profile_status" != 0 ]]; then
    printf '\nKernel data was exported, but the profiler reported an error.\n' >&2
    printf 'Share the complete folder and run.log:\n%s\n' "$PROFILE_DIR" >&2
    exit 1
fi
printf '\nDone. Share the complete folder:\n%s\n' "$PROFILE_DIR"
printf 'This profiling run intentionally stops before normal PDF/video completion.\n'
