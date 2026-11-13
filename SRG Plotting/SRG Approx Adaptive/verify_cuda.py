#!/usr/bin/env python3
"""Build and check both CUDA solver variants for the bundled nonlinear model."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys


SEEDS = 108
ADAPTIVE_SAMPLES = 193
SOLVERS = (("rk", "RKF7(8)", "OFF"), ("lsoda", "LSODA", "ON"))


def build_solver(source, name, use_lsoda):
    build = source / "build" / f"verify-{name}"
    print(f"Building the {name.upper()} CUDA simulator...", flush=True)
    subprocess.run(
        [
            "cmake", "-S", source, "-B", build,
            f"-DCMAKE_CUDA_ARCHITECTURES={os.environ.get('CUDA_ARCH', '89')}",
            "-DCMAKE_BUILD_TYPE=Release", f"-DUSE_LSODA={use_lsoda}",
            "-DSRG_LINEAR_DRAG=OFF", "-DSRG_ENABLE_FRICTION=OFF",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", build, "--config", "Release", "--target", "drag_sys_cuda"],
        check=True,
    )
    for executable in (build / "Release" / "drag_sys_cuda", build / "drag_sys_cuda"):
        if executable.is_file():
            return executable
    raise RuntimeError(f"Build completed but drag_sys_cuda is missing from {build}")


def check_run(executable, integrator, sampler, pairing, samples):
    # Inherited profiler settings would otherwise replace the requested sampler.
    env = os.environ.copy()
    env.pop("SRG_PROFILE_BATCHES", None)
    env.pop("SRG_PROFILE_BATCH_SIZE", None)
    context = f"{integrator} {pairing} {sampler}"
    print(f"Running {context} with {samples} trials...", flush=True)
    lines = []
    with subprocess.Popen(
        [
            executable, "--no-plot", "--sampler", sampler, "--pairing", pairing,
            "--mode", "hconvex", "--seed", "42", "--samples", str(samples),
            "--batch-size", str(SEEDS),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        status = process.wait()
    if status:
        raise RuntimeError(f"{context} exited with status {status}")

    rows = [
        dict(field.split("=", 1) for field in line.split()[1:])
        for line in lines
        if line.startswith("SRG_PROGRESS ")
    ]
    expected = [min(end, samples) for end in range(SEEDS, samples + SEEDS, SEEDS)]
    output = "".join(lines)
    if (f"Integrator: {integrator}" not in output
            or f"Sampler: {sampler}, pairing={pairing}," not in output
            or len(rows) != len(expected)):
        raise RuntimeError(f"{context} did not run the requested configuration")

    areas = []
    for batch, (row, count) in enumerate(zip(rows, expected), 1):
        if (int(row["batch"]) != batch or int(row["evaluated"]) != count
                or int(row["accepted"]) != count or int(row["rejected"]) != 0):
            raise RuntimeError(f"{context} expected {count} accepted trials in batch {batch}. Got {row}")
        areas_in_batch = tuple(float(row[f"area_{output}"]) for output in ("x", "v", "vrx"))
        if not all(math.isfinite(area) and area > 0 for area in areas_in_batch):
            raise RuntimeError(f"{context} produced an invalid hull area in batch {batch}")
        if areas and any(new + 1e-8 < old for new, old in zip(areas_in_batch, areas[-1])):
            raise RuntimeError(f"{context} lost hull area in batch {batch}")
        areas.append(areas_in_batch)

    print(f"{context} passed {samples} trials, final areas {areas[-1]}", flush=True)
    return areas


def check_solvers(executables):
    seeded_areas = []
    for executable, (_, integrator, _) in zip(executables, SOLVERS):
        paired = check_run(executable, integrator, "adaptive", "paired", ADAPTIVE_SAMPLES)
        zero = check_run(executable, integrator, "random", "zero", SEEDS)
        if any(final <= initial for initial, final in zip(paired[0][:2], paired[-1][:2])):
            raise RuntimeError(f"{integrator} failed to grow the position and velocity hulls after the seed batch")
        seeded_areas.append((paired[0], zero[0]))

    # Only compare the seed batches. Subsequent proposals depend on the solver.
    rk_areas, lsoda_areas = seeded_areas
    for pairing, rk, lsoda in zip(("paired", "zero"), rk_areas, lsoda_areas):
        if not all(math.isclose(a, b, rel_tol=0.02) for a, b in zip(rk, lsoda)):
            raise RuntimeError(f"{pairing} seed areas differ between solvers, RK {rk}, LSODA {lsoda}")
    print("Both CUDA solvers passed the seeded area comparison.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true", help="compile both solvers without GPU execution")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "SRG Approx Testing GPU"
    try:
        executables = [build_solver(source, name, enabled) for name, _, enabled in SOLVERS]
        if args.build_only:
            print("Both CUDA solver variants built successfully.")
        else:
            check_solvers(executables)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"CUDA verification failed. {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
