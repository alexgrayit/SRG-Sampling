# Adaptive GPU SRG sampler

The CUDA simulator runs one input comparison per thread. The C++ host updates the sampled hulls for position, velocity and `v + r*x` after each batch. LCM sends the results to the live Python plots, PDF and SVG export, and optional video recording.

## Run

Use Linux with a CUDA toolkit and driver, CMake, a C++17 host compiler and the LCM C++ development package. The launcher defaults to CUDA architecture 89. Set `CUDA_ARCH` for a different GPU. The included CUDA LSODA source is built when `USE_LSODA=ON`.

For the plotter, reuse the original Python environment or set one up with Python 3.12 or later. The plotter uses LaTeX when available and falls back to Matplotlib's math text. Video output uses ffmpeg.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install numpy scipy matplotlib lcm control

# CUDA RKF7(8) with live plots and PDF and SVG export
./run_drag_cuda.sh --samples 32768 --batch-size 2048 --no-video

# CUDA LSODA, without opening the plotter
USE_LSODA=ON ./run_drag_cuda.sh --samples 768 --batch-size 96 --no-plot

# Equal-budget comparison using the same seed and waveform domain
USE_LSODA=ON ./run_drag_cuda.sh --sampler random --samples 768 --batch-size 96 --no-plot

# Linear model with automatic zero-reference pairing
SRG_LINEAR_DRAG=ON ./run_drag_cuda.sh --samples 4096 --no-video

# Quadratic drag with Stribeck sliding friction
USE_LSODA=ON SRG_ENABLE_FRICTION=ON ./run_drag_cuda.sh --batch-size 4096 --no-video

# Plot and save only the three analytical linear SRGs
./run_drag_cuda.sh --ideal-only
```

Set `SRG_PYTHON` to use a Python executable outside `.venv`. Ctrl+C stops the simulator after its in-flight work and sends the completion message so the plotter can save the current figures.

To check CUDA independently of the plotter, run `USE_LSODA=ON ./run_drag_cuda.sh --no-plot --samples 8 --batch-size 8`. The simulator prints when a batch starts and reports elapsed time if it is still running after five seconds. For live plots, omit `--no-plot` and use `--no-video` if you only need the figures. On a non-interactive Matplotlib backend, the plotter saves the files and exits after the run. An existing LCM connection URL can be passed to both processes with `SRG_LCM_URL`.

The combined output uses `r = 2*K_D` by default. Pass `--r VALUE` to use a different value in both the sampled and analytical SRGs. The analytical transfer function for this output is `(s + r)/(s^2 + K_D*s + K_P)`. Its files use `vrx` in the name and include the chosen value of `r`.

Computed SRGs are purple. The plotter saves the overlaid figures and the computed-only figures (`_no_ideal`) as PDF and SVG files in `Python/output/`. The copies use the same axis limits. `--ideal-only` skips simulation and saves the analytical plots with the suffix `_ideal_only` in both formats.

## Sliding friction

Set `SRG_ENABLE_FRICTION=ON` in the launcher or set `ENABLE_FRICTION` to true in the simulation parameters in `SRG Approx Testing GPU/src/drag_sys_cuda.cu`. The default is off. With friction enabled, the acceleration subtracts the sliding force

```text
F(v) = A * (1 + exp(-abs(v)/h)) * tanh(k*v)
```

The editable constants are `FRICTION_STRENGTH` for `A`, `STRIBECK_SPEED` for `h`, and `FRICTION_TANH_SCALE` for `k`. Their starting values are 1, 1 and 10. The force adds to the selected linear or quadratic drag. The analytical overlay uses the linear spring-damper reference with damping `K_D`. Friction-enabled exports include `_friction` in the filename.

## Include an SVG in LaTeX

The SVG labels are editable text, with any mathematics stored as LaTeX source. Install Inkscape and the LaTeX package `svg`. On Ubuntu, install Inkscape with `sudo apt install inkscape`. Compile with shell escape enabled so the package can run Inkscape and pass the labels to LaTeX.

```latex
\usepackage{svg}
% In the document body
\includesvg[width=\linewidth,inkscapelatex=true]{figures/SRG_x_0_2_2_nonlin_friction_no_ideal}
```

The standalone PDFs include their labels and can be included directly with `\includegraphics`.

## Sampling behaviour

- `--sampler adaptive` is the default. The first 108 trial slots are for one-tone seeds across frequency, amplitude and waveform shape, including the nominal spring frequency. After the seeds, one trial in four is reserved for global exploration. The other trials mutate inputs associated with current boundary vertices. Identical waveform configurations are skipped.
- Each boundary vertex is weighted by the plotted area lost if its neighbouring edges are joined directly. Parent selection mixes these weights with uniform selection along the boundary. The default `--target both` splits local trials between position and velocity. Use `--target all` to split them across all three outputs or select one with `--target x`, `--target v` or `--target vrx`.
- The default `--mode hconvex` measures Euclidean area enclosed by the hyperbolic hull in the plotted complex plane. `--mode convex` uses straight edges. `--mode raw` plots individual points and uses the straight-edge hull for sampling. The `area_*` fields on each `SRG_PROGRESS` line include both reflected halves. The hull area is a search objective for the sampled approximation. It provides no coverage guarantee for a general nonlinear operator.
- `--pairing auto` uses a baseline and a perturbation for quadratic drag or sliding friction. A purely linear simulation uses the zero reference. For each pair, the two inputs are `u_base` and `u_base + delta_u`, starting from the same zero state. Both solvers integrate the output differences and the five SRG inner products directly. `--pairing zero` keeps the original comparison against zero input.
- Adaptive and random sampling use continuous frequencies from 0.002 to 10 Hz and a per-term cosine/sine coefficient magnitude from 0.01 to 3333.33. Each of the perturbation and baseline can have up to three terms, with smooth or square components. The domain and model constants are in `src/drag_sys_cuda.cu`, and the term limit is in `include/srg_inputs.hpp`.
- The default budget is 32,768 trials with batches of up to 2,048. Feedback is applied between batches. `--seed` defaults to 42. `--sampler sweep` runs the original combinatorial grid and zero-reference comparison. The profiling script also retains its bounded sweep.

## Copy to another archive

The archive has the same GPU and Python directory layout as the original. Copy these together for the adaptive launcher.

- `run_drag_cuda.sh` and, if used, `verify_cuda.py` or `profile_drag_cuda.sh`
- `SRG Approx Testing GPU/CMakeLists.txt`
- `SRG Approx Testing GPU/src/drag_sys_cuda.cu`
- `SRG Approx Testing GPU/include/srg_inputs.hpp` and `include/srg_sampler.hpp` in the same GPU directory
- `SRG Approx Testing GPU/include/integrators/LSODA/` and `include/integrators/lsoda-cuda/` in the same GPU directory
- The supplied `Python/` plotter and the GPU directory's `lcm_messages/` for the three-output message bindings

The GPU subdirectory's `run_drag_cuda.sh` delegates to the top-level launcher. The adaptive loop has no dependency on the original CPU simulator.

## Verification

```bash
# Build both CUDA solvers and run the checks without a plotter
python3 verify_cuda.py

# Build both solvers on a machine with the CUDA toolkit
python3 verify_cuda.py --build-only
```

The check uses the bundled nonlinear model and the default frequency and amplitude settings. It checks that both solvers accept every trial, that their hull areas agree within 2% on identical seeds in both pairing modes and that adaptive sampling grows the position and velocity hulls. It builds each solver in its own directory under `SRG Approx Testing GPU/build/` and honours `CUDA_ARCH`.

In host execution of the CUDA integration logic with LSODA, adaptive sampling at 768 trials produced 2.38 to 3.41 times the position hull area and 2.55 to 4.29 times the velocity hull area of random sampling across seeds 42, 127 and 2026. CUDA device compilation and execution remain unverified in the development environment.
