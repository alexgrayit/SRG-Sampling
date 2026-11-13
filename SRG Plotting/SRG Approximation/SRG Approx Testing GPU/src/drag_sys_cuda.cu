/**
 * drag_sys_cuda.cu
 *
 * GPU-accelerated SRG approximate Nyquist diagram computation
 * with combinatorial Fourier-term sweeps.
 *
 * Each simulation uses a different Fourier input:
 *   u(t) = Σ_i [ A_i cos(ω_i t) + B_i sin(ω_i t) ]
 *
 * The program generates all "basis terms" from the cartesian product
 *   A_candidates × B_candidates × ω_candidates
 * then simulates every combination of 1 term, 2 terms, 3 terms, ...
 * up to MAX_COMBO_SIZE.
 *
 * Results stream to LCM in small batches so the Python plotter
 * updates incrementally (cheap combos first).
 *
 * Ctrl+C is safe — GPU memory is reclaimed by the CUDA driver
 * on process exit, and the signal handler stops cleanly between
 * batches.
 *
 * Build:  see CMakeLists.txt
 * Run:    ../Python/run_drag_cuda.sh
 */

#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <csignal>
#include <atomic>
#include <type_traits>

#include <lcm/lcm-cpp.hpp>
#include "lcm_messages/simulation_messages/SRGState.hpp"
#include "lcm_messages/simulation_messages/SRGBatch.hpp"
#include "lcm_messages/simulation_messages/startSim.hpp"
#include "lcm_messages/simulation_messages/SimInfo.hpp"
#include "lcm_messages/simulation_messages/SimDone.hpp"

#ifndef RNum
#define RNum float
#endif

// Compile-time integrator selection: 0 = existing RKF7(8), 1 = LSODA.
// Add the external library as described in LSODA_SETUP.md before enabling it.
#ifndef USE_LSODA
#define USE_LSODA 0
#endif
#if USE_LSODA != 0 && USE_LSODA != 1
#error "USE_LSODA must be 0 or 1"
#endif

#if USE_LSODA
// Set either flag to 0 at compile time to compare against the original path.
#ifndef LSODA_USE_ANALYTIC_JACOBIAN
#define LSODA_USE_ANALYTIC_JACOBIAN 1
#endif
#ifndef LSODA_CACHE_SQUARE_FORCING
#define LSODA_CACHE_SQUARE_FORCING 1
#endif
#if (LSODA_USE_ANALYTIC_JACOBIAN != 0 && LSODA_USE_ANALYTIC_JACOBIAN != 1) || \
    (LSODA_CACHE_SQUARE_FORCING != 0 && LSODA_CACHE_SQUARE_FORCING != 1)
#error "LSODA optimization flags must be 0 or 1"
#endif
#include "integrators/LSODA/lsoda_cuda_bridge.cuh"
#endif

// =============================================================
// Simulation parameters
// =============================================================
static constexpr RNum K_D     = 2.0;
static constexpr RNum K_P     = 2.0;
static constexpr RNum TF      = 211.0;
static constexpr RNum T_TRUNC = TF - 0.0;
static constexpr bool LINEAR_DRAG = false;
#if !USE_LSODA
static constexpr RNum dt_max = RNum(0.01);
static constexpr bool SPLIT_SQUARE_JUMPS = true;
#endif

#if USE_LSODA
// The upstream DLSODA solver and its RHS always use double precision.
static constexpr double LSODA_RTOL = 1e-6;
static constexpr double LSODA_ATOL = 1e-9;
static constexpr int LSODA_MAX_STEPS = 100000; // per call, then continue
static_assert(LSODA_RTOL > 0 && LSODA_ATOL > 0 && LSODA_MAX_STEPS > 0);

// HMAX is the frequency-dependent dt passed to integrate() in simulate_batch.
#endif

// Maximum number of Fourier terms to combine: 1 = singles, 2 = singles +
// pairs, 3 = also triples, etc. Size device scratch from this same setting.
static constexpr int MAX_COMBO_SIZE = 3;
static_assert(MAX_COMBO_SIZE >= 1 && MAX_COMBO_SIZE <= 6,
              "MAX_COMBO_SIZE must be from 1 to 6");
static constexpr int MAX_TERMS = MAX_COMBO_SIZE;

// =============================================================
// Per-simulation configuration — lives in GPU global memory
// =============================================================
struct SimConfig {
    int    n_terms;              // how many active Fourier terms
    RNum A[MAX_TERMS];         // cosine amplitudes
    RNum B[MAX_TERMS];         // sine amplitudes
    RNum omega[MAX_TERMS];     // angular frequencies
    unsigned char waveform[MAX_TERMS]; // bit 0: square cosine, bit 1: square sine
    RNum max_abs_omega;        // cached for dt calculation & sorting
};


struct SimConfigSoA {
    int*    n_terms;
    RNum* A;
    RNum* B;
    RNum* omega;
    unsigned char* waveform;
    RNum* max_abs_omega;

    int stride;
};

// Per-simulation output
struct SRGResult {
    RNum x_norm_sq;   // ∫ x² dt
    RNum v_norm_sq;   // ∫ v² dt
    RNum u_norm_sq;   // ∫ u² dt
    RNum L2_x;        // ∫ x·u dt
    RNum L2_v;        // ∫ v·u dt
#if USE_LSODA
    int solver_status; // 2 = completed, negative = failure
    int solver_detail; // upstream diagnostic code, or bridge diagnostic
    double solver_time;
#endif
};

// =============================================================
// Signal handling for clean Ctrl+C
// =============================================================
static std::atomic<bool> g_shutdown{false};
#if USE_LSODA
static bool g_lsoda_failed = false;
#endif

static void signal_handler(int /*sig*/) {
    g_shutdown.store(true);
}

// =============================================================
// Device code
// =============================================================
__global__ void pack_configs_soa(
    const SimConfig* src,
    SimConfigSoA dst,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count)
        return;

    dst.n_terms[idx] = src[idx].n_terms;
    dst.max_abs_omega[idx] = src[idx].max_abs_omega;

    #pragma unroll
    for (int i = 0; i < MAX_TERMS; ++i) {
        const int dst_idx = i * dst.stride + idx;

        dst.A[dst_idx]     = src[idx].A[i];
        dst.B[dst_idx]     = src[idx].B[i];
        dst.omega[dst_idx] = src[idx].omega[i];
        dst.waveform[dst_idx] = src[idx].waveform[i];
    }
}

// Select the trigonometric overload from RNum, including when it is double.
template<class Real>
__device__ void sincos_rnum(Real phase, Real* s, Real* c) {
    if constexpr (std::is_same_v<Real, float>) sincosf(phase, s, c);
    else sincos(phase, s, c);
}

/** Fourier sum using per-simulation config from global memory. */
template<class Real>
__device__ Real fourier_sum_dev(
    const SimConfigSoA& cfg,
    int idx,
    double t,
    const RNum* square_values)
{
    Real u = Real(0);
    const int n_terms = cfg.n_terms[idx];
    for (int i = 0; i < n_terms; ++i) {
        const int config_idx = i * cfg.stride + idx;
        const Real omega = Real(cfg.omega[config_idx]);
        const Real A = Real(cfg.A[config_idx]);
        const Real B = Real(cfg.B[config_idx]);
        const auto waveform = cfg.waveform[config_idx];
        Real s = Real(0), c = Real(0);
        if (!square_values || (A != 0 && !(waveform & 1)) || (B != 0 && !(waveform & 2)))
            sincos_rnum(Real(double(omega) * t), &s, &c);
        // A and B independently choose smooth or sign waves.
        // https://en.wikipedia.org/wiki/Square_wave
        u += A * ((waveform & 1) ? (square_values ? Real(square_values[2*i]) : (c >= 0 ? Real(1) : Real(-1))) : c);
        u += B * ((waveform & 2) ? (square_values ? Real(square_values[2*i+1]) : (s >= 0 ? Real(1) : Real(-1))) : s);
    }
    return u;
}

/**
 * State derivative.
 * State layout (7 values, with the integrator's arithmetic type):
 *   [0] ∫x²   [1] ∫v²   [2] ∫u²
 *   [3] x      [4] v
 *   [5] ∫x·u   [6] ∫v·u
 */
template<class Real>
__device__ void state_deriv_for_input(
    const Real* x,
    Real* xdot,
    Real u)
{
    Real pos = x[3];
    Real vel = x[4];
    Real dvdt = u - Real(K_D) * vel - Real(K_P) * pos;
    if (!LINEAR_DRAG) {
        Real abs_vel = vel >= 0 ? vel : -vel;
        dvdt = u - Real(0.5) * Real(K_D) * vel * abs_vel - Real(K_P) * pos;
    }

    xdot[0] = pos * pos;
    xdot[1] = vel * vel;
    xdot[2] = u   * u;
    xdot[3] = vel;
    xdot[4] = dvdt;
    xdot[5] = pos * u;
    xdot[6] = vel * u;
}

template<class Real>
__device__ void state_deriv(
    double t,
    const Real* x,
    Real* xdot,
    const SimConfigSoA& cfg,
    int idx,
    const RNum* square_values,
    bool input_active)
{
    Real u = input_active ? fourier_sum_dev<Real>(cfg, idx, t, square_values) : Real(0);
    state_deriv_for_input(x, xdot, u);
}

#if USE_LSODA
// Capture each simulation's forcing without global callback state.
struct LsodaRhs {
    SimConfigSoA configs;
    int idx;
    const RNum* square_values;
    bool input_active;
#if LSODA_CACHE_SQUARE_FORCING
    bool input_is_constant = false;
    double held_input = 0;
#endif

    __device__ double input(double t) const {
#if LSODA_CACHE_SQUARE_FORCING
        if (input_is_constant) return held_input;
#endif
        return input_active
            ? fourier_sum_dev<double>(configs, idx, t, square_values) : 0.0;
    }

    __device__ void operator()(int*, double* t, double* y, double* ydot, double*) const {
        state_deriv_for_input(y, ydot, input(*t));
    }
};

#if LSODA_USE_ANALYTIC_JACOBIAN
// This Jacobian is specific to the existing seven-state mass-spring-damper.
// Replace it when adding different dynamics, or disable the flag above.
struct LsodaJacobian {
    LsodaRhs rhs;

    __device__ void operator()(int*, double* t, double* y, int, int,
                               double* jacobian, int leading_dimension) const {
        // DLSODA clears the dense buffer before calling JAC. Fill its seven
        // nonzero entries, in column-major order: df[row]/dy[column].
        const double u = rhs.input(*t);
        jacobian[3 * leading_dimension + 0] = 2.0 * y[3];
        jacobian[4 * leading_dimension + 1] = 2.0 * y[4];
        jacobian[4 * leading_dimension + 3] = 1.0;
        jacobian[3 * leading_dimension + 4] = -double(K_P);
        jacobian[4 * leading_dimension + 4] = LINEAR_DRAG
            ? -double(K_D) : -double(K_D) * fabs(y[4]);
        jacobian[3 * leading_dimension + 5] = u;
        jacobian[4 * leading_dimension + 6] = u;
    }
};
#endif
#else
/** Fixed-step RKF7(8).  All scratch in thread-local registers. */
__device__ void rkf78_step_dev(
    double t,
    double h,
    const RNum* x,
    RNum* x_next,
    const SimConfigSoA& cfg,
    int idx,
    const RNum* square_values,
    bool input_active)
{
    const RNum step = RNum(h);
    constexpr int N = 7;
    RNum k[13][N];
    RNum xtmp[N];

    // Stage 0
    state_deriv(t, x, k[0], cfg, idx, square_values, input_active);

    // Stage 1
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(2.0/27.0)*k[0][i]);
    state_deriv(t + h*2.0/27.0, xtmp, k[1], cfg, idx, square_values, input_active);

    // Stage 2
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(1.0/36.0)*k[0][i] + RNum(1.0/12.0)*k[1][i]);
    state_deriv(t + h/9.0, xtmp, k[2], cfg, idx, square_values, input_active);

    // Stage 3
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(1.0/24.0)*k[0][i] + RNum(1.0/8.0)*k[2][i]);
    state_deriv(t + h/6.0, xtmp, k[3], cfg, idx, square_values, input_active);

    // Stage 4
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(5.0/12.0)*k[0][i] - RNum(25.0/16.0)*k[2][i]
                             + RNum(25.0/16.0)*k[3][i]);
    state_deriv(t + h*5.0/12.0, xtmp, k[4], cfg, idx, square_values, input_active);

    // Stage 5
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(1.0/20.0)*k[0][i] + RNum(1.0/4.0)*k[3][i]
                             + RNum(1.0/5.0)*k[4][i]);
    state_deriv(t + h*0.5, xtmp, k[5], cfg, idx, square_values, input_active);

    // Stage 6
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(-RNum(25.0/108.0)*k[0][i] + RNum(125.0/108.0)*k[3][i]
                             - RNum(65.0/27.0)*k[4][i] + RNum(125.0/54.0)*k[5][i]);
    state_deriv(t + h*5.0/6.0, xtmp, k[6], cfg, idx, square_values, input_active);

    // Stage 7
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(31.0/300.0)*k[0][i] + RNum(61.0/225.0)*k[4][i]
                             - RNum(2.0/9.0)*k[5][i] + RNum(13.0/900.0)*k[6][i]);
    state_deriv(t + h/6.0, xtmp, k[7], cfg, idx, square_values, input_active);

    // Stage 8
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(2.0)*k[0][i] - RNum(53.0/6.0)*k[3][i]
                             + RNum(704.0/45.0)*k[4][i] - RNum(107.0/9.0)*k[5][i]
                             + RNum(67.0/90.0)*k[6][i] + RNum(3.0)*k[7][i]);
    state_deriv(t + h*2.0/3.0, xtmp, k[8], cfg, idx, square_values, input_active);

    // Stage 9
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(-RNum(91.0/108.0)*k[0][i] + RNum(23.0/108.0)*k[3][i]
                             - RNum(976.0/135.0)*k[4][i] + RNum(311.0/54.0)*k[5][i]
                             - RNum(19.0/60.0)*k[6][i] + RNum(17.0/6.0)*k[7][i]
                             - RNum(1.0/12.0)*k[8][i]);
    state_deriv(t + h/3.0, xtmp, k[9], cfg, idx, square_values, input_active);

    // Stage 10
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(2383.0/4100.0)*k[0][i] - RNum(341.0/164.0)*k[3][i]
                             + RNum(4496.0/1025.0)*k[4][i] - RNum(301.0/82.0)*k[5][i]
                             + RNum(2133.0/4100.0)*k[6][i] + RNum(45.0/82.0)*k[7][i]
                             + RNum(45.0/164.0)*k[8][i] + RNum(18.0/41.0)*k[9][i]);
    state_deriv(t + h, xtmp, k[10], cfg, idx, square_values, input_active);

    // Stage 11
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(3.0/205.0)*k[0][i] - RNum(6.0/41.0)*k[5][i]
                             - RNum(3.0/205.0)*k[6][i] - RNum(3.0/41.0)*k[7][i]
                             + RNum(3.0/41.0)*k[8][i] + RNum(6.0/41.0)*k[9][i]);
    state_deriv(t, xtmp, k[11], cfg, idx, square_values, input_active);

    // Stage 12
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(-RNum(1777.0/4100.0)*k[0][i] - RNum(341.0/164.0)*k[3][i]
                             + RNum(4496.0/1025.0)*k[4][i] - RNum(289.0/82.0)*k[5][i]
                             + RNum(2193.0/4100.0)*k[6][i] + RNum(51.0/82.0)*k[7][i]
                             + RNum(33.0/164.0)*k[8][i] + RNum(12.0/41.0)*k[9][i]
                             + k[11][i]);
    state_deriv(t + h, xtmp, k[12], cfg, idx, square_values, input_active);

    // 7th-order solution
    for (int i = 0; i < N; ++i) {
        x_next[i] = x[i] + step*(
            RNum(34.0/105.0)*k[5][i]  + RNum(9.0/35.0)*k[6][i]  +
             RNum(9.0/35.0)*k[7][i]   + RNum(9.0/280.0)*k[8][i]  +
             RNum(9.0/280.0)*k[9][i]  + RNum(41.0/840.0)*k[11][i] +
            RNum(41.0/840.0)*k[12][i]
        );
    }
}
#endif

// =============================================================
// Kernel — one thread per simulation
// =============================================================
// Sine jumps occur at n*pi/|omega|; cosine jumps at (n+1/2)*pi/|omega|.
__device__ double next_square_jump(double t, RNum omega, bool cosine) {
    double w = fabs(double(omega));
    if (w == 0) return INFINITY;
    double period = M_PI / w;
    double offset = cosine ? 0.5 : 0.0;
    double n = floor(t / period - offset) + 1.0;
    double next = (n + offset) * period;
    if (next <= t + 16.0 * 2.2204460492503131e-16 * (1.0 + t))
        next = (++n + offset) * period;
    return next;
}

#if USE_LSODA
// LSODA has a larger call stack and workspace than the RK kernel.
__global__ __launch_bounds__(128)
#else
__global__ __launch_bounds__(128, 4)
#endif
void simulate_batch(
    SimConfigSoA configs,
    SRGResult* results,
    int count
#if USE_LSODA
    , srg_lsoda::Workspace<7>* lsoda_work
#endif
    )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    RNum f_max = configs.max_abs_omega[idx] / RNum(2.0 * M_PI);
    RNum time_scale = K_D > 0 ? K_D : RNum(1);
    #if USE_LSODA
    RNum dt = f_max < RNum(1e-12) ? RNum(0.005) / time_scale
        : RNum(0.5) / (f_max);
    #else
    RNum dt = f_max < RNum(1e-12) ? RNum(0.005) / time_scale
        : RNum(0.005) / time_scale / (RNum(6) * f_max);
    dt = fmin(dt, dt_max); // dt_max is a maximum
    #endif

#if USE_LSODA
    double state[7] = {};
    int solver_status = 2;
    int solver_detail = 0;
    // Adaptive multistep methods must restart at forcing discontinuities.
    constexpr bool split_jumps = true;
#else
    RNum state[7] = {0,0,0,0,0,0,0};
    RNum state_next[7];
    constexpr bool split_jumps = SPLIT_SQUARE_JUMPS;
#endif
    RNum square_values[2 * MAX_TERMS];
    double next_cos[MAX_TERMS], next_sin[MAX_TERMS];
    int n_terms = configs.n_terms[idx];
#if USE_LSODA && LSODA_CACHE_SQUARE_FORCING
    bool square_only = true;
#endif
    for (int i = 0; i < n_terms; ++i) {
        int j = i * configs.stride + idx;
#if USE_LSODA && LSODA_CACHE_SQUARE_FORCING
        if ((configs.A[j] != 0 && !(configs.waveform[j] & 1)) ||
            (configs.B[j] != 0 && !(configs.waveform[j] & 2)))
            square_only = false;
#endif
        square_values[2*i] = RNum(1);
        square_values[2*i+1] = configs.omega[j] >= 0 ? RNum(1) : RNum(-1);
        next_cos[i] = configs.A[j] != 0 && (configs.waveform[j] & 1)
            ? next_square_jump(0, configs.omega[j], true) : INFINITY;
        next_sin[i] = configs.B[j] != 0 && (configs.waveform[j] & 2)
            ? next_square_jump(0, configs.omega[j], false) : INFINITY;
    }

    // The clock stays double. RK follows RNum; LSODA uses native double.
    double t = 0;
    while (t < double(TF)) {
        bool input_active = t < double(T_TRUNC);
#if USE_LSODA
        // One adaptive solve per smooth interval, keeping history within it.
        double end = double(TF);
#else
        double end = fmin(t + double(dt), double(TF));
#endif
        if (input_active) end = fmin(end, double(T_TRUNC));
        if (input_active && split_jumps) {
            for (int i = 0; i < n_terms; ++i) {
                end = fmin(end, next_cos[i]);
                end = fmin(end, next_sin[i]);
            }
        }
        // Squares are constant on each interval, including its endpoints.
#if USE_LSODA
        LsodaRhs rhs{configs, idx, square_values, input_active};
#if LSODA_CACHE_SQUARE_FORCING
        // Cache only intervals with no active smooth contribution. Squares
        // already have fixed signs until end; mixed/smooth inputs stay dynamic.
        rhs.input_is_constant = !input_active || square_only;
        if (rhs.input_is_constant && input_active)
            rhs.held_input = fourier_sum_dev<double>(configs, idx, t, square_values);
#endif
        solver_status = srg_lsoda::integrate<7>(
            rhs, state, t, end, double(dt), LSODA_RTOL, LSODA_ATOL,
            LSODA_MAX_STEPS, lsoda_work[idx], solver_detail
#if LSODA_USE_ANALYTIC_JACOBIAN
            , LsodaJacobian{rhs}
#endif
            );
        if (solver_status != 2) break;
#else
        rkf78_step_dev(t, end - t, state, state_next, configs, idx,
                      split_jumps ? square_values : nullptr, input_active);
        for (int i = 0; i < 7; ++i) state[i] = state_next[i];
#endif
        t = end;
        if (input_active && split_jumps) {
            double tolerance = 16.0 * 2.2204460492503131e-16 * (1.0 + t);
            for (int i = 0; i < n_terms; ++i) {
                int j = i * configs.stride + idx;
                if (next_cos[i] <= t + tolerance) {
                    square_values[2*i] = -square_values[2*i];
                    next_cos[i] = next_square_jump(t, configs.omega[j], true);
                }
                if (next_sin[i] <= t + tolerance) {
                    square_values[2*i+1] = -square_values[2*i+1];
                    next_sin[i] = next_square_jump(t, configs.omega[j], false);
                }
            }
        }
    }
    results[idx].x_norm_sq = state[0];
    results[idx].v_norm_sq = state[1];
    results[idx].u_norm_sq = state[2];
    results[idx].L2_x = state[5];
    results[idx].L2_v = state[6];
#if USE_LSODA
    results[idx].solver_status = solver_status;
    results[idx].solver_detail = solver_detail;
    results[idx].solver_time = t;
#endif
}

// =============================================================
// Host: combination generator
// =============================================================

/** A single Fourier "basis term": one (A, B, ω) triple. */
struct BasisTerm {
    RNum A;
    RNum B;
    RNum omega;
};

/**
 * Build all basis terms from the cartesian product
 *   A_vals × B_vals × omega_vals
 *
 * Skips the trivial (A=0, B=0) term since it produces zero input.
 */
static std::vector<BasisTerm> build_basis_terms(
    const std::vector<RNum>& A_vals,
    const std::vector<RNum>& B_vals,
    const std::vector<RNum>& omega_vals)
{
    std::vector<BasisTerm> terms;
    for (RNum a : A_vals)
        for (RNum b : B_vals)
            for (RNum w : omega_vals) {
                if (a == 0.0 && b == 0.0) continue;  // skip zero input
                terms.push_back({a, b, w});
            }
    std::stable_sort(terms.begin(), terms.end(), [](const BasisTerm& a, const BasisTerm& b) {
        return fabs(a.omega) < fabs(b.omega);
    });
    return terms;
}

/**
 * Turn one combination of basis-term indices into a SimConfig.
 */
static SimConfig make_config(const std::vector<BasisTerm>& basis,
                             const int* indices, int k)
{
    SimConfig cfg{};
    cfg.n_terms = k;
    RNum wmax = 0.0;
    for (int i = 0; i < k; ++i) {
        const auto& bt = basis[indices[i]];
        cfg.A[i]     = bt.A;
        cfg.B[i]     = bt.B;
        cfg.omega[i] = bt.omega;
        wmax = fmax(wmax, fabs(bt.omega));
    }
    cfg.max_abs_omega = wmax;
    return cfg;
}

// Distinct basis entries form C(n,k) sums, without permutations.
// https://en.wikipedia.org/wiki/Combination
// Emit immediately so the entire expanded sweep is not held in memory.
template<class Emit>
static bool generate_combinations(
    const std::vector<BasisTerm>& basis, int k,
    int first_last, int last_end, int family, Emit emit)
{
    for (int last = std::max(first_last, k - 1); last < last_end; ++last) {
        std::vector<int> idx(k);
        for (int i = 0; i < k - 1; ++i) idx[i] = i;
        idx[k - 1] = last;
        while (true) {
            SimConfig cfg = make_config(basis, idx.data(), k);
            int components = 0;
            for (int i = 0; i < k; ++i)
                components += (cfg.A[i] != 0) + (cfg.B[i] != 0);
            unsigned int all_square = (1u << components) - 1;
            auto emit_mask = [&](unsigned int mask) {
                int component = 0;
                for (int i = 0; i < k; ++i) {
                    cfg.waveform[i] = 0;
                    if (cfg.A[i] != 0 && ((mask >> component++) & 1u)) cfg.waveform[i] |= 1;
                    if (cfg.B[i] != 0 && ((mask >> component++) & 1u)) cfg.waveform[i] |= 2;
                }
                return emit(cfg, family);
            };
            // There are 2^m assignments for m nonzero contributions.
            // First the all-square and all-smooth assignments, then the rest.
            if (family != 2) {
                if (!emit_mask(family == 0 ? all_square : 0)) return false;
            } else {
                for (unsigned int mask = 1; mask < all_square; ++mask)
                    if (!emit_mask(mask)) return false;
            }
            int i = k - 2;
            while (i >= 0 && idx[i] == last - (k - 1) + i) --i;
            if (i < 0) break;
            ++idx[i];
            for (int j = i + 1; j < k - 1; ++j) idx[j] = idx[j-1] + 1;
        }
    }
    return true;
}

template<class Emit>
static bool generate_waveform_sweep(const std::vector<BasisTerm>& basis,
                                     int MAX_COMBO_SIZE, Emit emit)
{
    int n = int(basis.size());
    for (int family = 0; family < 3; ++family) {
        for (int first = 0; first < n;) {
            int end = first + 1;
            while (end < n && fabs(basis[end].omega) == fabs(basis[first].omega)) ++end;
            // Fix the highest-frequency entry, then choose its lower-index
            // companions: each sum appears once, in the original frequency order.
            for (int k = 1; k <= MAX_COMBO_SIZE && k <= end; ++k)
                if (!generate_combinations(basis, k, first, end, family, emit)) return false;
            first = end;
        }
    }
    return true;
}

// =============================================================
// Host: LCM helpers
// =============================================================

static bool start_flag = false;

class SimStartHandler {
public:
    void handleMessage(const lcm::ReceiveBuffer*, const std::string&,
                       const simulation_messages::startSim* msg)
    {
        printf("Received startSim: %s\n", msg->sim_name.c_str());
        start_flag = true;
    }
};

static int publish_batch(
    lcm::LCM& lc,
    const SRGResult* results,
    int count)
{
#if USE_LSODA
    if (g_lsoda_failed) return 0;
    // Never publish a partly integrated or non-finite batch as an SRG result.
    for (int i = 0; i < count; ++i) {
        const auto& r = results[i];
        if (r.solver_status != 2 || !std::isfinite(r.x_norm_sq) ||
            !std::isfinite(r.v_norm_sq) || !std::isfinite(r.u_norm_sq) ||
            !std::isfinite(r.L2_x) || !std::isfinite(r.L2_v) ||
            r.x_norm_sq < 0 || r.v_norm_sq < 0 || r.u_norm_sq < 0) {
            fprintf(stderr, "LSODA batch rejected: simulation %d, t=%.17g, "
                    "ISTATE=%d, diagnostic=%d (also check finite, nonnegative norms).\n",
                    i, r.solver_time, r.solver_status, r.solver_detail);
            g_lsoda_failed = true;
            g_shutdown.store(true);
            return 0;
        }
    }
#endif
    constexpr int MAX_LCM_BATCH = 512;

    for (int offset = 0; offset < count; offset += MAX_LCM_BATCH) {

        const int batch_count =
            std::min(MAX_LCM_BATCH, count - offset);

        simulation_messages::SRGBatch msg;

        msg.n = batch_count;

        msg.x_norm.resize(batch_count);
        msg.v_norm.resize(batch_count);
        msg.u_norm.resize(batch_count);
        msg.L2_x.resize(batch_count);
        msg.L2_v.resize(batch_count);

        for (int i = 0; i < batch_count; ++i) {

            const SRGResult& result = results[offset + i];

            msg.x_norm[i] = sqrt(result.x_norm_sq);
            msg.v_norm[i] = sqrt(result.v_norm_sq);
            msg.u_norm[i] = sqrt(result.u_norm_sq);
            msg.L2_x[i]   = result.L2_x;
            msg.L2_v[i]   = result.L2_v;
        }

        lc.publish("SRGBatch", &msg);
    }
    return count;
}
// =============================================================
// Main
// =============================================================
static void send_sim_done(lcm::LCM& lc, size_t n_simulations)
{
    simulation_messages::SimDone msg;
    msg.n_simulations = static_cast<int64_t>(n_simulations);
    msg.cancelled = g_shutdown.load();
    lc.publish("SimDone", &msg);
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--profile-support") == 0) {
        std::puts("SRG_PROFILE_BOUNDED_V2");
        return 0;
    }
    if (argc != 1) {
        std::fprintf(stderr, "Usage: %s [--profile-support]\n", argv[0]);
        return 2;
    }
    // The profiling script sets this only for bounded diagnostic/capture runs.
    // Normal simulation runs retain the complete sweep and batch size.
    constexpr int DEFAULT_BATCH_SIZE = 81920;
    size_t profile_batch_limit = 0;
    if (const char* value = std::getenv("SRG_PROFILE_BATCHES"); value && *value) {
        char* end = nullptr;
        errno = 0;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (std::strspn(value, "0123456789") != std::strlen(value) ||
            errno == ERANGE || *end || parsed == 0 || parsed > 2147483647ULL) {
            std::fprintf(stderr, "SRG_PROFILE_BATCHES must be an integer from 1 to 2147483647.\n");
            return 2;
        }
        profile_batch_limit = size_t(parsed);
    }
    int profile_batch_size = DEFAULT_BATCH_SIZE;
    if (profile_batch_limit) {
        if (const char* value = std::getenv("SRG_PROFILE_BATCH_SIZE"); value && *value) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (std::strspn(value, "0123456789") != std::strlen(value) ||
                errno == ERANGE || *end || parsed == 0 || parsed > DEFAULT_BATCH_SIZE) {
                std::fprintf(stderr, "SRG_PROFILE_BATCH_SIZE must be an integer from 1 to %d.\n",
                             DEFAULT_BATCH_SIZE);
                return 2;
            }
            profile_batch_size = int(parsed);
        }
    }
    const int BATCH_SIZE = profile_batch_limit ? profile_batch_size : DEFAULT_BATCH_SIZE;
    // Preserve configuration messages when the profiler stops the process.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    // ---- Install signal handler for clean Ctrl+C ----
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ---- LCM setup ----
    lcm::LCM lc(
        "udpm://239.255.76.67:7667?ttl=0&recv_buf_size=8388608"
    );
    if (!lc.good()) { fprintf(stderr, "LCM init failed\n"); return 1; }

    auto cuda_require = [&](cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA_ERROR operation=%s code=%d name=%s message=%s\n",
                         operation, int(status), cudaGetErrorName(status), cudaGetErrorString(status));
            g_shutdown.store(true);
            send_sim_done(lc, 0);
            std::exit(EXIT_FAILURE);
        }
    };

    SimStartHandler handler;
    lc.subscribe("startSim", &SimStartHandler::handleMessage, &handler);

    printf("Waiting for startSim message...\n");
    // Even an early cancellation replies after Python has subscribed.
    while (!start_flag) lc.handleTimeout(100);
    simulation_messages::SimInfo info;
    info.k_d = K_D;
    info.k_p = K_P;
    info.linear_drag = LINEAR_DRAG;
    lc.publish("SimInfo", &info); // one reply before any result batches
    if (g_shutdown.load()) {
        send_sim_done(lc, 0);
        return 0;
    }
#if USE_LSODA
    printf("Integrator: LSODA (double), rtol=%g, atol=%g\n", LSODA_RTOL, LSODA_ATOL);
    printf("LSODA options: analytic Jacobian=%d, square forcing cache=%d; "
           "frequency-dependent HMAX unchanged\n",
           LSODA_USE_ANALYTIC_JACOBIAN, LSODA_CACHE_SQUARE_FORCING);
#else
    printf("Integrator: RKF7(8) (RNum)\n");
#endif

    if (profile_batch_limit) {
        std::printf("PROFILE_LIMIT batches=%zu\n", profile_batch_limit);
        std::printf("PROFILE_BATCH_SIZE simulations=%d normal=%d\n", BATCH_SIZE, DEFAULT_BATCH_SIZE);
        cudaDeviceProp properties{};
        int device = 0;
        cuda_require(cudaGetDevice(&device), "cudaGetDevice");
        cuda_require(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        int timeout_enabled = 0;
        cuda_require(cudaDeviceGetAttribute(&timeout_enabled, cudaDevAttrKernelExecTimeout, device),
                     "cudaDeviceGetAttribute(kernel timeout)");
        size_t stack_limit = 0, free_memory = 0, total_memory = 0;
        cuda_require(cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize), "cudaDeviceGetLimit(stack)");
        cuda_require(cudaMemGetInfo(&free_memory, &total_memory), "cudaMemGetInfo");
        std::printf("PROFILE_DEVICE name=%s timeout_enabled=%d stack_limit=%zu "
                    "free_MiB=%.1f total_MiB=%.1f\n", properties.name,
                    timeout_enabled, stack_limit,
                    free_memory / (1024.0 * 1024.0), total_memory / (1024.0 * 1024.0));
    }

    // ============================================================
    // *** CONFIGURE YOUR SWEEP HERE ***
    //
    // Edit these three vectors and MAX_COMBO_SIZE.
    //
    // A basis term is one (A, B, ω) triple from A × B × ω.
    // The program simulates every combination of 1, 2, ...,
    // MAX_COMBO_SIZE basis terms as summed Fourier inputs.
    //
    // A and B choose their waveforms independently. The three complete
    // passes are square-only, sine/cosine-only, and mixed sums.
    // ============================================================

    // Cosine amplitudes to try
    std::vector<RNum> A_vals = { 0.0, 0.01 , 1.0 , 1340.9};

    // Sine amplitudes to try
    std::vector<RNum> B_vals = { 0.02 , 1.0 , 10.0, 3333.33};

    // Frequencies to try (rad/s)
    constexpr int    N_OMEGA   = 10;
    constexpr RNum OMEGA_MAX = 10.0 * 2.0 * M_PI;
    constexpr RNum OMEGA_MIN = 0.002 * 2.0 * M_PI;
    const RNum omega_0  = OMEGA_MIN;
    const RNum n_omega  = (RNum)N_OMEGA / log(OMEGA_MAX / OMEGA_MIN);

    // Two simulations per frequency index (positive & negative omega)
    const int n_omega_total = N_OMEGA * 2;
    std::vector<RNum> omega_vals(n_omega_total);

    for (int n = 0; n < N_OMEGA; ++n) {
        RNum w = omega_0 * exp((RNum)n / n_omega);
        omega_vals[2*n + 0] =  w;
        omega_vals[2*n + 1] = -w;
    }

    // MAX_COMBO_SIZE near the top controls the sweep and device capacity.

    // ============================================================
    // Build all simulation configs
    // ============================================================
    auto basis = build_basis_terms(A_vals, B_vals, omega_vals);
    printf("Basis terms: %zu  (from |A|=%zu x |B|=%zu x |omega|=%zu, "
           "excluding A=B=0)\n",
           basis.size(), A_vals.size(), B_vals.size(), omega_vals.size());

    // Count the weighted combinations without constructing them.
    // A term contributes 2^m choices, with m its nonzero amplitudes.
    std::vector<long double> combinations(MAX_COMBO_SIZE + 1, 0);
    std::vector<long double> assignments(MAX_COMBO_SIZE + 1, 0);
    combinations[0] = assignments[0] = 1;
    for (const auto& term : basis) {
        int m = (term.A != 0) + (term.B != 0);
        for (int k = MAX_COMBO_SIZE; k > 0; --k) {
            combinations[k] += combinations[k-1];
            assignments[k] += (1u << m) * assignments[k-1];
        }
    }
    long double counts[3] = {0, 0, 0};
    for (int k = 1; k <= MAX_COMBO_SIZE; ++k) {
        counts[0] += combinations[k];
        counts[1] += combinations[k];
        counts[2] += assignments[k] - 2 * combinations[k];
    }
    const long double n_total = counts[0] + counts[1] + counts[2];
    printf("Square: %.0Lf, sine: %.0Lf, mixed: %.0Lf; total: %.0Lf\n",
           counts[0], counts[1], counts[2], n_total);
    size_t n_batches = 0;
    for (long double count : counts) n_batches += size_t(ceil(count / BATCH_SIZE));

    if (profile_batch_limit > n_batches) {
        std::fprintf(stderr, "Requested %zu profile batches, but this sweep has only %zu.\n",
                     profile_batch_limit, n_batches);
        g_shutdown.store(true);
        send_sim_done(lc, 0);
        return 2;
    }
    const size_t planned_batches = profile_batch_limit ? profile_batch_limit : n_batches;

    // GPU buffers retain their original double-buffered layout.
    // AoS staging buffer used only for H->D transfer
    SimConfig* d_configs_aos[2];

    // SoA representation used by the simulation kernel
    SimConfigSoA d_configs[2];

    SRGResult* d_results[2];
#if USE_LSODA
    // Separate persistent scratch for every simulation and both CUDA streams.
    srg_lsoda::Workspace<7>* d_lsoda_work[2] = {nullptr, nullptr};
    printf("LSODA workspaces: %.1f MiB for both buffers\n",
           2.0 * BATCH_SIZE * sizeof(srg_lsoda::Workspace<7>) / (1024.0 * 1024.0));
    for (int i = 0; i < 2; ++i) {
        cudaError_t allocation = cudaMalloc(&d_lsoda_work[i],
            BATCH_SIZE * sizeof(srg_lsoda::Workspace<7>));
        if (allocation != cudaSuccess) {
            fprintf(stderr, "LSODA workspace allocation failed: %s. Reduce BATCH_SIZE.\n",
                    cudaGetErrorString(allocation));
            cudaFree(d_lsoda_work[0]);
            cudaFree(d_lsoda_work[1]);
            g_shutdown.store(true);
            send_sim_done(lc, 0);
            return 1;
        }
    }
#endif

    for (int i = 0; i < 2; ++i) {

        // Temporary AoS upload
        cuda_require(cudaMalloc(
            &d_configs_aos[i],
            BATCH_SIZE * sizeof(SimConfig)
        ), "cudaMalloc");

        // SoA arrays
        cuda_require(cudaMalloc(
            &d_configs[i].n_terms,
            BATCH_SIZE * sizeof(int)
        ), "cudaMalloc");

        cuda_require(cudaMalloc(
            &d_configs[i].max_abs_omega,
            BATCH_SIZE * sizeof(RNum)
        ), "cudaMalloc");

        cuda_require(cudaMalloc(
            &d_configs[i].A,
            MAX_TERMS * BATCH_SIZE * sizeof(RNum)
        ), "cudaMalloc");

        cuda_require(cudaMalloc(
            &d_configs[i].B,
            MAX_TERMS * BATCH_SIZE * sizeof(RNum)
        ), "cudaMalloc");

        cuda_require(cudaMalloc(
            &d_configs[i].omega,
            MAX_TERMS * BATCH_SIZE * sizeof(RNum)
        ), "cudaMalloc");
        cuda_require(cudaMalloc(&d_configs[i].waveform,
                   MAX_TERMS * BATCH_SIZE * sizeof(unsigned char)), "cudaMalloc");

        d_configs[i].stride = BATCH_SIZE;

        cuda_require(cudaMalloc(
            &d_results[i],
            BATCH_SIZE * sizeof(SRGResult)
        ), "cudaMalloc");
    }

    // Pinned host buffers for async D→H (RNum-buffered)
    SRGResult* h_pinned[2];
    cuda_require(cudaHostAlloc(&h_pinned[0], BATCH_SIZE * sizeof(SRGResult),
                  cudaHostAllocDefault), "cudaHostAlloc");
    cuda_require(cudaHostAlloc(&h_pinned[1], BATCH_SIZE * sizeof(SRGResult),
                  cudaHostAllocDefault), "cudaHostAlloc");
    SimConfig* h_configs[2];
    cuda_require(cudaHostAlloc(&h_configs[0], BATCH_SIZE * sizeof(SimConfig), cudaHostAllocDefault), "cudaHostAlloc");
    cuda_require(cudaHostAlloc(&h_configs[1], BATCH_SIZE * sizeof(SimConfig), cudaHostAllocDefault), "cudaHostAlloc");

    cudaStream_t streams[2];
    cuda_require(cudaStreamCreate(&streams[0]), "cudaStreamCreate");
    cuda_require(cudaStreamCreate(&streams[1]), "cudaStreamCreate");
    cudaEvent_t kernel_begin[2]{}, kernel_end[2]{};
    if (profile_batch_limit) {
        for (int i = 0; i < 2; ++i) {
            cuda_require(cudaEventCreate(&kernel_begin[i]), "cudaEventCreate(begin)");
            cuda_require(cudaEventCreate(&kernel_end[i]), "cudaEventCreate(end)");
        }
    }

    // ============================================================
    // Streaming launch loop
    // ============================================================
    constexpr int BLOCK = 128;

    printf("Launching in %zu batches of <=%d  (Ctrl+C to stop early)...\n",
           planned_batches, BATCH_SIZE);

    auto t0 = std::chrono::high_resolution_clock::now();

    int prev_buf   = -1;
    int prev_count =  0;
    cudaStream_t prev_stream = nullptr;
    size_t sims_published = 0;
    size_t b = 0;
    bool profile_limit_reached = false;
    size_t completed_profile_batches = 0;
    int cur = 0;
    int count = 0;
    int current_family = -1;

    auto finish_previous_batch = [&]() {
        cuda_require(cudaStreamSynchronize(prev_stream), "cudaStreamSynchronize(batch)");
        cuda_require(cudaGetLastError(), "CUDA batch completion");
        if (profile_batch_limit) {
            for (int i = 0; i < prev_count; ++i) {
                const auto& result = h_pinned[prev_buf][i];
                if (!std::isfinite(result.x_norm_sq) || !std::isfinite(result.v_norm_sq) ||
                    !std::isfinite(result.u_norm_sq) || !std::isfinite(result.L2_x) ||
                    !std::isfinite(result.L2_v) || result.x_norm_sq < 0 ||
                    result.v_norm_sq < 0 || result.u_norm_sq < 0
#if USE_LSODA
                    || result.solver_status != 2 || result.solver_time != double(TF)
#endif
                    ) {
                    std::fprintf(stderr, "PROFILE_RESULT_FAILED simulation=%d", i);
#if USE_LSODA
                    std::fprintf(stderr, " time=%.17g status=%d diagnostic=%d",
                                 result.solver_time, result.solver_status, result.solver_detail);
#endif
                    std::fprintf(stderr, "\n");
                    g_shutdown.store(true);
                    send_sim_done(lc, 0);
                    std::exit(EXIT_FAILURE);
                }
            }
        }
        const int published = publish_batch(lc, h_pinned[prev_buf], prev_count);
        if (profile_batch_limit && published == prev_count) {
            float milliseconds = 0;
            cuda_require(cudaEventElapsedTime(&milliseconds, kernel_begin[prev_buf],
                                              kernel_end[prev_buf]), "cudaEventElapsedTime");
            if (!std::isfinite(milliseconds) || milliseconds <= 0) {
                std::fprintf(stderr, "PROFILE_TIMING_FAILED milliseconds=%g\n", double(milliseconds));
                g_shutdown.store(true);
                send_sim_done(lc, 0);
                std::exit(EXIT_FAILURE);
            }
            double sums[5] = {};
            for (int i = 0; i < prev_count; ++i) {
                const auto& result = h_pinned[prev_buf][i];
                sums[0] += result.x_norm_sq;
                sums[1] += result.v_norm_sq;
                sums[2] += result.u_norm_sq;
                sums[3] += result.L2_x;
                sums[4] += result.L2_v;
            }
            std::printf("PROFILE_BATCH_OK batch=%zu simulations=%d kernel_event_seconds=%.9g "
                        "integral_sums=%.17g,%.17g,%.17g,%.17g,%.17g\n",
                        ++completed_profile_batches, prev_count, double(milliseconds) / 1000,
                        sums[0], sums[1], sums[2], sums[3], sums[4]);
        }
        return published;
    };

    auto launch_batch = [&]() {
        if (count == 0) return;
        cudaStream_t cur_stream = streams[cur];
        cuda_require(cudaMemcpyAsync(d_configs_aos[cur], h_configs[cur], count * sizeof(SimConfig),
                        cudaMemcpyHostToDevice, cur_stream), "cudaMemcpyAsync");
        int grid = (count + BLOCK - 1) / BLOCK;
        pack_configs_soa<<<grid, BLOCK, 0, cur_stream>>>(d_configs_aos[cur], d_configs[cur], count);
        cuda_require(cudaGetLastError(), "pack_configs_soa launch");
        // Reused GPU memory must not make an unwritten result look healthy.
        if (profile_batch_limit)
            cuda_require(cudaMemsetAsync(d_results[cur], 0xff, count * sizeof(SRGResult),
                                         cur_stream), "cudaMemsetAsync(result sentinel)");
        if (profile_batch_limit)
            cuda_require(cudaEventRecord(kernel_begin[cur], cur_stream), "cudaEventRecord(begin)");
        simulate_batch<<<grid, BLOCK, 0, cur_stream>>>(d_configs[cur], d_results[cur], count
#if USE_LSODA
            , d_lsoda_work[cur]
#endif
        );
        cuda_require(cudaGetLastError(), "simulate_batch launch");
        if (profile_batch_limit)
            cuda_require(cudaEventRecord(kernel_end[cur], cur_stream), "cudaEventRecord(end)");
        cuda_require(cudaMemcpyAsync(h_pinned[cur], d_results[cur], count * sizeof(SRGResult),
                        cudaMemcpyDeviceToHost, cur_stream), "cudaMemcpyAsync");
        // While the GPU runs this batch, publish the preceding one.
        if (prev_buf >= 0) {
            int published = finish_previous_batch();
            sims_published += published;
            printf("  [%zu/%zu] published %d (total %zu/%.0Lf)\n",
                   b, planned_batches, published, sims_published, n_total);
        }
        ++b;
        prev_buf = cur;
        prev_count = count;
        prev_stream = cur_stream;
        cur ^= 1;
        count = 0;
        if (profile_batch_limit && b >= profile_batch_limit)
            profile_limit_reached = true;
    };

    generate_waveform_sweep(basis, MAX_COMBO_SIZE, [&](const SimConfig& cfg, int family) {
        if (g_shutdown.load() || profile_limit_reached) return false;
        if (family != current_family) {
            launch_batch();
            if (g_shutdown.load() || profile_limit_reached) return false;
            current_family = family;
        }
        h_configs[cur][count++] = cfg;
        if (count == BATCH_SIZE) launch_batch();
        return !g_shutdown.load() && !profile_limit_reached;
    });
    if (!g_shutdown.load() && !profile_limit_reached) launch_batch();
    if (prev_buf >= 0) {
        int published = finish_previous_batch();
        sims_published += published;
        printf("  [%zu/%zu] published %d (total %zu/%.0Lf, final)\n",
               b, planned_batches, published, sims_published, n_total);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    RNum elapsed = std::chrono::duration<RNum>(t1 - t0).count();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));

    printf("\nDone: %zu/%.0Lf sims published in %.3f s\n",
           sims_published, n_total, elapsed);

    // ---- Cleanup ----
    cuda_require(cudaStreamSynchronize(streams[0]), "cudaStreamSynchronize");
    cuda_require(cudaStreamSynchronize(streams[1]), "cudaStreamSynchronize");
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    if (profile_batch_limit) {
        for (int i = 0; i < 2; ++i) {
            cuda_require(cudaEventDestroy(kernel_begin[i]), "cudaEventDestroy(begin)");
            cuda_require(cudaEventDestroy(kernel_end[i]), "cudaEventDestroy(end)");
        }
    }
    cudaFreeHost(h_pinned[0]);
    cudaFreeHost(h_pinned[1]);
    cudaFreeHost(h_configs[0]);
    cudaFreeHost(h_configs[1]);
    for (int i = 0; i < 2; ++i) {

        cudaFree(d_configs_aos[i]);

        cudaFree(d_configs[i].n_terms);
        cudaFree(d_configs[i].A);
        cudaFree(d_configs[i].B);
        cudaFree(d_configs[i].omega);
        cudaFree(d_configs[i].waveform);
        cudaFree(d_configs[i].max_abs_omega);

        cudaFree(d_results[i]);
#if USE_LSODA
        cudaFree(d_lsoda_work[i]);
#endif
    }

    if (profile_batch_limit && !g_shutdown.load())
        std::printf("PROFILE_COMPLETE batches=%zu simulations=%zu\n",
                    completed_profile_batches, sims_published);

    send_sim_done(lc, sims_published); // GPU work and final result batches are drained
#if USE_LSODA
    return g_lsoda_failed ? 1 : 0;
#else
    return 0;
#endif
}
