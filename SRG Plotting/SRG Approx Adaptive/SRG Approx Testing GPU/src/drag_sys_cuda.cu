/**
 * drag_sys_cuda.cu
 *
 * GPU-accelerated SRG sampling of a mass-spring-damper.
 * Each thread integrates the difference between a baseline and that
 * baseline plus a Fourier perturbation, with a zero-baseline fast path.
 * The host chooses the next adaptive batch from the accumulated boundary.
 * Run with ../run_drag_cuda.sh.
 */

#include <cstdio>
#include <cstdlib>
#include <charconv>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <algorithm>
#include <csignal>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string_view>
#include <thread>
#include <type_traits>

#include <cuda_runtime.h>
#include <lcm/lcm-cpp.hpp>
#include "lcm_messages/simulation_messages/SRGPlotBatch.hpp"
#include "lcm_messages/simulation_messages/startSim.hpp"
#include "lcm_messages/simulation_messages/SimFrictionInfo.hpp"
#include "lcm_messages/simulation_messages/SimDone.hpp"

#include "srg_sampler.hpp"

// Compile-time integrator selection: 0 = existing RKF7(8), 1 = LSODA.
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
// static constexpr double DEFAULT_R = 0.5 * K_D;
static constexpr RNum TF      = 211.0;
static constexpr RNum T_TRUNC = TF - 0.0;
#ifdef SRG_LINEAR_DRAG
static constexpr bool LINEAR_DRAG = SRG_LINEAR_DRAG != 0;
#else
static constexpr bool LINEAR_DRAG = true;
#endif

// Sliding friction F(v) = A * (1 + exp(-|v|/h)) * tanh(k*v).
// Enable with SRG_ENABLE_FRICTION=ON in the launcher, or edit this boolean.
#ifdef SRG_ENABLE_FRICTION
static constexpr bool ENABLE_FRICTION = SRG_ENABLE_FRICTION != 0;
#else
static constexpr bool ENABLE_FRICTION = false;
#endif
static constexpr double FRICTION_STRENGTH = 1.0;    // A, limiting sliding force
static constexpr double STRIBECK_SPEED = 1.0;       // h, exponential decay speed
static constexpr double FRICTION_TANH_SCALE = 10.0; // k, inverse smoothing speed
static_assert(FRICTION_STRENGTH >= 0 && STRIBECK_SPEED > 0 && FRICTION_TANH_SCALE > 0,
              "Friction strength must be nonnegative and both speed scales positive.");
static constexpr double DEFAULT_R = ENABLE_FRICTION
    ? std::min(0.5 * (K_D - FRICTION_STRENGTH / STRIBECK_SPEED),
               2.0 * K_P / (K_D + 2.0 * FRICTION_STRENGTH * FRICTION_TANH_SCALE))
    : 0.25 * K_D;
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

template<bool Paired>
constexpr int state_size = Paired ? 9 : 7;

// =============================================================
// Per-simulation configuration in GPU global memory
// =============================================================
struct SimConfigSoA {
    int*    n_terms; // banks for perturbation, then baseline (when paired)
    RNum* A;
    RNum* B;
    RNum* omega;
    unsigned char* waveform;
    RNum* max_abs_omega;

    int stride;
};

// Per-simulation output
struct SRGResult {
    double x_norm_sq;   // ∫ Δx² dt
    double v_norm_sq;   // ∫ Δv² dt
    double u_norm_sq;   // ∫ Δu² dt
    double L2_x;        // ∫ Δx·Δu dt
    double L2_v;        // ∫ Δv·Δu dt
    double vrx_norm_sq; // ∫ (Δv + r Δx)² dt
    double L2_vrx;      // ∫ (Δv + r Δx)·Δu dt
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

static void signal_handler(int /*sig*/) {
    g_shutdown.store(true);
}

// =============================================================
// Device code
// =============================================================
template<bool Paired>
__global__ void pack_configs_soa(
    const SimConfig* src,
    SimConfigSoA dst,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count)
        return;

    dst.max_abs_omega[idx] = src[idx].max_abs_omega;

    for (int signal = 0; signal < (Paired ? 2 : 1); ++signal) {
        const FourierInput& input = signal == 0 ? src[idx].delta : src[idx].base;
        dst.n_terms[signal * dst.stride + idx] = input.n_terms;
        #pragma unroll
        for (int i = 0; i < MAX_TERMS; ++i) {
            const int j = (signal * MAX_TERMS + i) * dst.stride + idx;
            dst.A[j] = input.A[i];
            dst.B[j] = input.B[i];
            dst.omega[j] = input.omega[i];
            dst.waveform[j] = input.waveform[i];
        }
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
    int signal,
    int n_terms,
    double t,
    const RNum* square_values)
{
    Real u = Real(0);
    for (int i = 0; i < n_terms; ++i) {
        const int term = signal * MAX_TERMS + i;
        const int config_idx = term * cfg.stride + idx;
        const Real omega = Real(cfg.omega[config_idx]);
        const Real A = Real(cfg.A[config_idx]);
        const Real B = Real(cfg.B[config_idx]);
        const auto waveform = cfg.waveform[config_idx];
        Real s = Real(0), c = Real(0);
        if (!square_values || (A != 0 && !(waveform & 1)) || (B != 0 && !(waveform & 2)))
            sincos_rnum(Real(double(omega) * t), &s, &c);
        // A and B independently choose smooth or sign waves.
        u += A * ((waveform & 1) ? (square_values ? Real(square_values[2*term]) : (c >= 0 ? Real(1) : Real(-1))) : c);
        u += B * ((waveform & 2) ? (square_values ? Real(square_values[2*term+1]) : (s >= 0 ? Real(1) : Real(-1))) : s);
    }
    return u;
}

template<class Real>
__device__ Real friction_force(Real velocity)
{
    const Real envelope = exp(-fabs(velocity) / Real(STRIBECK_SPEED));
    return Real(FRICTION_STRENGTH) * (Real(1) + envelope) *
        tanh(Real(FRICTION_TANH_SCALE) * velocity);
}

template<class Real>
__device__ Real friction_slope(Real velocity)
{
    const Real speed = fabs(velocity);
    const Real envelope = exp(-speed / Real(STRIBECK_SPEED));
    const Real q = exp(-Real(2) * Real(FRICTION_TANH_SCALE) * speed);
    const Real sech_sq = Real(4) * q / ((Real(1) + q) * (Real(1) + q));
    const Real weakening = envelope * fabs(tanh(Real(FRICTION_TANH_SCALE) * velocity))
        / Real(STRIBECK_SPEED);
    return Real(FRICTION_STRENGTH) *
        ((Real(1) + envelope) * Real(FRICTION_TANH_SCALE) * sech_sq - weakening);
}

template<class Real>
__device__ Real friction_difference(Real base, Real difference)
{
    const Real driven = base + difference;
    if ((base >= 0) != (driven >= 0))
        return friction_force(driven) - friction_force(base);

    // Factor same-sign differences to retain small perturbations on a large
    // baseline. Every expm1 argument is nonpositive, including near saturation.
    const Real sign = base >= 0 ? Real(1) : Real(-1);
    const Real speed_difference = sign * difference;
    const Real base_envelope = exp(-fabs(base) / Real(STRIBECK_SPEED));
    const Real driven_envelope = exp(-fabs(driven) / Real(STRIBECK_SPEED));
    const Real envelope_difference = speed_difference >= 0
        ? base_envelope * expm1(-speed_difference / Real(STRIBECK_SPEED))
        : -driven_envelope * expm1(speed_difference / Real(STRIBECK_SPEED));
    // For q = exp(-2*k*|v|), tanh(k*v) = sign(v)*(1-q)/(1+q).
    const Real base_q = exp(-Real(2) * Real(FRICTION_TANH_SCALE) * fabs(base));
    const Real driven_q = exp(-Real(2) * Real(FRICTION_TANH_SCALE) * fabs(driven));
    const Real q_difference = speed_difference >= 0
        ? base_q * expm1(-Real(2) * Real(FRICTION_TANH_SCALE) * speed_difference)
        : -driven_q * expm1(Real(2) * Real(FRICTION_TANH_SCALE) * speed_difference);
    const Real tanh_difference = -sign * Real(2) * q_difference /
        ((Real(1) + base_q) * (Real(1) + driven_q));
    const Real base_tanh = tanh(Real(FRICTION_TANH_SCALE) * base);
    return Real(FRICTION_STRENGTH) *
        ((Real(1) + driven_envelope) * tanh_difference + base_tanh * envelope_difference);
}

template<class Real>
__device__ Real acceleration(Real position, Real velocity, Real input)
{
    const Real damping = LINEAR_DRAG ? Real(K_D) : Real(K_D) * Real(0.5) * fabs(velocity);
    Real value = input - Real(K_P) * position - damping * velocity;
    if constexpr (ENABLE_FRICTION) value -= friction_force(velocity);
    return value;
}

// State layout: [integral dx², dv², du², dx, dv, integral dx*du,
// dv*du, x_base, v_base]. The last two states are omitted for zero pairing.
template<bool Paired>
struct SimRhs {
    SimConfigSoA configs;
    const RNum* square_values;
    int idx;
    int delta_terms;
    int base_terms;
    bool input_active;
#if USE_LSODA && LSODA_CACHE_SQUARE_FORCING
    bool input_is_constant = false;
    double held_delta = 0;
    double held_base = 0;
#endif

    template<class Real>
    __device__ Real input(int signal, double t) const
    {
#if USE_LSODA && LSODA_CACHE_SQUARE_FORCING
        if (input_is_constant) return Real(signal == 0 ? held_delta : held_base);
#endif
        const int count = signal == 0 ? delta_terms : base_terms;
        return input_active ? fourier_sum_dev<Real>(configs, idx, signal, count, t, square_values) : Real(0);
    }

    template<class Real>
    __device__ void evaluate(double t, const Real* x, Real* xdot) const
    {
        const Real du = input<Real>(0, t);
        const Real dx = x[3], dv = x[4];
        xdot[0] = dx * dx;
        xdot[1] = dv * dv;
        xdot[2] = du * du;
        xdot[3] = dv;
        xdot[5] = dx * du;
        xdot[6] = dv * du;
        if constexpr (Paired) {
            const Real base_v = x[8], driven_v = base_v + dv;
            xdot[7] = base_v;
            xdot[8] = acceleration(x[7], base_v, input<Real>(1, t));
            Real damping_difference;
            if constexpr (LINEAR_DRAG) {
                damping_difference = Real(K_D) * dv;
            } else {
                const Real abs_base = fabs(base_v), abs_driven = fabs(driven_v);
                // Factor the same-sign difference to preserve tiny perturbations
                // on top of a large baseline. Opposite signs do not cancel.
                const Real quadratic_difference = (base_v >= 0) == (driven_v >= 0)
                    ? dv * (abs_base + abs_driven)
                    : driven_v * abs_driven - base_v * abs_base;
                damping_difference = Real(0.5) * Real(K_D) * quadratic_difference;
            }
            if constexpr (ENABLE_FRICTION)
                damping_difference += friction_difference(base_v, dv);
            xdot[4] = du - Real(K_P) * dx - damping_difference;
        } else {
            xdot[4] = acceleration(dx, dv, du);
        }
    }

#if USE_LSODA
    __device__ void operator()(int*, double* t, double* y, double* ydot, double*) const {
        evaluate(*t, y, ydot);
    }
#endif
};

#if USE_LSODA
#if LSODA_USE_ANALYTIC_JACOBIAN
// DLSODA clears the dense buffer before calling JAC. Entries are column-major.
template<bool Paired>
struct LsodaJacobian {
    SimRhs<Paired> rhs;

    __device__ void operator()(int*, double* t, double* y, int, int,
                               double* jacobian, int leading_dimension) const
    {
        const double du = rhs.template input<double>(0, *t);
        const double base_v = Paired ? y[8] : 0;
        const double driven_v = base_v + y[4];
        jacobian[3 * leading_dimension + 0] = 2.0 * y[3];
        jacobian[4 * leading_dimension + 1] = 2.0 * y[4];
        jacobian[4 * leading_dimension + 3] = 1.0;
        jacobian[3 * leading_dimension + 4] = -double(K_P);
        jacobian[4 * leading_dimension + 4] = LINEAR_DRAG
            ? -double(K_D) : -double(K_D) * fabs(driven_v);
        if constexpr (ENABLE_FRICTION)
            jacobian[4 * leading_dimension + 4] -= friction_slope(driven_v);
        jacobian[3 * leading_dimension + 5] = du;
        jacobian[4 * leading_dimension + 6] = du;
        if constexpr (Paired) {
            jacobian[8 * leading_dimension + 4] = 0;
            jacobian[8 * leading_dimension + 7] = 1.0;
            jacobian[7 * leading_dimension + 8] = -double(K_P);
            jacobian[8 * leading_dimension + 8] = LINEAR_DRAG
                ? -double(K_D) : -double(K_D) * fabs(base_v);
            if constexpr (ENABLE_FRICTION) {
                jacobian[8 * leading_dimension + 8] -= friction_slope(base_v);
                jacobian[8 * leading_dimension + 4] =
                    -(friction_slope(driven_v) - friction_slope(base_v));
            }
            if constexpr (!LINEAR_DRAG) {
                const double abs_difference = (base_v >= 0) == (driven_v >= 0)
                    ? (base_v >= 0 ? y[4] : -y[4])
                    : fabs(driven_v) - fabs(base_v);
                jacobian[8 * leading_dimension + 4] -= double(K_D) * abs_difference;
            }
        }
    }
};
#endif
#else
/** Fixed-step RKF7(8).  All scratch in thread-local registers. */
template<bool Paired>
__device__ void rkf78_step_dev(
    double t,
    double h,
    const RNum* x,
    RNum* x_next,
    const SimRhs<Paired>& rhs)
{
    const RNum step = RNum(h);
    constexpr int N = state_size<Paired>;
    RNum k[13][N];
    RNum xtmp[N];

    // Stage 0
    rhs.evaluate(t, x, k[0]);

    // Stage 1
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(2.0/27.0)*k[0][i]);
    rhs.evaluate(t + h*2.0/27.0, xtmp, k[1]);

    // Stage 2
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(1.0/36.0)*k[0][i] + RNum(1.0/12.0)*k[1][i]);
    rhs.evaluate(t + h/9.0, xtmp, k[2]);

    // Stage 3
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(1.0/24.0)*k[0][i] + RNum(1.0/8.0)*k[2][i]);
    rhs.evaluate(t + h/6.0, xtmp, k[3]);

    // Stage 4
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(5.0/12.0)*k[0][i] - RNum(25.0/16.0)*k[2][i]
                             + RNum(25.0/16.0)*k[3][i]);
    rhs.evaluate(t + h*5.0/12.0, xtmp, k[4]);

    // Stage 5
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(1.0/20.0)*k[0][i] + RNum(1.0/4.0)*k[3][i]
                             + RNum(1.0/5.0)*k[4][i]);
    rhs.evaluate(t + h*0.5, xtmp, k[5]);

    // Stage 6
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(-RNum(25.0/108.0)*k[0][i] + RNum(125.0/108.0)*k[3][i]
                             - RNum(65.0/27.0)*k[4][i] + RNum(125.0/54.0)*k[5][i]);
    rhs.evaluate(t + h*5.0/6.0, xtmp, k[6]);

    // Stage 7
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(31.0/300.0)*k[0][i] + RNum(61.0/225.0)*k[4][i]
                             - RNum(2.0/9.0)*k[5][i] + RNum(13.0/900.0)*k[6][i]);
    rhs.evaluate(t + h/6.0, xtmp, k[7]);

    // Stage 8
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(2.0)*k[0][i] - RNum(53.0/6.0)*k[3][i]
                             + RNum(704.0/45.0)*k[4][i] - RNum(107.0/9.0)*k[5][i]
                             + RNum(67.0/90.0)*k[6][i] + RNum(3.0)*k[7][i]);
    rhs.evaluate(t + h*2.0/3.0, xtmp, k[8]);

    // Stage 9
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(-RNum(91.0/108.0)*k[0][i] + RNum(23.0/108.0)*k[3][i]
                             - RNum(976.0/135.0)*k[4][i] + RNum(311.0/54.0)*k[5][i]
                             - RNum(19.0/60.0)*k[6][i] + RNum(17.0/6.0)*k[7][i]
                             - RNum(1.0/12.0)*k[8][i]);
    rhs.evaluate(t + h/3.0, xtmp, k[9]);

    // Stage 10
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(2383.0/4100.0)*k[0][i] - RNum(341.0/164.0)*k[3][i]
                             + RNum(4496.0/1025.0)*k[4][i] - RNum(301.0/82.0)*k[5][i]
                             + RNum(2133.0/4100.0)*k[6][i] + RNum(45.0/82.0)*k[7][i]
                             + RNum(45.0/164.0)*k[8][i] + RNum(18.0/41.0)*k[9][i]);
    rhs.evaluate(t + h, xtmp, k[10]);

    // Stage 11
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(RNum(3.0/205.0)*k[0][i] - RNum(6.0/41.0)*k[5][i]
                             - RNum(3.0/205.0)*k[6][i] - RNum(3.0/41.0)*k[7][i]
                             + RNum(3.0/41.0)*k[8][i] + RNum(6.0/41.0)*k[9][i]);
    rhs.evaluate(t, xtmp, k[11]);

    // Stage 12
    for (int i = 0; i < N; ++i)
        xtmp[i] = x[i] + step*(-RNum(1777.0/4100.0)*k[0][i] - RNum(341.0/164.0)*k[3][i]
                             + RNum(4496.0/1025.0)*k[4][i] - RNum(289.0/82.0)*k[5][i]
                             + RNum(2193.0/4100.0)*k[6][i] + RNum(51.0/82.0)*k[7][i]
                             + RNum(33.0/164.0)*k[8][i] + RNum(12.0/41.0)*k[9][i]
                             + k[11][i]);
    rhs.evaluate(t + h, xtmp, k[12]);

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
// GPU kernel with one thread per simulation
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

template<bool Paired>
#if USE_LSODA
// LSODA has a larger call stack and workspace than the RK kernel.
__global__ __launch_bounds__(128)
#else
__global__ __launch_bounds__(128, 4)
#endif
void simulate_batch(
    SimConfigSoA configs,
    SRGResult* results,
    int count,
    double r
#if USE_LSODA
    , srg_lsoda::Workspace<state_size<Paired>>* lsoda_work
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
    if constexpr (ENABLE_FRICTION && FRICTION_STRENGTH > 0) {
        // Limit the explicit step for a steep regularised friction law.
        constexpr double slope_bound = FRICTION_STRENGTH *
            (2 * FRICTION_TANH_SCALE + 1 / STRIBECK_SPEED);
        dt = fmin(dt, RNum(0.5 / slope_bound));
    }
    #endif

#if USE_LSODA
    double state[state_size<Paired>] = {};
    int solver_status = 2;
    int solver_detail = 0;
    // Adaptive multistep methods must restart at forcing discontinuities.
    constexpr bool split_jumps = true;
#else
    RNum state[state_size<Paired>] = {};
    RNum state_next[state_size<Paired>];
    constexpr bool split_jumps = SPLIT_SQUARE_JUMPS;
#endif
    constexpr int slots = MAX_TERMS * (Paired ? 2 : 1);
    RNum square_values[2 * slots];
    double next_cos[slots], next_sin[slots];
    const int delta_terms = configs.n_terms[idx];
    const int base_terms = Paired ? configs.n_terms[configs.stride + idx] : 0;
    const int n_terms = delta_terms + base_terms;
#if USE_LSODA && LSODA_CACHE_SQUARE_FORCING
    bool square_only = true;
#endif
    for (int i = 0; i < n_terms; ++i) {
        const int term = i < delta_terms ? i : MAX_TERMS + i - delta_terms;
        const int j = term * configs.stride + idx;
#if USE_LSODA && LSODA_CACHE_SQUARE_FORCING
        if ((configs.A[j] != 0 && !(configs.waveform[j] & 1)) ||
            (configs.B[j] != 0 && !(configs.waveform[j] & 2)))
            square_only = false;
#endif
        square_values[2*term] = RNum(1);
        square_values[2*term+1] = configs.omega[j] >= 0 ? RNum(1) : RNum(-1);
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
        SimRhs<Paired> rhs{configs, split_jumps ? square_values : nullptr,
                           idx, delta_terms, base_terms, input_active};
#if USE_LSODA
#if LSODA_CACHE_SQUARE_FORCING
        if (input_active && square_only) {
            rhs.held_delta = rhs.template input<double>(0, t);
            if constexpr (Paired) rhs.held_base = rhs.template input<double>(1, t);
        }
        rhs.input_is_constant = !input_active || square_only;
#endif
        solver_status = srg_lsoda::integrate<state_size<Paired>>(
            rhs, state, t, end, double(dt), LSODA_RTOL, LSODA_ATOL,
            LSODA_MAX_STEPS, lsoda_work[idx], solver_detail
#if LSODA_USE_ANALYTIC_JACOBIAN
            , LsodaJacobian<Paired>{rhs}
#endif
            );
        if (solver_status != 2) break;
#else
        rkf78_step_dev(t, end - t, state, state_next, rhs);
        for (int i = 0; i < state_size<Paired>; ++i) state[i] = state_next[i];
#endif
        t = end;
        if (input_active && split_jumps) {
            double tolerance = 16.0 * 2.2204460492503131e-16 * (1.0 + t);
            for (int i = 0; i < n_terms; ++i) {
                const int term = i < delta_terms ? i : MAX_TERMS + i - delta_terms;
                const int j = term * configs.stride + idx;
                if (next_cos[i] <= t + tolerance) {
                    square_values[2*term] = -square_values[2*term];
                    next_cos[i] = next_square_jump(t, configs.omega[j], true);
                }
                if (next_sin[i] <= t + tolerance) {
                    square_values[2*term+1] = -square_values[2*term+1];
                    next_sin[i] = next_square_jump(t, configs.omega[j], false);
                }
            }
        }
    }
    const double x_norm_sq = state[0], v_norm_sq = state[1];
    const double final_x = state[3];
    // The initial displacement difference is zero and Δv = d(Δx)/dt.
    // Thus 2 * integral(Δx * Δv) = Δx(T)^2, including for paired inputs.
    const double vrx_norm_sq = fma(r, fma(r, x_norm_sq, final_x * final_x), v_norm_sq);
    results[idx].x_norm_sq = x_norm_sq;
    results[idx].v_norm_sq = v_norm_sq;
    results[idx].u_norm_sq = state[2];
    results[idx].L2_x = state[5];
    results[idx].L2_v = state[6];
    // A negative value can arise from integration roundoff for r < 0.
    results[idx].vrx_norm_sq = vrx_norm_sq < 0 ? 0 : vrx_norm_sq;
    results[idx].L2_vrx = fma(r, double(state[5]), double(state[6]));
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
    cfg.delta.n_terms = k;
    RNum wmax = 0.0;
    for (int i = 0; i < k; ++i) {
        const auto& bt = basis[indices[i]];
        cfg.delta.A[i]     = bt.A;
        cfg.delta.B[i]     = bt.B;
        cfg.delta.omega[i] = bt.omega;
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
                components += (cfg.delta.A[i] != 0) + (cfg.delta.B[i] != 0);
            unsigned int all_square = (1u << components) - 1;
            auto emit_mask = [&](unsigned int mask) {
                int component = 0;
                for (int i = 0; i < k; ++i) {
                    cfg.delta.waveform[i] = 0;
                    if (cfg.delta.A[i] != 0 && ((mask >> component++) & 1u)) cfg.delta.waveform[i] |= 1;
                    if (cfg.delta.B[i] != 0 && ((mask >> component++) & 1u)) cfg.delta.waveform[i] |= 2;
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

class SimStartHandler {
public:
    SimStartHandler(lcm::LCM* lc, double r, bool ideal_only) : lc_(lc)
    {
        info_.k_d = K_D;
        info_.k_p = K_P;
        info_.r = r;
        info_.linear_drag = ideal_only || LINEAR_DRAG;
        info_.ideal_only = ideal_only;
        info_.friction_enabled = ENABLE_FRICTION;
        info_.friction_strength = FRICTION_STRENGTH;
        info_.stribeck_speed = STRIBECK_SPEED;
        info_.tanh_scale = FRICTION_TANH_SCALE;
    }

    bool ready = false;

    void handleMessage(const lcm::ReceiveBuffer*, const std::string&,
                       const simulation_messages::startSim* msg)
    {
        if (lc_->publish("SimFrictionInfo", &info_) == 0 && !ready) {
            printf("Plotter connected (%s).\n", msg->sim_name.c_str());
            ready = true;
        }
    }

private:
    lcm::LCM* lc_;
    simulation_messages::SimFrictionInfo info_;
};

static bool completed_result(const SRGResult& result)
{
#if USE_LSODA
    if (result.solver_status != 2 || result.solver_time != double(TF)) return false;
#endif
    return std::isfinite(result.x_norm_sq) && std::isfinite(result.v_norm_sq) &&
           std::isfinite(result.u_norm_sq) && std::isfinite(result.L2_x) &&
           std::isfinite(result.L2_v) && std::isfinite(result.vrx_norm_sq) &&
           std::isfinite(result.L2_vrx) && result.x_norm_sq >= 0 &&
           result.v_norm_sq >= 0 && result.u_norm_sq >= 0 && result.vrx_norm_sq >= 0;
}

static void report_bad_result(const char* label, int index, const SRGResult& result)
{
    std::fprintf(stderr, "%s simulation=%d", label, index);
#if USE_LSODA
    std::fprintf(stderr, " time=%.17g status=%d diagnostic=%d",
                 result.solver_time, result.solver_status, result.solver_detail);
#endif
    std::fprintf(stderr, " norm_sq=(%.9g,%.9g,%.9g,%.9g) products=(%.9g,%.9g,%.9g)\n",
                 result.x_norm_sq, result.v_norm_sq, result.vrx_norm_sq, result.u_norm_sq,
                 result.L2_x, result.L2_v, result.L2_vrx);
}

static void publish_batch(lcm::LCM* lc, const SRGResult* results, int count)
{
    if (!lc) return;
    constexpr int MAX_LCM_BATCH = 512;

    for (int offset = 0; offset < count; offset += MAX_LCM_BATCH) {

        const int batch_count =
            std::min(MAX_LCM_BATCH, count - offset);

        simulation_messages::SRGPlotBatch msg;

        msg.n = batch_count;

        msg.x_norm.resize(batch_count);
        msg.v_norm.resize(batch_count);
        msg.vrx_norm.resize(batch_count);
        msg.u_norm.resize(batch_count);
        msg.L2_x.resize(batch_count);
        msg.L2_v.resize(batch_count);
        msg.L2_vrx.resize(batch_count);

        for (int i = 0; i < batch_count; ++i) {

            const SRGResult& result = results[offset + i];

            msg.x_norm[i] = sqrt(result.x_norm_sq);
            msg.v_norm[i] = sqrt(result.v_norm_sq);
            msg.vrx_norm[i] = sqrt(result.vrx_norm_sq);
            msg.u_norm[i] = sqrt(result.u_norm_sq);
            msg.L2_x[i]   = result.L2_x;
            msg.L2_v[i]   = result.L2_v;
            msg.L2_vrx[i] = result.L2_vrx;
        }

        lc->publish("SRGPlotBatch", &msg);
    }
}

// =============================================================
// Main
// =============================================================
static void send_sim_done(lcm::LCM* lc, size_t n_simulations)
{
    if (!lc) return;
    simulation_messages::SimDone msg;
    msg.n_simulations = static_cast<int64_t>(n_simulations);
    msg.cancelled = g_shutdown.load();
    lc->publish("SimDone", &msg);
}

enum class Sampling { adaptive, random, sweep };
enum class Pairing { automatic, zero, paired };

struct RunOptions {
    Sampling sampling = Sampling::adaptive;
    Pairing pairing = Pairing::automatic;
    srg::Geometry geometry = srg::Geometry::hconvex;
    srg::Target target = srg::Target::both;
    std::size_t samples = 0; // use the default budget unless specified
    int batch_size = 0; // use the mode's default
    std::uint64_t seed = 42;
    double r = DEFAULT_R;
    bool plot = true;
    bool ideal_only = false;
};

static constexpr int SWEEP_BATCH_SIZE = 81920;
static constexpr int ADAPTIVE_BATCH_SIZE = 2048;

static void usage(FILE* out, const char* executable)
{
    std::fprintf(out,
        "Usage: %s [--sampler adaptive|random|sweep] [--samples N]\n"
        "       [--batch-size N] [--seed N] [--pairing auto|zero|paired]\n"
        "       [--target x|v|vrx|both|all] [--mode hconvex|convex|raw]\n"
        "       [--r VALUE (default 2*K_D)] [--no-plot | --ideal-only]\n",
        executable);
}

static bool parse_unsigned(std::string_view value, std::uint64_t& number)
{
    if (value.empty()) return false;
    const auto end = value.data() + value.size();
    const auto result = std::from_chars(value.data(), end, number);
    return result.ec == std::errc{} && result.ptr == end;
}

static bool parse_finite_double(std::string_view value, double& number)
{
    if (value.empty()) return false;
    const auto end = value.data() + value.size();
    const auto result = std::from_chars(value.data(), end, number);
    return result.ec == std::errc{} && result.ptr == end && std::isfinite(number);
}

static bool parse_options(int argc, char** argv, RunOptions& options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--no-plot") {
            options.plot = false;
            continue;
        }
        if (arg == "--ideal-only") {
            options.ideal_only = true;
            continue;
        }
        const auto equals = arg.find('=');
        const auto name = arg.substr(0, equals);
        if (equals == std::string_view::npos && i + 1 == argc) return false;
        const std::string_view value = equals == std::string_view::npos ? argv[++i] : arg.substr(equals + 1);
        std::uint64_t number = 0;
        if (name == "--sampler" && value == "adaptive") options.sampling = Sampling::adaptive;
        else if (name == "--sampler" && value == "random") options.sampling = Sampling::random;
        else if (name == "--sampler" && value == "sweep") options.sampling = Sampling::sweep;
        else if (name == "--pairing" && value == "auto") options.pairing = Pairing::automatic;
        else if (name == "--pairing" && value == "zero") options.pairing = Pairing::zero;
        else if (name == "--pairing" && value == "paired") options.pairing = Pairing::paired;
        else if (name == "--target" && value == "x") options.target = srg::Target::position;
        else if (name == "--target" && value == "v") options.target = srg::Target::velocity;
        else if (name == "--target" && value == "vrx") options.target = srg::Target::combined;
        else if (name == "--target" && value == "both") options.target = srg::Target::both;
        else if (name == "--target" && value == "all") options.target = srg::Target::all;
        else if (name == "--mode" && value == "hconvex") options.geometry = srg::Geometry::hconvex;
        else if (name == "--mode" && (value == "convex" || value == "raw"))
            options.geometry = srg::Geometry::convex;
        else if (name == "--seed" && parse_unsigned(value, number)) options.seed = number;
        else if (name == "--r") {
            if (!parse_finite_double(value, options.r)) {
                std::fprintf(stderr, "--r requires a finite number.\n");
                return false;
            }
        }
        else if (name == "--samples" && parse_unsigned(value, number) && number > 0 && number <= 2147483647)
            options.samples = number;
        else if (name == "--batch-size" && parse_unsigned(value, number) && number > 0 && number <= SWEEP_BATCH_SIZE)
            options.batch_size = int(number);
        else {
            std::fprintf(stderr, "Invalid option or value near %s\n", argv[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::strcmp(argv[1], "--profile-support") == 0) {
        std::puts("SRG_PROFILE_BOUNDED_V2");
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        usage(stdout, argv[0]);
        return 0;
    }
    RunOptions options;
    if (!parse_options(argc, argv, options)) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (options.ideal_only && !options.plot) {
        std::fprintf(stderr, "--ideal-only requires the plotter. Use run_drag_cuda.sh --ideal-only.\n");
        return 2;
    }
    // The profiler retains the original deterministic sweep and kernel sizes.
    size_t profile_batch_limit = 0;
    if (const char* value = std::getenv("SRG_PROFILE_BATCHES"); value && *value) {
        std::uint64_t parsed = 0;
        if (!parse_unsigned(value, parsed) || parsed == 0 || parsed > 2147483647) {
            std::fprintf(stderr, "SRG_PROFILE_BATCHES must be an integer from 1 to 2147483647.\n");
            return 2;
        }
        profile_batch_limit = size_t(parsed);
        options.sampling = Sampling::sweep;
    }
    int profile_batch_size = SWEEP_BATCH_SIZE;
    if (profile_batch_limit) {
        if (const char* value = std::getenv("SRG_PROFILE_BATCH_SIZE"); value && *value) {
            std::uint64_t parsed = 0;
            if (!parse_unsigned(value, parsed) || parsed == 0 || parsed > SWEEP_BATCH_SIZE) {
                std::fprintf(stderr, "SRG_PROFILE_BATCH_SIZE must be an integer from 1 to %d.\n",
                             SWEEP_BATCH_SIZE);
                return 2;
            }
            profile_batch_size = int(parsed);
        }
    }
    const bool sweep = options.sampling == Sampling::sweep;
    if (sweep && options.pairing == Pairing::paired) {
        std::fprintf(stderr, "Paired inputs require --sampler adaptive or random.\n");
        return 2;
    }
    if (sweep && options.samples > 0) {
        std::fprintf(stderr, "--samples requires --sampler adaptive or random.\n");
        return 2;
    }
    if (options.samples == 0) options.samples = 32768;
    const bool paired = !sweep && (options.pairing == Pairing::paired ||
                                   (options.pairing == Pairing::automatic &&
                                    (!LINEAR_DRAG || ENABLE_FRICTION)));
    const int default_batch_size = sweep ? SWEEP_BATCH_SIZE : ADAPTIVE_BATCH_SIZE;
    const int requested_batch_size = options.batch_size > 0 ? options.batch_size : default_batch_size;
    const int BATCH_SIZE = profile_batch_limit ? profile_batch_size : requested_batch_size;
    // Preserve configuration messages when the profiler stops the process.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    // ---- Install signal handler for clean Ctrl+C ----
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    // ---- LCM setup ----
    std::unique_ptr<lcm::LCM> connection;
    if (options.plot) {
        const char* url = std::getenv("SRG_LCM_URL");
        connection = std::make_unique<lcm::LCM>(url && *url ? url :
            "udpm://239.255.76.67:7667?ttl=0&recv_buf_size=8388608");
    }
    lcm::LCM* lc = connection.get();
    if (lc && !lc->good()) {
        std::fprintf(stderr, "Could not open the LCM plot connection. Use --no-plot to run CUDA alone.\n");
        return 2;
    }

    size_t sims_published = 0;
    auto cuda_require = [&](cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            std::fprintf(stderr, "CUDA_ERROR operation=%s code=%d name=%s message=%s\n",
                         operation, int(status), cudaGetErrorName(status), cudaGetErrorString(status));
            g_shutdown.store(true);
            send_sim_done(lc, sims_published);
            std::exit(EXIT_FAILURE);
        }
    };

    SimStartHandler handler(lc, options.r, options.ideal_only);
    if (lc) {
        lc->subscribe("startSim", &SimStartHandler::handleMessage, &handler);
        printf("Waiting for the plotter to connect over LCM...\n");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!handler.ready && !g_shutdown.load()) {
            if (lc->handleTimeout(100) < 0 && !g_shutdown.load()) {
                std::fprintf(stderr, "LCM failed while waiting for the plotter.\n");
                return 2;
            }
            if (!handler.ready && std::chrono::steady_clock::now() >= deadline) {
                std::fprintf(stderr, "No plotter connected within 60 seconds. Use run_drag_cuda.sh to start both processes, or pass --no-plot.\n");
                return 2;
            }
        }
    }
    if (g_shutdown.load()) {
        if (handler.ready) send_sim_done(lc, 0);
        return 0;
    }
    printf("Combined output v + r*x with r=%.12g\n", options.r);
    printf("Sliding friction %s, strength=%.12g, Stribeck speed=%.12g, tanh scale=%.12g\n",
           ENABLE_FRICTION ? "enabled" : "disabled",
           FRICTION_STRENGTH, STRIBECK_SPEED, FRICTION_TANH_SCALE);
    if (options.ideal_only) {
        printf("Exporting the analytical linear SRGs.\n");
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
        std::printf("PROFILE_BATCH_SIZE simulations=%d normal=%d\n", BATCH_SIZE, SWEEP_BATCH_SIZE);
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

    // The adaptive and random modes share a continuous domain spanning the
    // sweep's amplitude and frequency bounds. The sweep keeps its old grid.
    const srg::Domain domain{0.01, 3333.33, OMEGA_MIN, OMEGA_MAX,
                             std::sqrt(std::max(0.0, double(K_P)))};
    srg::Sampler sampler(domain, options.geometry, options.target, paired, options.seed);

    // ============================================================
    // Build all simulation configs
    // ============================================================
    std::vector<BasisTerm> basis;
    long double n_total = options.samples;
    size_t n_batches = (options.samples - 1) / BATCH_SIZE + 1;
    if (sweep) {
        basis = build_basis_terms(A_vals, B_vals, omega_vals);
        printf("Basis terms: %zu\n", basis.size());
        // A term contributes 2^m waveform assignments, with m nonzero coefficients.
        long double combinations[MAX_COMBO_SIZE + 1] = {1};
        long double assignments[MAX_COMBO_SIZE + 1] = {1};
        for (const auto& term : basis) {
            const int m = (term.A != 0) + (term.B != 0);
            for (int k = MAX_COMBO_SIZE; k > 0; --k) {
                combinations[k] += combinations[k-1];
                assignments[k] += (1u << m) * assignments[k-1];
            }
        }
        long double single_family = 0, mixed = 0;
        for (int k = 1; k <= MAX_COMBO_SIZE; ++k) {
            single_family += combinations[k];
            mixed += assignments[k] - 2 * combinations[k];
        }
        n_total = 2 * single_family + mixed;
        n_batches = 2 * size_t(std::ceil(single_family / BATCH_SIZE))
            + size_t(std::ceil(mixed / BATCH_SIZE));
        printf("Square: %.0Lf, sine: %.0Lf, mixed: %.0Lf; total: %.0Lf\n",
               single_family, single_family, mixed, n_total);
    } else {
        const char* target = "both";
        switch (options.target) {
            case srg::Target::position: target = "x"; break;
            case srg::Target::velocity: target = "v"; break;
            case srg::Target::combined: target = "vrx"; break;
            case srg::Target::all: target = "all"; break;
            case srg::Target::both: break;
        }
        printf("Sampler: %s, pairing=%s, target=%s, geometry=%s, seed=%llu\n",
               options.sampling == Sampling::adaptive ? "adaptive" : "random",
               paired ? "paired" : "zero", target,
               options.geometry == srg::Geometry::hconvex ? "hconvex" : "convex",
               static_cast<unsigned long long>(options.seed));
    }

    if (profile_batch_limit > n_batches) {
        std::fprintf(stderr, "Requested %zu profile batches, but this sweep has only %zu.\n",
                     profile_batch_limit, n_batches);
        g_shutdown.store(true);
        send_sim_done(lc, 0);
        return 2;
    }
    const size_t planned_batches = profile_batch_limit ? profile_batch_limit : n_batches;
    const int banks = paired ? 2 : 1;
    const int term_slots = banks * MAX_TERMS;
    printf("Allocating CUDA buffers for batches of up to %d trials...\n", BATCH_SIZE);

    // GPU buffers retain their original double-buffered layout.
    // AoS staging buffer used only for H->D transfer
    SimConfig* d_configs_aos[2];

    // SoA representation used by the simulation kernel
    SimConfigSoA d_configs[2];

    SRGResult* d_results[2];
#if USE_LSODA
    // Separate persistent scratch for every simulation and both CUDA streams.
    void* d_lsoda_work[2] = {nullptr, nullptr};
    const size_t workspace_bytes = BATCH_SIZE * (paired
        ? sizeof(srg_lsoda::Workspace<9>) : sizeof(srg_lsoda::Workspace<7>));
    printf("LSODA workspaces: %.1f MiB for both buffers\n",
           2.0 * workspace_bytes / (1024.0 * 1024.0));
    for (int i = 0; i < 2; ++i) {
        cudaError_t allocation = cudaMalloc(&d_lsoda_work[i], workspace_bytes);
        if (allocation != cudaSuccess) {
            fprintf(stderr, "LSODA workspace allocation failed: %s. Reduce --batch-size.\n",
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
            banks * BATCH_SIZE * sizeof(int)
        ), "cudaMalloc");

        cuda_require(cudaMalloc(
            &d_configs[i].max_abs_omega,
            BATCH_SIZE * sizeof(RNum)
        ), "cudaMalloc");

        cuda_require(cudaMalloc(
            &d_configs[i].A,
            term_slots * BATCH_SIZE * sizeof(RNum)
        ), "cudaMalloc");

        cuda_require(cudaMalloc(
            &d_configs[i].B,
            term_slots * BATCH_SIZE * sizeof(RNum)
        ), "cudaMalloc");

        cuda_require(cudaMalloc(
            &d_configs[i].omega,
            term_slots * BATCH_SIZE * sizeof(RNum)
        ), "cudaMalloc");
        cuda_require(cudaMalloc(&d_configs[i].waveform,
                   term_slots * BATCH_SIZE * sizeof(unsigned char)), "cudaMalloc");

        d_configs[i].stride = BATCH_SIZE;

        cuda_require(cudaMalloc(
            &d_results[i],
            BATCH_SIZE * sizeof(SRGResult)
        ), "cudaMalloc");
    }

    // Pinned host buffers for the asynchronous copies.
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
    size_t sims_evaluated = 0;
    size_t batches_launched = 0;
    size_t completed_profile_batches = 0;
    bool simulation_failed = false;
    int cur = 0;
    int count = 0;

    auto finish_previous_batch = [&]() {
        if (prev_buf < 0) return;
        const auto wait_started = std::chrono::steady_clock::now();
        auto next_update = wait_started + std::chrono::seconds(5);
        cudaError_t status;
        while ((status = cudaStreamQuery(prev_stream)) == cudaErrorNotReady) {
            // A new startSim may be a retry after dropped plot settings.
            if (lc) lc->handleTimeout(0);
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_update) {
                printf("Waiting for GPU batch %zu of %zu (%d trials, %.0f seconds elapsed)...\n",
                       batches_launched, planned_batches, prev_count,
                       std::chrono::duration<double>(now - wait_started).count());
                next_update = now + std::chrono::seconds(10);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        cuda_require(status, "cudaStreamQuery(batch)");
        cuda_require(cudaGetLastError(), "CUDA batch completion");
        if (lc) lc->handleTimeout(0);
        SRGResult* results = h_pinned[prev_buf];
        int accepted = prev_count;
        sims_evaluated += prev_count;
        if (simulation_failed) {
            prev_buf = -1;
            return;
        }
        if (profile_batch_limit || (sweep && USE_LSODA)) {
            for (int i = 0; i < prev_count; ++i) {
                if (!completed_result(results[i])) {
                    report_bad_result(profile_batch_limit ? "PROFILE_RESULT_FAILED"
                                                         : "LSODA batch rejected", i, results[i]);
                    simulation_failed = true;
                    g_shutdown.store(true);
                    prev_buf = -1;
                    return;
                }
            }
        }
        if (!sweep) {
            accepted = 0;
            bool logged_rejection = false;
            for (int i = 0; i < prev_count; ++i) {
                const auto& result = results[i];
                if (!completed_result(result) || result.u_norm_sq == 0) {
                    if (!logged_rejection) {
                        report_bad_result("Discarded SRG result", i, result);
                        logged_rejection = true;
                    }
                    continue;
                }
                const auto& config = h_configs[prev_buf][i];
                sampler.position.add(srg::from_integrals(result.x_norm_sq, result.u_norm_sq, result.L2_x), config);
                sampler.velocity.add(srg::from_integrals(result.v_norm_sq, result.u_norm_sq, result.L2_v), config);
                sampler.combined.add(srg::from_integrals(result.vrx_norm_sq, result.u_norm_sq, result.L2_vrx), config);
                results[accepted++] = result;
            }
            sampler.update();
            if (accepted == 0) {
                std::fprintf(stderr, "No usable simulations in this batch. Check the solver and input bounds.\n");
                simulation_failed = true;
                g_shutdown.store(true);
            }
        }
        if (profile_batch_limit) {
            float milliseconds = 0;
            cuda_require(cudaEventElapsedTime(&milliseconds, kernel_begin[prev_buf],
                                              kernel_end[prev_buf]), "cudaEventElapsedTime");
            if (!std::isfinite(milliseconds) || milliseconds <= 0) {
                std::fprintf(stderr, "PROFILE_TIMING_FAILED milliseconds=%g\n", double(milliseconds));
                simulation_failed = true;
                g_shutdown.store(true);
                prev_buf = -1;
                return;
            }
            double sums[5] = {};
            for (int i = 0; i < prev_count; ++i) {
                sums[0] += results[i].x_norm_sq;
                sums[1] += results[i].v_norm_sq;
                sums[2] += results[i].u_norm_sq;
                sums[3] += results[i].L2_x;
                sums[4] += results[i].L2_v;
            }
            std::printf("PROFILE_BATCH_OK batch=%zu simulations=%d kernel_event_seconds=%.9g "
                        "integral_sums=%.17g,%.17g,%.17g,%.17g,%.17g\n",
                        ++completed_profile_batches, prev_count, double(milliseconds) / 1000,
                        sums[0], sums[1], sums[2], sums[3], sums[4]);
        }
        publish_batch(lc, results, accepted);
        sims_published += accepted;
        if (sweep) {
            printf("  [%zu/%zu] published %d (total %zu/%.0Lf)\n",
                   batches_launched, planned_batches, accepted, sims_published, n_total);
        } else {
            printf("SRG_PROGRESS batch=%zu evaluated=%zu accepted=%zu rejected=%zu "
                   "area_x=%.12g area_v=%.12g area_vrx=%.12g "
                   "vertices_x=%zu vertices_v=%zu vertices_vrx=%zu\n",
                   batches_launched, sims_evaluated, sims_published, sims_evaluated - sims_published,
                   2 * sampler.position.area(), 2 * sampler.velocity.area(), 2 * sampler.combined.area(),
                   sampler.position.size(), sampler.velocity.size(), sampler.combined.size());
        }
        prev_buf = -1;
    };

    auto launch_kernel = [&](auto pairing_tag, cudaStream_t stream) {
        constexpr bool is_paired = decltype(pairing_tag)::value;
        const int grid = (count + BLOCK - 1) / BLOCK;
        pack_configs_soa<is_paired><<<grid, BLOCK, 0, stream>>>(d_configs_aos[cur], d_configs[cur], count);
        cuda_require(cudaGetLastError(), "pack_configs_soa launch");
        // Reused GPU memory must not make an unwritten profiling result look healthy.
        if (profile_batch_limit) {
            cuda_require(cudaMemsetAsync(d_results[cur], 0xff, count * sizeof(SRGResult),
                                         stream), "cudaMemsetAsync(result sentinel)");
            cuda_require(cudaEventRecord(kernel_begin[cur], stream), "cudaEventRecord(begin)");
        }
        simulate_batch<is_paired><<<grid, BLOCK, 0, stream>>>(d_configs[cur], d_results[cur], count, options.r
#if USE_LSODA
            , static_cast<srg_lsoda::Workspace<state_size<is_paired>>*>(d_lsoda_work[cur])
#endif
        );
        cuda_require(cudaGetLastError(), "simulate_batch launch");
        if (profile_batch_limit)
            cuda_require(cudaEventRecord(kernel_end[cur], stream), "cudaEventRecord(end)");
    };

    auto launch_batch = [&]() {
        if (count == 0) return;
        if (batches_launched == 0)
            printf("Starting GPU batch 1 of %zu (%d trials). Results arrive after the batch finishes.\n",
                   planned_batches, count);
        const cudaStream_t stream = streams[cur];
        cuda_require(cudaMemcpyAsync(d_configs_aos[cur], h_configs[cur], count * sizeof(SimConfig),
                        cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");
        if (paired) launch_kernel(std::true_type{}, stream);
        else launch_kernel(std::false_type{}, stream);
        cuda_require(cudaMemcpyAsync(h_pinned[cur], d_results[cur], count * sizeof(SRGResult),
                        cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
        // The exhaustive sweep can overlap publication with the next launch.
        finish_previous_batch();
        ++batches_launched;
        prev_buf = cur;
        prev_count = count;
        prev_stream = stream;
        cur ^= 1;
        count = 0;
    };

    if (sweep) {
        int current_family = -1;
        auto running = [&]() {
            return !g_shutdown.load() && (!profile_batch_limit || batches_launched < profile_batch_limit);
        };
        generate_waveform_sweep(basis, MAX_COMBO_SIZE, [&](const SimConfig& cfg, int family) {
            if (!running()) return false;
            if (family != current_family) {
                launch_batch();
                if (!running()) return false;
                current_family = family;
            }
            h_configs[cur][count++] = cfg;
            if (count == BATCH_SIZE) launch_batch();
            return running();
        });
        if (running()) launch_batch();
        finish_previous_batch();
    } else {
        const bool adaptive = options.sampling == Sampling::adaptive;
        while (!g_shutdown.load() && sims_evaluated < options.samples) {
            count = static_cast<int>(std::min<std::size_t>(BATCH_SIZE, options.samples - sims_evaluated));
            for (int i = 0; i < count; ++i)
                h_configs[cur][i] = sampler.next(sims_evaluated + i, adaptive);
            // Adjacent GPU threads then have similar frequency-dependent work.
            std::stable_sort(h_configs[cur], h_configs[cur] + count, [](const SimConfig& a, const SimConfig& b) {
                return a.max_abs_omega < b.max_abs_omega;
            });
            if (g_shutdown.load()) break;
            launch_batch();
            finish_previous_batch(); // feed the new hull back before proposing more
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    RNum elapsed = std::chrono::duration<RNum>(t1 - t0).count();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));

    printf("\nDone: %zu accepted, %zu evaluated of %.0Lf planned in %.3f s\n",
           sims_published, sims_evaluated, n_total, elapsed);

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
    return simulation_failed ? 1 : 0;
}
