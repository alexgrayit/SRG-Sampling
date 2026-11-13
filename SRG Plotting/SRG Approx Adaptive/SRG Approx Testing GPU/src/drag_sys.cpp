#include <eigen3/Eigen/Dense>
#include "include/integrators/RK/RK78.h"
#include <math.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <atomic>
#include <csignal>
#include <new>
#include <iostream>
#include <chrono>
#include <thread>
#include <lcm/lcm-cpp.hpp>
#include "include/drag_core.h"

#include "lcm_messages/simulation_messages/SRGState.hpp"
#include "lcm_messages/simulation_messages/startSim.hpp"
#include "lcm_messages/simulation_messages/SRGBatch.hpp"
#include "lcm_messages/simulation_messages/SimInfo.hpp"
#include "lcm_messages/simulation_messages/SimDone.hpp"

lcm::LCM lc("udpm://239.255.76.67:7667?ttl=0&recv_buf_size=8388608");

bool start_flag = false;

class SimStartHandler
{
public:
    SimStartHandler() {}
    ~SimStartHandler() {}
    void handleMessage(const lcm::ReceiveBuffer* rbuf, const std::string& chan, const simulation_messages::startSim* msg)
    {
        std::cout << "Received startSim message: " << msg->sim_name << std::endl;
        start_flag = true;
    }

};


// CPU uses double by default; select RNum in the existing RK78.h header.
constexpr RNum k_d = RNum(1.0); // Drag coefficient
constexpr RNum k_p = RNum(1.0); // Spring coefficient
constexpr bool LINEAR_DRAG = true;
constexpr bool SPLIT_SQUARE_JUMPS = true;
constexpr dragParams params = {k_d};

constexpr RNum dt_max = RNum(0.001);
constexpr double tf = 201.0;
constexpr double T = tf; // Truncate the input at T, then let the system decay.
constexpr int MAX_TERMS = 6;
const int N = MAX_TERMS;

struct SimConfig {
    int n_terms;
    RNum A[MAX_TERMS], B[MAX_TERMS], omega[MAX_TERMS];
    unsigned char waveform[MAX_TERMS]; // bit 0: square cosine, bit 1: square sine
    RNum max_abs_omega;
};

Eigen::Vector<RNum, N> As = Eigen::Vector<RNum, N>::Zero();
Eigen::Vector<RNum, N> Bs = Eigen::Vector<RNum, N>::Zero();
Eigen::Vector<RNum, N> omegas;
int n_terms = 0;
unsigned char waveform[MAX_TERMS];
RNum square_values[2 * MAX_TERMS];
bool input_active = true;

// State: integrals of x^2, v^2, u^2, then x, v, and integrals of x*u, v*u.
Eigen::Vector<RNum, 7> state;
Eigen::Vector<RNum, 7> state_0 = Eigen::Vector<RNum, 7>::Zero();
simulation_messages::SRGBatch batch;
constexpr int BATCH_SIZE = 512;
static std::atomic<bool> g_shutdown{false};

static void signal_handler(int) { g_shutdown.store(true); }

RNum drag(RNum v, RNum u, const RNum k) {
    return LINEAR_DRAG ? -k * v : -RNum(0.5) * k * v * std::abs(v);
}

RNum spring(RNum x, RNum u, const RNum k)
{
    return -k * x;
}

RNum fourier_sum(const Eigen::Vector<RNum, N>& As,
                 const Eigen::Vector<RNum, N>& Bs,
                 const Eigen::Vector<RNum, N>& omega, double t)
{
    RNum u = RNum(0);
    for (int i = 0; i < n_terms; ++i) {
        RNum c = RNum(0), s = RNum(0);
        if ((As(i) != 0 && !(waveform[i] & 1)) ||
            (Bs(i) != 0 && !(waveform[i] & 2)) || !SPLIT_SQUARE_JUMPS) {
            RNum phase = RNum(double(omega(i)) * t);
            c = std::cos(phase);
            s = std::sin(phase);
        }
        // A square wave is sign(cos) or sign(sin), with the same amplitude.
        // https://en.wikipedia.org/wiki/Square_wave
        u += As(i) * ((waveform[i] & 1) ? (SPLIT_SQUARE_JUMPS ? square_values[2*i] : (c >= 0 ? RNum(1) : RNum(-1))) : c);
        u += Bs(i) * ((waveform[i] & 2) ? (SPLIT_SQUARE_JUMPS ? square_values[2*i+1] : (s >= 0 ? RNum(1) : RNum(-1))) : s);
    }
    return u;
}

Eigen::Vector<RNum, 7> state_derivative(const Eigen::Vector<RNum, 7>& state, double t) {
    RNum x = state(3);
    RNum v = state(4);
    RNum u = input_active ? fourier_sum(As, Bs, omegas, t) : RNum(0);
    RNum dxdt = v;
    RNum dvdt = u + drag(v, u, k_d) + spring(x, u, k_p);
    Eigen::Vector<RNum, 7> derivatives;
    derivatives << x*x, v*v, u*u, dxdt, dvdt, x*u, v*u;
    return derivatives;
}

void rk_state_derivative(double t, RNum* x, RNum* x_dot, void* params)
{
    Eigen::Vector<RNum, 7> state;
    for (int i = 0; i < 7; ++i) state(i) = x[i];
    Eigen::Vector<RNum, 7> derivatives = state_derivative(state, t);
    for (int i = 0; i < 7; ++i) x_dot[i] = derivatives(i);
}

void publish_batch()
{
    if (batch.n == 0) return;
    lc.publish("SRGBatch", &batch);
    batch.n = 0;
    batch.x_norm.clear(); batch.v_norm.clear(); batch.u_norm.clear();
    batch.L2_x.clear(); batch.L2_v.clear();
}

void send_srg_state()
{
    // Batch the same L2 statistics as the GPU, rather than send each separately.
    batch.x_norm.push_back(std::sqrt(state(0)));
    batch.v_norm.push_back(std::sqrt(state(1)));
    batch.u_norm.push_back(std::sqrt(state(2)));
    batch.L2_x.push_back(state(5));
    batch.L2_v.push_back(state(6));
    if (++batch.n == BATCH_SIZE) publish_batch();
}

void do_rk_step(double t, double dt)
{
    RNum x[7], x_next[7], err_est[7];
    for (int i = 0; i < 7; ++i) x[i] = state(i);
    int status = rkf78_step(t, dt, x, 7, (void*)&params, rk_state_derivative, x_next, err_est);
    if (status != 0) throw std::bad_alloc();
    for (int i = 0; i < 7; ++i) state(i) = x_next[i];
}

// Sine jumps are n*pi/|omega|; cosine jumps are (n+1/2)*pi/|omega|.
static double next_square_jump(double t, RNum omega, bool cosine) {
    double w = std::abs(double(omega));
    if (w == 0) return INFINITY;
    double period = M_PI / w;
    double offset = cosine ? 0.5 : 0.0;
    double n = floor(t / period - offset) + 1.0;
    double next = (n + offset) * period;
    if (next <= t + 16.0 * 2.2204460492503131e-16 * (1.0 + t))
        next = (++n + offset) * period;
    return next;
}

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

void send_sim_done(size_t n_simulations)
{
    simulation_messages::SimDone msg;
    msg.n_simulations = static_cast<int64_t>(n_simulations);
    msg.cancelled = g_shutdown.load();
    lc.publish("SimDone", &msg);
}

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    if (!lc.good()) {
        std::cerr << "LCM initialization failed" << std::endl;
        return 1;
    }

    std::cout << "Waiting for startSim message..." << std::endl;
    SimStartHandler sim_start_handler;
    lc.subscribe("startSim", &SimStartHandler::handleMessage, &sim_start_handler);
    // Even an early cancellation replies after Python has subscribed.
    while (!start_flag) lc.handleTimeout(100);
    simulation_messages::SimInfo info;
    info.k_d = k_d;
    info.k_p = k_p;
    info.linear_drag = LINEAR_DRAG;
    lc.publish("SimInfo", &info); // one reply before any result batches
    if (g_shutdown.load()) {
        send_sim_done(0);
        return 0;
    }
    batch.n = 0;

    // The same candidates and complete square/smooth/mixed passes as the GPU.
    // Cosine amplitudes to try
    std::vector<RNum> A_vals = { 0.0, 1.0 };

    // Sine amplitudes to try
    std::vector<RNum> B_vals = { 0.5, 1.0 , 10.0};

    // Frequencies to try (rad/s)
    constexpr int    N_omega   = 100;
    constexpr RNum omega_max = 8.0 * 2.0 * M_PI;
    constexpr RNum omega_min = 0.002 * 2.0 * M_PI;
    const RNum omega_0  = omega_min;
    const RNum n_omega  = (RNum)N_omega / log(omega_max / omega_min);

    // Two simulations per frequency index (positive & negative omega)
    const int n_omega_total = N_omega * 2;
    std::vector<RNum> omega_vals(n_omega_total);

    for (int n = 0; n < N_omega; ++n) {
        RNum w = omega_0 * exp((RNum)n / n_omega);
        omega_vals[2*n + 0] =  w;
        omega_vals[2*n + 1] = -w;
    }

    // Max number of terms to combine
    // 1 = singles only, 2 = singles + pairs, 3 = + trios, ...
    // Must be <= MAX_TERMS (currently 6)
    constexpr int MAX_COMBO_SIZE = 2;
    static_assert(MAX_COMBO_SIZE <= MAX_TERMS,
                  "MAX_COMBO_SIZE cannot exceed MAX_TERMS");

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

    std::cout << "Starting simulation..." << std::endl;
    size_t sims_published = 0;
    int current_family = -1;
    generate_waveform_sweep(basis, MAX_COMBO_SIZE, [&](const SimConfig& cfg, int family) {
        if (g_shutdown.load()) return false;
        if (family != current_family) {
            publish_batch();
            current_family = family;
            std::cout << "Starting " << (family == 0 ? "square" : family == 1 ? "sine" : "mixed") << " sums" << std::endl;
        }
        n_terms = cfg.n_terms;
        double next_cos[MAX_TERMS], next_sin[MAX_TERMS];
        for (int i = 0; i < n_terms; ++i) {
            As(i) = cfg.A[i]; Bs(i) = cfg.B[i]; omegas(i) = cfg.omega[i];
            waveform[i] = cfg.waveform[i];
            square_values[2*i] = RNum(1);
            square_values[2*i+1] = omegas(i) >= 0 ? RNum(1) : RNum(-1);
            next_cos[i] = As(i) != 0 && (waveform[i] & 1)
                ? next_square_jump(0, omegas(i), true) : INFINITY;
            next_sin[i] = Bs(i) != 0 && (waveform[i] & 2)
                ? next_square_jump(0, omegas(i), false) : INFINITY;
        }
        RNum f = cfg.max_abs_omega / RNum(2.0 * M_PI);
        RNum time_scale = k_d > 0 ? k_d : RNum(1);
        RNum dt = f < RNum(1e-12) ? RNum(0.005) / time_scale
            : RNum(0.005) / time_scale / (RNum(6) * f);
        dt = std::min(dt, dt_max);
        state = state_0;
        double t = 0;
        while (t < tf && !g_shutdown.load()) {
            input_active = t < T;
            double end = std::min(t + double(dt), tf);
            if (input_active) end = std::min(end, T);
            if (input_active && SPLIT_SQUARE_JUMPS) {
                for (int i = 0; i < n_terms; ++i) {
                    end = std::min(end, next_cos[i]);
                    end = std::min(end, next_sin[i]);
                }
            }
            // Keep each square constant through the RK endpoints, then flip.
            // No integration step crosses a jump, or omits the final interval.
            do_rk_step(t, end - t);
            t = end;
            if (input_active && SPLIT_SQUARE_JUMPS) {
                double tolerance = 16.0 * 2.2204460492503131e-16 * (1.0 + t);
                for (int i = 0; i < n_terms; ++i) {
                    if (next_cos[i] <= t + tolerance) {
                        square_values[2*i] = -square_values[2*i];
                        next_cos[i] = next_square_jump(t, omegas(i), true);
                    }
                    if (next_sin[i] <= t + tolerance) {
                        square_values[2*i+1] = -square_values[2*i+1];
                        next_sin[i] = next_square_jump(t, omegas(i), false);
                    }
                }
            }
        }
        if (g_shutdown.load()) return false; // do not publish a partial experiment
        send_srg_state();
        if (++sims_published % BATCH_SIZE == 0)
            std::cout << "Completed " << sims_published << " simulations" << std::endl;
        return true;
    });
    publish_batch(); // include the final completed partial batch
    send_sim_done(sims_published); // completion follows all result messages
    std::cout << "Simulation " << (g_shutdown.load() ? "stopped" : "complete") << "." << std::endl;
    return 0;
}
