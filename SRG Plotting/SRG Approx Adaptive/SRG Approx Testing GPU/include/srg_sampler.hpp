#pragma once

#include "srg_inputs.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace srg {

inline constexpr double pi = 3.14159265358979323846;

enum class Geometry { convex, hconvex };
enum class Target { position, velocity, combined, both, all };

struct Point {
    double x;
    double y;
};

inline double cross(Point a, Point b, Point c)
{
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// The Python plotter uses the same Beltrami-Klein projection to find the
// hyperbolic hull. Area is measured back in the plotted complex plane.
inline Point project(Point z, Geometry geometry)
{
    if (geometry == Geometry::convex) return z;
    const double norm_sq = z.x * z.x + z.y * z.y;
    return {(norm_sq - 1) / (norm_sq + 1), -2 * z.x / (norm_sq + 1)};
}

inline Point from_integrals(double yy, double uu, double yu)
{
    if (yy == 0) return {0, 0};
    const double output_norm = std::sqrt(yy), input_norm = std::sqrt(uu);
    const double cosine = std::clamp(yu / (output_norm * input_norm), -1.0, 1.0);
    const double gain = output_norm / input_norm;
    return {gain * cosine, gain * std::sqrt(std::max(0.0, 1 - cosine * cosine))};
}

// Signed area between an oriented upper-half-plane geodesic and its chord.
// A vertical geodesic is straight. For a circular arc, compute its segment
// from the chord length and central angle, avoiding a distant circle centre.
inline double arc_area(Point a, Point b)
{
    const double dx = b.x - a.x;
    if (dx == 0) return 0;
    const double dy = b.y - a.y;
    const double half_angle = std::atan2(std::abs(dx), a.y + b.y);
    const double angle = 2 * half_angle;
    const double sine = std::sin(half_angle);
    const double factor = angle < 0.01
        ? angle / 12 * (1 + angle * angle / 30)
        : (angle - std::sin(angle)) / (8 * sine * sine);
    return std::copysign((dx * dx + dy * dy) * factor, -dx);
}

inline double triangle_area(Point a, Point b, Point c, Geometry geometry)
{
    double area = cross(a, b, c) / 2;
    if (geometry == Geometry::hconvex)
        area += arc_area(a, b) + arc_area(b, c) + arc_area(c, a);
    return std::max(0.0, area);
}

class Boundary {
public:
    explicit Boundary(Geometry geometry) : geometry_(geometry) {}

    void add(Point z, const SimConfig& config)
    {
        pending_.push_back({z, project(z, geometry_), config});
    }

    void update()
    {
        if (pending_.empty()) return;
        // Only previous vertices can survive the hull of the enlarged set.
        pending_.insert(pending_.end(), vertices_.begin(), vertices_.end());
        std::stable_sort(pending_.begin(), pending_.end(), [](const Vertex& a, const Vertex& b) {
            return a.projected.x < b.projected.x ||
                   (a.projected.x == b.projected.x && a.projected.y < b.projected.y);
        });
        const auto last = std::unique(pending_.begin(), pending_.end(), [](const Vertex& a, const Vertex& b) {
            return a.projected.x == b.projected.x && a.projected.y == b.projected.y;
        });
        pending_.erase(last, pending_.end());

        vertices_.clear();
        if (pending_.size() <= 2) {
            vertices_.swap(pending_);
        } else {
            // Monotone chain gives the counterclockwise hull, also for a
            // collinear batch. Keep the waveform associated with each vertex.
            auto append = [&](const Vertex& point, std::size_t begin) {
                while (vertices_.size() >= begin + 2 &&
                       cross(vertices_[vertices_.size() - 2].projected,
                             vertices_.back().projected, point.projected) <= 0)
                    vertices_.pop_back();
                vertices_.push_back(point);
            };
            for (const auto& vertex : pending_) append(vertex, 0);
            const std::size_t lower_size = vertices_.size();
            for (std::size_t i = pending_.size() - 1; i-- > 0;)
                append(pending_[i], lower_size - 1);
            vertices_.pop_back(); // the first vertex is repeated at the end
        }
        pending_.clear();
        update_weights();
    }

    bool empty() const { return vertices_.empty(); }
    std::size_t size() const { return vertices_.size(); }
    double area() const { return area_; }

    const SimConfig& parent(std::mt19937_64& random)
    {
        // Uniform boundary selection retains coverage when a few large ears
        // would otherwise dominate. Each output has its own weights and units.
        const auto index = std::bernoulli_distribution(0.25)(random)
            ? std::uniform_int_distribution<std::size_t>(0, vertices_.size() - 1)(random)
            : area_weights_(random);
        return vertices_[index].config;
    }

private:
    struct Vertex {
        Point z;
        Point projected;
        SimConfig config;
    };

    void update_weights()
    {
        area_ = 0;
        const auto count = vertices_.size();
        std::vector<double> weights(count, 0);
        if (count >= 3) {
            for (std::size_t i = 1; i + 1 < count; ++i)
                area_ += triangle_area(vertices_[0].z, vertices_[i].z, vertices_[i + 1].z, geometry_);
            for (std::size_t i = 0; i < count; ++i) {
                // Area lost by replacing the vertex's two incident edges
                // with a single edge between its neighbours.
                weights[i] = triangle_area(vertices_[(i + count - 1) % count].z,
                                           vertices_[i].z, vertices_[(i + 1) % count].z, geometry_);
            }
        }
        if (std::none_of(weights.begin(), weights.end(), [](double weight) { return weight > 0; }))
            std::fill(weights.begin(), weights.end(), 1.0);
        area_weights_ = std::discrete_distribution<std::size_t>(weights.begin(), weights.end());
    }

    Geometry geometry_;
    std::vector<Vertex> vertices_, pending_;
    std::discrete_distribution<std::size_t> area_weights_;
    double area_ = 0;
};

struct Domain {
    double min_amplitude;
    double max_amplitude;
    double min_omega;
    double max_omega;
    double seed_omega;
};

class Sampler {
public:
    Sampler(Domain domain, Geometry geometry, Target target, bool paired, std::uint64_t seed)
        : position(geometry), velocity(geometry), combined(geometry), domain_(domain), target_(target),
          paired_(paired), exploration_(seed), local_(seed ^ 0x9e3779b97f4a7c15ULL) {}

    SimConfig next(std::size_t index, bool adaptive)
    {
        // Advance the global RNG on every post-seed trial in both modes.
        // Exploration therefore starts from the same proposals in A/B runs.
        SimConfig config = index < seed_count ? seed(index) : explore();
        if (adaptive && index >= seed_count && index % 4 != 0) {
            Boundary& boundary = target_boundary(index);
            if (!boundary.empty()) {
                for (int attempt = 0; attempt < 4; ++attempt) {
                    auto child = boundary.parent(local_);
                    mutate(child);
                    if (remember(child)) return with_max_frequency(child);
                }
            }
        }
        while (!remember(config)) config = explore();
        return with_max_frequency(config);
    }

    void update()
    {
        position.update();
        velocity.update();
        combined.update();
    }

    Boundary position;
    Boundary velocity;
    Boundary combined;

private:
    Boundary& target_boundary(std::size_t index)
    {
        if (target_ == Target::all) {
            // The three slots following each global proposal alternate outputs.
            const auto slot = index % 4;
            return slot == 1 ? position : slot == 2 ? velocity : combined;
        }
        if (target_ == Target::combined) return combined;
        if (target_ == Target::position ||
            (target_ == Target::both && (index / 4 + index % 4) % 2 == 0))
            return position;
        return velocity;
    }

    static constexpr int frequency_count = 12;
    static constexpr int seed_count = 3 * 3 * frequency_count;

    using WaveformKey = std::array<std::array<RNum, 4>, MAX_TERMS>;
    static WaveformKey waveform_key(const FourierInput& input)
    {
        WaveformKey key{};
        for (int i = 0; i < input.n_terms; ++i)
            key[i] = {input.omega[i], input.A[i], input.B[i], RNum(input.waveform[i])};
        // Term order has no meaning in a Fourier sum.
        std::sort(key.begin(), key.begin() + input.n_terms);
        return key;
    }

    bool remember(const SimConfig& config)
    {
        return seen_.insert({waveform_key(config.delta), waveform_key(config.base)}).second;
    }

    static SimConfig with_max_frequency(SimConfig config)
    {
        config.max_abs_omega = 0;
        for (const auto* input : {&config.delta, &config.base})
            for (int i = 0; i < input->n_terms; ++i)
                config.max_abs_omega = std::max(config.max_abs_omega, std::abs(input->omega[i]));
        return config;
    }

    struct Term {
        double amplitude;
        double phase;
        double omega;
        unsigned char waveform;
    };

    static double log_mix(double low, double high, double fraction)
    {
        return std::exp(std::log(low) + fraction * std::log(high / low));
    }

    static double unit(std::mt19937_64& random)
    {
        return std::uniform_real_distribution<double>(0, 1)(random);
    }

    static int choose(std::mt19937_64& random, int count)
    {
        return std::uniform_int_distribution<int>(0, count - 1)(random);
    }

    static void set_term(FourierInput& input, int i, Term term)
    {
        input.A[i] = RNum(term.amplitude * std::cos(term.phase));
        input.B[i] = RNum(term.amplitude * std::sin(term.phase));
        input.omega[i] = RNum(term.omega);
        input.waveform[i] = term.waveform;
    }

    Term draw_term(std::mt19937_64& random) const
    {
        return {log_mix(domain_.min_amplitude, domain_.max_amplitude, unit(random)),
                2 * pi * unit(random), log_mix(domain_.min_omega, domain_.max_omega, unit(random)),
                static_cast<unsigned char>(choose(random, 4))};
    }

    SimConfig seed(std::size_t index) const
    {
        const int frequency = index % frequency_count;
        const int amplitude = (index / frequency_count) % 3;
        const int shape = index / (3 * frequency_count);
        const double omega = frequency == 0
            ? std::clamp(domain_.seed_omega, domain_.min_omega, domain_.max_omega)
            : log_mix(domain_.min_omega, domain_.max_omega,
                      double(frequency - 1) / (frequency_count - 2));
        const double level = log_mix(domain_.min_amplitude, domain_.max_amplitude, amplitude / 2.0);
        SimConfig config{};
        config.delta.n_terms = 1;
        config.delta.omega[0] = RNum(omega);
        if (shape == 1) {
            config.delta.B[0] = RNum(level);
        } else {
            config.delta.A[0] = RNum(level);
            config.delta.waveform[0] = shape == 2 ? 1 : 0;
        }
        if (paired_ && index % 4 == 3) {
            config.base.n_terms = 1;
            set_term(config.base, 0, {log_mix(domain_.min_amplitude, domain_.max_amplitude, 0.5),
                                      pi / 4, omega, 0});
        }
        return config;
    }

    FourierInput draw_input(std::mt19937_64& random) const
    {
        FourierInput input{};
        input.n_terms = unit(random) < 0.6 ? 1 : 1 + choose(random, MAX_TERMS);
        for (int i = 0; i < input.n_terms; ++i) set_term(input, i, draw_term(random));
        return input;
    }

    SimConfig explore()
    {
        SimConfig config{};
        config.delta = draw_input(exploration_);
        if (paired_ && unit(exploration_) < 0.75) {
            config.base = draw_input(exploration_);
            if (unit(exploration_) < 0.5) config.base.omega[0] = config.delta.omega[0];
        }
        return config;
    }

    void mutate(SimConfig& config)
    {
        const bool use_base = paired_ && unit(local_) < 0.3;
        auto& input = use_base ? config.base : config.delta;
        if (input.n_terms == 0) {
            input = draw_input(local_);
            if (unit(local_) < 0.5) input.omega[0] = config.delta.omega[0];
            return;
        }
        const double action = unit(local_);
        const int i = choose(local_, input.n_terms);
        if (action < 0.08 && input.n_terms < MAX_TERMS) {
            set_term(input, input.n_terms, draw_term(local_));
            ++input.n_terms;
        } else if (action < 0.16 && (input.n_terms > 1 || use_base)) {
            --input.n_terms;
            input.A[i] = input.A[input.n_terms];
            input.B[i] = input.B[input.n_terms];
            input.omega[i] = input.omega[input.n_terms];
            input.waveform[i] = input.waveform[input.n_terms];
        } else if (action < 0.23) {
            const unsigned char active = (input.A[i] != 0) | ((input.B[i] != 0) << 1);
            input.waveform[i] ^= active == 3 ? 1u << choose(local_, 2) : active;
        } else {
            // Mostly local steps with a mix of scales for refining edges
            // and for moving between nearby response regimes.
            const double scale = unit(local_) < 0.7 ? 0.12 : 0.6;
            const double step = std::normal_distribution<double>(0, scale)(local_);
            const double factor = std::exp(step);
            if (action < 0.55) {
                input.omega[i] = RNum(std::clamp(input.omega[i] * factor, domain_.min_omega, domain_.max_omega));
            } else if (action < 0.83) {
                const double amplitude = std::hypot(double(input.A[i]), double(input.B[i]));
                const double level = std::clamp(amplitude * factor, domain_.min_amplitude, domain_.max_amplitude);
                input.A[i] = RNum(input.A[i] * (level / amplitude));
                input.B[i] = RNum(input.B[i] * (level / amplitude));
            } else {
                const double amplitude = std::hypot(double(input.A[i]), double(input.B[i]));
                const double phase = std::atan2(double(input.B[i]), double(input.A[i]));
                set_term(input, i, {amplitude, phase + pi * step, input.omega[i], input.waveform[i]});
            }
        }
    }

    Domain domain_;
    Target target_;
    bool paired_;
    std::mt19937_64 exploration_, local_;
    std::set<std::array<WaveformKey, 2>> seen_;
};

} // namespace srg
