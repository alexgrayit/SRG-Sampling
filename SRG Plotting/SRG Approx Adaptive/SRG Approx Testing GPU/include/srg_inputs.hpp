#pragma once

#ifndef RNum
#define RNum float
#endif

// The sweep uses this many terms in total. The adaptive sampler can use
// this many in each of the perturbation and baseline signals.
inline constexpr int MAX_COMBO_SIZE = 3;
inline constexpr int MAX_TERMS = MAX_COMBO_SIZE;
static_assert(MAX_TERMS >= 1 && MAX_TERMS <= 6);

struct FourierInput {
    int n_terms;
    RNum A[MAX_TERMS];
    RNum B[MAX_TERMS];
    RNum omega[MAX_TERMS];
    unsigned char waveform[MAX_TERMS]; // bit 0: square cosine, bit 1: square sine
};

struct SimConfig {
    FourierInput delta;
    FourierInput base;
    RNum max_abs_omega;
};
