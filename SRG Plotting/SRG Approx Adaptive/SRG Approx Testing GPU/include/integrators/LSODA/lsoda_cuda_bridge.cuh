#pragma once
#include <cmath>
#include <type_traits>

// This file is the SRG adapter, not a copy of the external LSODA library.
// TLAJ/lsoda-cuda, inspected at commit 543f92ecec0890830b547e4dcac2783afc21b93b.
// Its header includes cuLsoda.cu: do not compile that file separately.
#if !__has_include("integrators/lsoda-cuda/Model/gpu/cuLsoda.cu.h")
#error "Copy the bundled include/integrators/lsoda-cuda directory, or set USE_LSODA to 0."
#else
#include "integrators/lsoda-cuda/Model/gpu/cuLsoda.cu.h"
#endif

// The upstream implementation leaves common-block aliases (including n, l,
// min and max) defined. Do not let them rewrite the simulator's identifiers.
#undef min
#undef max
#undef conit
#undef crate
#undef ccmax
#undef el0
#undef h__
#undef hmin
#undef hmxi
#undef hu
#undef rc
#undef tn
#undef uround
#undef pdest
#undef pdlast
#undef ratio
#undef hold
#undef rmax
#undef el
#undef elco
#undef tesco
#undef rls
#undef tsw
#undef pdnorm
#undef cm1
#undef cm2
#undef rlsa
#undef sm1
#undef init
#undef mxstep
#undef mxhnil
#undef nhnil
#undef nslast
#undef nyh
#undef icf
#undef ierpj
#undef iersl
#undef jcur
#undef jstart
#undef kflag
#undef l
#undef lyh
#undef lewt
#undef lacor
#undef lsavf
#undef lwm
#undef liwm
#undef meth
#undef miter
#undef maxord
#undef maxcor
#undef msbp
#undef mxncf
#undef n
#undef nq
#undef nst
#undef nfe
#undef nje
#undef nqu
#undef ialth
#undef ipup
#undef lmax
#undef nqnyh
#undef nslp
#undef ils
#undef insufr
#undef insufi
#undef ixpr
#undef jtyp
#undef mused
#undef mxordn
#undef mxords
#undef icount
#undef irflag
#undef ilsa

namespace srg_lsoda {

template<int N>
struct Workspace {
    // DLSODA's full-Jacobian allocation supports both Adams and BDF.
    static constexpr int real_size = 22 + N * (N + 9 > 16 ? N + 9 : 16);
    static constexpr int integer_size = 20 + N;
    double real_work[real_size];
    int integer_work[integer_size];
    cuLsodaCommonBlock common;
};

struct UnusedJacobian {
    // JT=2 asks LSODA to form a dense finite-difference Jacobian itself.
    __device__ void operator()(int*, double*, double*, int, int, double*, int) const {}
};

// Integrate one smooth forcing interval. A new call deliberately restarts the
// multistep history, retaining y and t but discarding derivatives from before
// a square-wave jump or the input cutoff. Each thread owns its workspace.
template<int N, class Rhs, class Jacobian = UnusedJacobian>
__device__ int integrate(Rhs rhs, double* y, double& t, double stop,
                         double max_step, double relative_tol, double absolute_tol,
                         int max_steps, Workspace<N>& work, int& detail,
                         Jacobian jacobian = Jacobian{})
{
    detail = 0;
    if (!(stop > t) || !(max_step > 0) || !std::isfinite(max_step) || !std::isfinite(stop) || !std::isfinite(t))
        return -1000;

    for (int i = 0; i < Workspace<N>::real_size; ++i) work.real_work[i] = 0;
    for (int i = 0; i < Workspace<N>::integer_size; ++i) work.integer_work[i] = 0;
    cuLsodaCommonBlockInit(&work.common);

    int neq = N;
    int real_size = Workspace<N>::real_size;
    int integer_size = Workspace<N>::integer_size;
    int itol = 1;     // scalar RTOL and ATOL, applied to all states
    int itask = 4;    // reach TOUT without stepping past TCRIT
    int istate = 1;   // fresh history for this smooth interval
    int iopt = 1;     // enable HMAX and MXSTEP
    // Keep finite differences as the default for other systems/callers.
    int jt = std::is_same_v<Jacobian, UnusedJacobian> ? 2 : 1;
    double unused_parameter = 0;
    work.real_work[0] = stop;       // TCRIT = TOUT, in zero-based C storage
    work.real_work[5] = max_step;   // HMAX: same upper step bound as RK
    work.integer_work[5] = max_steps;

    // A long interval may exceed MXSTEP even though integration is healthy.
    // Continue with the existing history, but bound retries and require progress.
    for (int continuation = 0; continuation < 1000; ++continuation) {
        double previous_t = t;
        detail = dlsoda_(rhs, &neq, y, &t, &stop, &itol, &relative_tol,
                         &absolute_tol, &itask, &istate, &iopt,
                         work.real_work, &real_size, work.integer_work,
                         &integer_size, jacobian, &jt, &work.common,
                         &unused_parameter);
        if (!std::isfinite(t)) return -1001;
        for (int i = 0; i < N; ++i)
            if (!std::isfinite(y[i])) return -1001;

        if (istate == 2) {
            const double tolerance = 128.0 * 2.2204460492503131e-16 * (1.0 + fabs(stop));
            return fabs(t - stop) <= tolerance ? 2 : -1002;
        }
        if (istate == -1 && t > previous_t && t < stop) {
            istate = 2;
            continue;
        }
        return istate < 0 ? istate : -1002;
    }
    return -1003;
}

} // namespace srg_lsoda
