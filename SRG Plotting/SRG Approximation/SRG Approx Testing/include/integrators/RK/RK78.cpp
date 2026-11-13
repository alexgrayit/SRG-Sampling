#include <stdlib.h>
#include <string.h>

#include "RK78.h"

/*
 * One fixed RKF7(8) step.
 *
 * Inputs:
 *   t       current time
 *   h       timestep
 *   x       current state vector, length n
 *   n       state dimension
 *   params  user parameters passed through to f
 *   f       derivative callback
 *
 * Outputs:
 *   x_next  next state after one step (7th-order solution)
 *   err_est optional embedded error estimate vector = x7 - x8
 *           pass NULL if you do not need it
 *
 * Returns:
 *   0 on success
 *  -1 on allocation failure
 */

int rkf78_step(
    double t,
    double h,
    RNum* x,
    int n,
    void* params,
    DerivFunc f,
    RNum* x_next,
    RNum* err_est
)
{
    int i;
    const RNum step = RNum(h);

    /* 13 stages, each of length n */
    RNum* k = (RNum*)malloc((size_t)(13 * n) * sizeof(RNum));
    RNum* xtmp = (RNum*)malloc((size_t)n * sizeof(RNum));

    if (k == NULL || xtmp == NULL) {
        free(k);
        free(xtmp);
        return -1;
    }

    #define K(stage, idx) k[(stage) * n + (idx)]

    /* Stage 0 */
    f(t, x, &k[0 * n], params);

    /* Stage 1 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (RNum(2.0 / 27.0) * K(0, i));
    }
    f(t + h * (2.0 / 27.0), xtmp, &k[1 * n], params);

    /* Stage 2 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(1.0 / 36.0) * K(0, i) +
            RNum(1.0 / 12.0) * K(1, i)
        );
    }
    f(t + h * (1.0 / 9.0), xtmp, &k[2 * n], params);

    /* Stage 3 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(1.0 / 24.0) * K(0, i) +
            RNum(1.0 / 8.0)  * K(2, i)
        );
    }
    f(t + h * (1.0 / 6.0), xtmp, &k[3 * n], params);

    /* Stage 4 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(5.0  / 12.0) * K(0, i) +
            RNum(-25.0 / 16.0) * K(2, i) +
            RNum(25.0 / 16.0) * K(3, i)
        );
    }
    f(t + h * (5.0 / 12.0), xtmp, &k[4 * n], params);

    /* Stage 5 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(1.0 / 20.0) * K(0, i) +
            RNum(1.0 / 4.0)  * K(3, i) +
            RNum(1.0 / 5.0)  * K(4, i)
        );
    }
    f(t + h * (1.0 / 2.0), xtmp, &k[5 * n], params);

    /* Stage 6 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(-25.0 / 108.0) * K(0, i) +
            RNum(125.0 / 108.0) * K(3, i) +
            RNum(-65.0 / 27.0)  * K(4, i) +
            RNum(125.0 / 54.0)  * K(5, i)
        );
    }
    f(t + h * (5.0 / 6.0), xtmp, &k[6 * n], params);

    /* Stage 7 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(31.0 / 300.0) * K(0, i) +
            RNum(61.0 / 225.0) * K(4, i) +
            RNum(-2.0 / 9.0)   * K(5, i) +
            RNum(13.0 / 900.0) * K(6, i)
        );
    }
    f(t + h * (1.0 / 6.0), xtmp, &k[7 * n], params);

    /* Stage 8 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(2.0)             * K(0, i) +
            RNum(-53.0 / 6.0)   * K(3, i) +
            RNum(704.0 / 45.0)  * K(4, i) +
            RNum(-107.0 / 9.0)  * K(5, i) +
            RNum(67.0 / 90.0)   * K(6, i) +
            RNum(3.0)             * K(7, i)
        );
    }
    f(t + h * (2.0 / 3.0), xtmp, &k[8 * n], params);

    /* Stage 9 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(-91.0 / 108.0) * K(0, i) +
            RNum(23.0 / 108.0)  * K(3, i) +
            RNum(-976.0 / 135.0)* K(4, i) +
            RNum(311.0 / 54.0)  * K(5, i) +
            RNum(-19.0 / 60.0)  * K(6, i) +
            RNum(17.0 / 6.0)    * K(7, i) +
            RNum(-1.0 / 12.0)   * K(8, i)
        );
    }
    f(t + h * (1.0 / 3.0), xtmp, &k[9 * n], params);

    /* Stage 10 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(2383.0 / 4100.0) * K(0, i) +
            RNum(-341.0 / 164.0)  * K(3, i) +
            RNum(4496.0 / 1025.0) * K(4, i) +
            RNum(-301.0 / 82.0)   * K(5, i) +
            RNum(2133.0 / 4100.0) * K(6, i) +
            RNum(45.0 / 82.0)     * K(7, i) +
            RNum(45.0 / 164.0)    * K(8, i) +
            RNum(18.0 / 41.0)     * K(9, i)
        );
    }
    f(t + h, xtmp, &k[10 * n], params);

    /* Stage 11 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(3.0 / 205.0)   * K(0, i) +
            RNum(-6.0 / 41.0)   * K(5, i) +
            RNum(-3.0 / 205.0)  * K(6, i) +
            RNum(-3.0 / 41.0)   * K(7, i) +
            RNum(3.0 / 41.0)    * K(8, i) +
            RNum(6.0 / 41.0)    * K(9, i)
        );
    }
    f(t, xtmp, &k[11 * n], params);

    /* Stage 12 */
    for (i = 0; i < n; ++i) {
        xtmp[i] = x[i] + step * (
            RNum(-1777.0 / 4100.0) * K(0, i) +
            RNum(-341.0 / 164.0)   * K(3, i) +
            RNum(4496.0 / 1025.0)  * K(4, i) +
            RNum(-289.0 / 82.0)    * K(5, i) +
            RNum(2193.0 / 4100.0)  * K(6, i) +
            RNum(51.0 / 82.0)      * K(7, i) +
            RNum(33.0 / 164.0)     * K(8, i) +
            RNum(12.0 / 41.0)      * K(9, i) +
            RNum(1.0)                * K(11, i)
        );
    }
    f(t + h, xtmp, &k[12 * n], params);

    /*
     * 7th-order propagated solution
     *
     * x7 = x + h * (
     *      34/105 * k6
     *    +  9/35  * k7
     *    +  9/35  * k8
     *    +  9/280 * k9
     *    +  9/280 * k10
     *    + 41/840 * k12
     *    + 41/840 * k13 )
     */
    for (i = 0; i < n; ++i) {
        x_next[i] = x[i] + step * (
            RNum(34.0 / 105.0) * K(5, i) +
            RNum(9.0  / 35.0)  * K(6, i) +
            RNum(9.0  / 35.0)  * K(7, i) +
            RNum(9.0  / 280.0) * K(8, i) +
            RNum(9.0  / 280.0) * K(9, i) +
            RNum(41.0 / 840.0) * K(11, i) +
            RNum(41.0 / 840.0) * K(12, i)
        );
    }

    /*
     * Embedded difference: err_est = x7 - x8
     *
     * err = -41/840 * h * (k1 + k11 - k12 - k13)
     * with 1-based paper indexing.
     */
    if (err_est != NULL) {
        for (i = 0; i < n; ++i) {
            err_est[i] = step * (
                RNum(-41.0 / 840.0) * K(0, i)  +
                RNum(-41.0 / 840.0) * K(10, i) +
                RNum( 41.0 / 840.0) * K(11, i) +
                RNum( 41.0 / 840.0) * K(12, i)
            );
        }
    }

    #undef K

    free(k);
    free(xtmp);
    return 0;
}