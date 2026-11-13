#ifndef RK78_H
#define RK78_H

// State arithmetic follows RNum; time remains double.
#ifndef RNum
#define RNum double
#endif

/* Derivative function type */
typedef void (*DerivFunc)(double t, RNum* x, RNum* x_dot, void* params);

/**
 * @brief This does a Runge-Kutta 7/8 Step
 * 
 * @param t The time step
 * @param h ...
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
);


#endif // RK78_H