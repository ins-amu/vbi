/**
 * @file wc_ode.hpp
 * @brief Wilson-Cowan neural mass model (ODE) for a network of N nodes.
 *
 * Implements the Wilson-Cowan equations describing the mean activity of
 * coupled excitatory (E) and inhibitory (I) neural populations at each node.
 * The dynamics are deterministic (ODE); for SDE variants see the Numba/JAX
 * backends.
 *
 * **Model reference:**
 * Wilson, H.R. & Cowan, J.D. (1972). Excitatory and inhibitory interactions
 * in localized populations of model neurons. *Biophysical Journal*, 12(1), 1–24.
 * https://doi.org/10.1016/S0006-3495(72)86068-5
 *
 * **State-vector layout** (flat vector of length 2×N):
 * | Index range | Variable | Description |
 * |-------------|----------|-------------|
 * | [0,   N)    | E        | Excitatory population activity (fraction active) |
 * | [N,  2N)    | I        | Inhibitory population activity (fraction active) |
 *
 * Inter-node coupling enters separately through E (gain g_e) and I (gain g_i)
 * via a dense matrix–vector product with the weight matrix.
 */

#ifndef WC_ODE_HPP_
#define WC_ODE_HPP_

#include <cmath>
#include <fenv.h>
#include <vector>
#include <random>
#include <string>
#include <assert.h>
#include <iostream>
#include <fstream>
#include "utility.hpp"

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1> dim2;
typedef std::vector<unsigned int> dim1I;
typedef std::vector<dim1I> dim2I;

/**
 * @brief Wilson-Cowan ODE — N coupled excitatory/inhibitory neural populations.
 *
 * Supports three integration schemes: Euler, Heun, and fourth-order
 * Runge-Kutta (RK4).  States are recorded after the transient period t_cut.
 */
class WC_ode
{
private:
    int N;          ///< Number of network nodes
    double dt;      ///< Integration time step [s]
    double c_ee;    ///< Intra-population E→E coupling weight
    double c_ei;    ///< Intra-population I→E coupling weight
    double c_ie;    ///< Intra-population E→I coupling weight
    double c_ii;    ///< Intra-population I→I coupling weight
    double tau_e;   ///< Excitatory population time constant [ms]
    double tau_i;   ///< Inhibitory population time constant [ms]
    double a_e;     ///< Excitatory sigmoid gain
    double a_i;     ///< Inhibitory sigmoid gain
    double b_e;     ///< Excitatory sigmoid threshold
    double b_i;     ///< Inhibitory sigmoid threshold
    double c_e;     ///< Excitatory sigmoid maximum response
    double c_i;     ///< Inhibitory sigmoid maximum response
    double theta_i; ///< Inhibitory external input bias
    double theta_e; ///< Excitatory external input bias
    double r_e;     ///< Excitatory refractory scaling
    double r_i;     ///< Inhibitory refractory scaling
    double k_e;     ///< Excitatory maximum activity bound
    double k_i;     ///< Inhibitory maximum activity bound
    double alpha_e; ///< Excitatory input scaling
    double alpha_i; ///< Inhibitory input scaling
    double g_e;     ///< Long-range coupling gain for excitatory populations
    double g_i;     ///< Long-range coupling gain for inhibitory populations
    double inv_tau_e; ///< 1 / tau_e (precomputed)
    double inv_tau_i; ///< 1 / tau_i (precomputed)
    size_t num_steps; ///< Total integration steps = t_end / dt
    size_t index_cut; ///< Step index at which recording starts = t_cut / dt
    dim1 P;           ///< Per-node external excitatory input (length N)
    dim1 Q;           ///< Per-node external inhibitory input (length N)
    dim1 x0;          ///< Flat initial state vector [E0..EN-1, I0..IN-1]
    dim2 weights;     ///< N×N inter-node weight matrix
    dim2I adjlist;    ///< Sparse adjacency list (used internally by matvec)

    int fix_seed;    ///< Seed flag (reserved; not used in the ODE version)
    double t_end;    ///< Total simulation time [s]
    double t_cut;    ///< Transient period to discard [s]

    vector<float> times;             ///< Recorded post-transient time array
    vector<double> initial_state;    ///< Alias for x0 (unused separately)
    vector<vector<float>> states;    ///< Recorded states, shape n_record × (2N)

public:
    /**
     * @brief Construct a WC_ode simulator.
     *
     * All biophysical parameters have physiologically motivated defaults that
     * reproduce the typical excitatory-inhibitory dynamics described in
     * Wilson & Cowan (1972).
     *
     * @param N       Number of network nodes.
     * @param dt      Integration time step [s].
     * @param P       Per-node external excitatory input (length N).
     * @param Q       Per-node external inhibitory input (length N).
     * @param x0      Flat initial state [E0..EN-1, I0..IN-1] (length 2N).
     * @param weights N×N inter-node coupling weight matrix.
     * @param t_end   Total simulation time [s] (default 300.0).
     * @param t_cut   Transient period to discard [s] (default 0.0).
     * @param c_ee    E→E intra-population coupling (default 16.0).
     * @param c_ei    I→E intra-population coupling (default 12.0).
     * @param c_ie    E→I intra-population coupling (default 15.0).
     * @param c_ii    I→I intra-population coupling (default 3.0).
     * @param tau_e   Excitatory time constant [ms] (default 8.0).
     * @param tau_i   Inhibitory time constant [ms] (default 8.0).
     * @param a_e     Excitatory sigmoid gain (default 1.3).
     * @param a_i     Inhibitory sigmoid gain (default 2.0).
     * @param b_e     Excitatory sigmoid threshold (default 4.0).
     * @param b_i     Inhibitory sigmoid threshold (default 3.7).
     * @param c_e     Excitatory sigmoid maximum (default 1.0).
     * @param c_i     Inhibitory sigmoid maximum (default 1.0).
     * @param theta_i Inhibitory external bias (default 0.0).
     * @param theta_e Excitatory external bias (default 0.0).
     * @param r_e     Excitatory refractory scaling (default 1.0).
     * @param r_i     Inhibitory refractory scaling (default 1.0).
     * @param k_e     Excitatory maximum activity bound (default 0.994).
     * @param k_i     Inhibitory maximum activity bound (default 0.999).
     * @param alpha_e Excitatory input scaling (default 1.0).
     * @param alpha_i Inhibitory input scaling (default 1.0).
     * @param g_e     Long-range coupling gain for E (default 0.0).
     * @param g_i     Long-range coupling gain for I (default 0.0).
     * @param fix_seed RNG seed flag (reserved, default 0).
     */
    WC_ode(
        int N,
        double dt,
        dim1 P,
        dim1 Q,
        dim1 x0,
        dim2 weights,
        double t_end = 300.0,
        double t_cut = 0.0,
        double c_ee = 16.0,
        double c_ei = 12.0,
        double c_ie = 15.0,
        double c_ii = 3.0,
        double tau_e = 8.0,
        double tau_i = 8.0,
        double a_e = 1.3,
        double a_i = 2.0,
        double b_e = 4.0,
        double b_i = 3.7,
        double c_e = 1.0,
        double c_i = 1.0,
        double theta_i = 0.0,
        double theta_e = 0.0,
        double r_e = 1.0,
        double r_i = 1.0,
        double k_e = 0.994,
        double k_i = 0.999,
        double alpha_e = 1.0,
        double alpha_i = 1.0,
        double g_e = 0.0,
        double g_i = 0.0,
        int fix_seed = 0)
    {
        (void)fix_seed; // Mark as intentionally unused
        this->N = N;
        this->dt = dt;
        this->c_ee = c_ee;
        this->c_ei = c_ei;
        this->c_ie = c_ie;
        this->c_ii = c_ii;
        this->tau_e = tau_e;
        this->tau_i = tau_i;
        this->a_e = a_e;
        this->a_i = a_i;
        this->b_e = b_e;
        this->b_i = b_i;
        this->c_e = c_e;
        this->c_i = c_i;
        this->theta_i = theta_i;
        this->theta_e = theta_e;
        this->r_e = r_e;
        this->r_i = r_i;
        this->k_e = k_e;
        this->k_i = k_i;
        this->alpha_e = alpha_e;
        this->alpha_i = alpha_i;
        this->g_e = g_e;
        this->g_i = g_i;
        this->weights = weights;
        this->P = P;
        this->Q = Q;
        this->x0 = x0;

        inv_tau_e = 1.0 / tau_e;
        inv_tau_i = 1.0 / tau_i;

        adjlist = adjmat_to_adjlist(weights);
        num_steps = int(t_end / dt);
        index_cut = int(t_cut / dt);
        size_t buffer_size = num_steps - index_cut;
        times.resize(buffer_size);
        states.resize(buffer_size);
        for (size_t i = 0; i < buffer_size; ++i)
            states[i].resize(2 * N);
    }

    /**
     * @brief Wilson-Cowan sigmoid transfer function.
     *
     * `sigmoid(x) = c / (1 + exp(-a * (x - b)))`
     *
     * @param x  Net input.
     * @param a  Slope (gain).
     * @param b  Threshold (shift).
     * @param c  Maximum response.
     * @return   Fraction of active cells in [0, c].
     */
    double sigmoid(const double x,
                   const double a,
                   const double b,
                   const double c)
    {
        return c / (1.0 + exp(-a * (x - b)));
    }

    /**
     * @brief Evaluate the Wilson-Cowan right-hand side.
     *
     * For node i:
     * ```
     *   dE/dt = inv_tau_e * (-E + (k_e - r_e*E) * Se(x_e))
     *   dI/dt = inv_tau_i * (-I + (k_i - r_i*I) * Si(x_i))
     * ```
     * where x_e = alpha_e*(c_ee*E - c_ei*I + P - theta_e + g_e*lc_e),
     *       x_i = alpha_i*(c_ie*E - c_ii*I + Q - theta_i + g_i*lc_i),
     * and lc_e, lc_i are the long-range coupling terms (dense matrix×vector).
     *
     * @param y     Current state vector [E0..EN-1, I0..IN-1].
     * @param dydt  Output derivative vector (pre-allocated, length 2N).
     * @param dt    Time step (unused; kept for API consistency).
     */
    void rhs(const dim1 &y,
             dim1 &dydt,
             const double dt)
    {
        (void)dt; // Mark as intentionally unused
        dim1 lc_e(N);
        dim1 lc_i(N);
        double thr = 1e-6;
        if (std::abs(g_e) > thr)
            lc_e = matvec(weights, y, 0);
        if (std::abs(g_i) > thr)
            lc_i = matvec(weights, y, N);

        for (int i = 0; i < N; ++i)
        {
            double x_e = alpha_e * (c_ee * y[i] - c_ei * y[i + N] + P[i] - theta_e + g_e * lc_e[i]);
            double x_i = alpha_i * (c_ie * y[i] - c_ii * y[i + N] + Q[i] - theta_i + g_i * lc_i[i]);
            double s_e = sigmoid(x_e, a_e, b_e, c_e);
            double s_i = sigmoid(x_i, a_i, b_i, c_i);
            dydt[i] = inv_tau_e * (-y[i] + (k_e - r_e * y[i]) * s_e);
            dydt[i + N] = inv_tau_i * (-y[i + N] + (k_i - r_i * y[i + N]) * s_i);
        }
    }

    /**
     * @brief Perform one Euler step, updating @p y in-place.
     * @param y   State vector (length 2N), modified in-place.
     * @param dt  Time step [s].
     */
    void euler_step(dim1 &y, const double dt)
    {

        dim1 dydt(2 * N);
        rhs(y, dydt, dt);
        for (int i = 0; i < 2 * N; ++i)
            y[i] += dt * dydt[i];
    }

    /**
     * @brief Run the full simulation using the Euler scheme.
     *
     * Discards the first index_cut steps and records states post-transient.
     */
    void eulerIntegrate()
    {
        size_t nc = 2 * N;
        dim1 dy(nc);
        dim1 y(nc);

        y = x0;
        for (size_t i = 0; i < index_cut; ++i)
            euler_step(y, dt);

        size_t ind = 0;
        for (size_t i = index_cut; i < num_steps; ++i)
        {
            euler_step(y, dt);
            times[ind] = i * dt;
            for (size_t j = 0; j < 2 * static_cast<size_t>(N); ++j)
                states[ind][j] = y[j];
            ind++;
        }
    }

    /**
     * @brief Perform one Heun (predictor-corrector) step, updating @p y in-place.
     * @param y   State vector (length 2N), modified in-place.
     * @param dt  Time step [s].
     */
    void heun_step(dim1 &y, const double dt)
    {
        dim1 dydt(2 * N);
        dim1 tmp(2 * N);
        rhs(y, dydt, dt);
        for (size_t i = 0; i < 2 * static_cast<size_t>(N); ++i)
            tmp[i] = y[i] + dt * dydt[i];
        rhs(tmp, dydt, dt);
        for (size_t i = 0; i < 2 * static_cast<size_t>(N); ++i)
            y[i] += 0.5 * dt * (dydt[i] + dydt[i]);
    }

    /**
     * @brief Run the full simulation using the Heun scheme.
     *
     * Discards the first index_cut steps and records states post-transient.
     */
    void heunIntegrate()
    {
        size_t nc = 2 * N;
        dim1 dy(nc);
        dim1 y(nc);

        y = x0;
        for (size_t i = 0; i < index_cut; ++i)
            heun_step(y, dt);

        size_t ind = 0;
        for (size_t i = index_cut; i < num_steps; ++i)
        {
            heun_step(y, dt);
            times[ind] = i * dt;
            for (size_t j = 0; j < 2 * static_cast<size_t>(N); ++j)
                states[ind][j] = y[j];
            ind++;
        }
    }

    /**
     * @brief Perform one fourth-order Runge-Kutta step, updating @p y in-place.
     * @param y   State vector (length 2N), modified in-place.
     * @param dt  Time step [s].
     */
    void rk4_step(dim1 &y, const double dt)
    {
        dim1 dydt(2 * N);
        dim1 tmp(2 * N);
        dim1 k1(2 * N);
        dim1 k2(2 * N);
        dim1 k3(2 * N);
        dim1 k4(2 * N);
        double dt_over_6 = dt / 6.0;

        rhs(y, k1, dt);
        for (size_t i = 0; i < 2 * static_cast<size_t>(N); ++i)
            tmp[i] = y[i] + 0.5 * dt * k1[i];
        rhs(tmp, k2, dt);
        for (size_t i = 0; i < 2 * static_cast<size_t>(N); ++i)
            tmp[i] = y[i] + 0.5 * dt * k2[i];
        rhs(tmp, k3, dt);
        for (size_t i = 0; i < 2 * static_cast<size_t>(N); ++i)
            tmp[i] = y[i] + dt * k3[i];
        rhs(tmp, k4, dt);
        for (size_t i = 0; i < 2 * static_cast<size_t>(N); ++i)
            y[i] += dt_over_6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    /**
     * @brief Run the full simulation using the RK4 scheme.
     *
     * Discards the first index_cut steps and records states post-transient.
     */
    void rk4Integrate()
    {
        size_t nc = 2 * N;
        dim1 dy(nc);
        dim1 y(nc);

        y = x0;
        for (size_t i = 0; i < index_cut; ++i)
            rk4_step(y, dt);

        size_t ind = 0;
        for (size_t i = index_cut; i < num_steps; ++i)
        {
            rk4_step(y, dt);
            times[ind] = i * dt;
            for (size_t j = 0; j < 2 * static_cast<size_t>(N); ++j)
                states[ind][j] = y[j];
            ind++;
        }
    }

    /**
     * @brief Return the recorded post-transient state time series.
     * @return states of shape n_record × (2N), stored as vector<vector<float>>.
     */
    vector<vector<float>> get_states()
    {
        return states;
    }

    /**
     * @brief Return the recorded post-transient time array.
     * @return times of length n_record = (t_end - t_cut) / dt, as vector<float>.
     */
    vector<float> get_times()
    {
        return times;
    }
};

#endif // WC_ODE_HPP_
