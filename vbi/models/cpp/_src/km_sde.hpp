/**
 * @file km_sde.hpp
 * @brief Kuramoto model with stochastic noise, phase lags, and OpenMP
 *        parallelism for a network of N oscillators.
 *
 * Implements the generalised Kuramoto model with heterogeneous natural
 * frequencies, a global coupling strength G, and per-edge phase lags α_{ij}.
 * Noise enters additively as white Gaussian noise.  Integration uses the
 * Heun (predictor-corrector) SDE scheme.
 *
 * **Model references:**
 * - Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence.*
 *   Springer, Berlin. https://doi.org/10.1007/978-3-642-69689-3
 * - Breakspear, M., Heitmann, S., & Daffertshofer, A. (2010). Generative
 *   models of cortical oscillations. *Frontiers in Human Neuroscience*, 4, 190.
 *
 * **State vector:** theta[i] — instantaneous phase of oscillator i [rad].
 *
 * **Equations:**
 * ```
 *   dθ_i/dt = ω_i + G * sum_j w_ij * sin(θ_j - θ_i - α_ij) + noise
 * ```
 */

#ifndef KM_SDE_HPP
#define KM_SDE_HPP

#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <assert.h>
#include <iostream>
#include <omp.h>
#include "utility.hpp"

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1> dim2;
typedef std::vector<float> dim1f;
typedef std::vector<dim1f> dim2f;
typedef std::vector<size_t> dim1i;
typedef std::vector<dim1i> dim2i;
typedef std::vector<std::vector<unsigned>> dim2I;

/**
 * @brief Kuramoto SDE — N phase oscillators with noise and phase lags.
 *
 * The coupling is through sine differences with an optional per-edge
 * phase lag matrix @p alpha.  OpenMP parallelism is used in the RHS loop.
 */
class KM_sde
{
    /*
    Kuramoto model with stochastic noise
    solve the SDE using heun's stochastic integration scheme.
    */

private:
    double dt;            ///< Integration time step [s]
    double t_initial;     ///< Simulation start time [s]
    double t_transition;  ///< End of transient period [s]
    double t_end;         ///< Total simulation end time [s]
    double G;             ///< Global coupling strength
    double noise_amp;     ///< Additive noise amplitude (std of Gaussian increment)
    dim1 theta;           ///< Oscillator phase state vector (length N) [rad]
    dim1 omega;           ///< Natural frequencies (length N) [rad/s]
    dim2 alpha;           ///< Phase lag matrix α_{ij} (N×N) [rad]
    dim2 weights;         ///< N×N coupling weight matrix
    int num_nodes;        ///< Number of oscillators
    dim2 Theta;           ///< Recorded phase time series, shape n_record × N
    dim1 times;           ///< Recorded time array, length n_record
    dim2I adjlist;        ///< Sparse adjacency list
    size_t num_steps;             ///< Total integration steps
    size_t num_steps_transition;  ///< Steps to discard (transient)
    size_t fix_seed;              ///< RNG seed flag forwarded to rng()
    size_t buffer_size;           ///< Number of post-transient steps to record

public:
    /**
     * @brief Construct a KM_sde simulator.
     *
     * @param dt           Integration time step [s].
     * @param t_initial    Simulation start time [s].
     * @param t_transition End of transient period [s]; data before this is
     *                     discarded.
     * @param t_end        Total simulation end time [s].
     * @param G            Global coupling strength.
     * @param noise_amp    Standard deviation of additive Gaussian noise per
     *                     step (not scaled by sqrt(dt) internally — caller
     *                     should pre-scale if desired).
     * @param theta        Initial phase vector (length N) [rad].
     * @param omega        Natural frequency vector (length N) [rad/s].
     * @param alpha        N×N phase lag matrix [rad].
     * @param weights      N×N coupling weight matrix.
     * @param fix_seed     RNG seed flag (0 = random, non-zero = fixed,
     *                     default 0).
     * @param num_threads  Number of OpenMP threads for the RHS loop (default 1).
     */
    KM_sde(double dt,
           double t_initial,
           double t_transition,
           double t_end,
           double G,
           double noise_amp,
           dim1 theta,
           dim1 omega,
           dim2 alpha,
           dim2 weights,
           size_t fix_seed = 0,
           size_t num_threads = 1
           )
    {
        num_nodes = theta.size();
        this->dt = dt;
        this->t_initial = t_initial;
        this->t_transition = t_transition;
        this->t_end = t_end;
        this->G = G;
        this->theta = theta;
        this->omega = omega;
        this->alpha = alpha;
        this->weights = weights;
        this->noise_amp = noise_amp;
        this->fix_seed = fix_seed;

        omp_set_num_threads(num_threads);

        adjlist = adjmat_to_adjlist(weights);
        num_steps = int((t_end - t_initial) / dt);
        num_steps_transition = int((t_transition - t_initial) / dt);
        buffer_size = num_steps - num_steps_transition;
        Theta.resize(buffer_size);
        for (size_t i = 0; i < buffer_size; i++)
        {
            Theta[i].resize(num_nodes);
        }
        times.resize(buffer_size);
    }

    /**
     * @brief Evaluate the Kuramoto right-hand side.
     *
     * For oscillator i:
     * ```
     *   dθ_i/dt = ω_i + G * sum_{j in neighbours(i)} w_ij * sin(θ_j - θ_i - α_ij)
     * ```
     * The loop is parallelised with OpenMP.
     *
     * @param theta   Current phase vector (length N) [rad].
     * @param dtheta  Output derivative vector (length N, pre-allocated).
     * @param t       Current time [s] (unused; kept for API consistency).
     */
    void rhs_f(dim1 &theta, dim1 &dtheta, const double /*t*/)
    {
        double sumj = 0.0;
#pragma omp parallel for reduction(+ : sumj)
        for (int i = 0; i < num_nodes; i++)
        {
            sumj = 0.0;
            for (int j = 0; j < adjlist[i].size(); j++)
            {
                int k = adjlist[i][j];
                sumj += weights[i][k] * sin(theta[k] - theta[i] - alpha[i][k]);
            }
            dtheta[i] = omega[i] + G * sumj;
        }
    }

    /**
     * @brief Perform one Heun SDE step, updating @p y (phase vector) in-place.
     *
     * Additive noise (normal distribution, std = noise_amp) is applied to all
     * oscillators in both the predictor and corrector passes.
     *
     * @param y  Phase vector (length N) [rad], modified in-place.
     * @param t  Current time [s].
     */
    void heun(dim1 &y, const double t)
    {
        std::normal_distribution<> normal(0, 1);

        size_t nn = num_nodes;
        dim1 tmp(nn);
        dim1 k1(nn);
        dim1 k2(nn);

        rhs_f(y, k1, t);

        for (size_t i = 0; i < nn; ++i)
            tmp[i] = y[i] + dt * k1[i] + noise_amp * normal(rng(fix_seed));

        rhs_f(tmp, k2, t + dt);
        for (size_t i = 0; i < nn; ++i)
            y[i] += 0.5 * dt * (k1[i] + k2[i]) + noise_amp * normal(rng(fix_seed));

    }

    /**
     * @brief Run the full simulation using the Heun SDE scheme.
     *
     * First advances the state for @c num_steps_transition steps (transient),
     * then records phase vectors and times for @c buffer_size post-transient
     * steps.
     */
    void IntegrateHeun()
    {
        for (size_t i = 0; i < num_steps_transition; i++)
        {
            heun(theta, t_initial);
            t_initial += dt;
        }

        for (size_t i = 0; i < buffer_size; i++)
        {
            heun(theta, t_initial);
            Theta[i] = theta;
            times[i] = t_initial;
            t_initial += dt;
        }
    }

    /**
     * @brief Return the recorded post-transient time array.
     * @return dim1 of length buffer_size = (t_end - t_transition) / dt.
     */
    dim1 get_times()
    {
        return times;
    }

    /**
     * @brief Return the recorded post-transient phase time series.
     * @return dim2 of shape buffer_size × N [rad].
     */
    dim2 get_theta()
    {
        return Theta;
    }
};

#endif
