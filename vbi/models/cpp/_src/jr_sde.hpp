/**
 * @file jr_sde.hpp
 * @brief Jansen-Rit neural mass model with stochastic noise and no axonal
 *        delays (SDE — Stochastic Differential Equations).
 *
 * Implements the Jansen-Rit (JR) model for a network of N instantaneously
 * coupled cortical columns.  Each column is described by three second-order
 * linear operators (excitatory pyramidal, excitatory interneuron, inhibitory
 * interneuron) driven by a sigmoidal firing-rate function.
 *
 * **Model reference:**
 * Jansen, B.H. & Rit, V.G. (1995). Electroencephalogram and visual evoked
 * potential generation in a mathematical model of coupled cortical columns.
 * *Biological Cybernetics*, 73(4), 357–366.
 * https://doi.org/10.1007/BF00199471
 *
 * **State-vector layout** (flat vector of length 6×N):
 * | Index range | Variable | Description |
 * |-------------|----------|-------------|
 * | [0,   N)    | x0       | Slow IPSP — inhibitory interneuron output |
 * | [N,  2N)    | x1       | EPSP — pyramidal cell output |
 * | [2N, 3N)    | x2       | Fast IPSP — inhibitory interneuron output |
 * | [3N, 4N)    | ẋ0       | Velocity of x0 |
 * | [4N, 5N)    | ẋ1       | Velocity of x1 — **noise applied here** |
 * | [5N, 6N)    | ẋ2       | Velocity of x2 |
 *
 * The EEG proxy (pyramidal-cell PSP) is: `x1 - x2` = `y[i+N] - y[i+2N]`.
 */

#ifndef jr_sde_HPP
#define jr_sde_HPP

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

/**
 * @brief Jansen-Rit SDE — N instantaneously coupled cortical columns.
 *
 * Noise enters only through the ẋ1 (excitatory pyramidal velocity) component,
 * modelling stochastic afferent input.  Integration uses either the
 * Euler-Maruyama or Heun (order-1.5 strong) scheme.
 */
class JR_sde
{
private:
    size_t N;               ///< Number of network nodes
    double dt;              ///< Integration time step [s]
    double t_initial;       ///< Simulation start time [s]
    double t_final;         ///< Total simulation time [s]
    double t_transition;    ///< Transient period to discard [s]
    size_t dimension;       ///< Total state dimension (= 6*N)
    size_t num_steps;       ///< Total number of integration steps
    size_t index_transition;///< Step index corresponding to t_transition
    vector<vector<unsigned>> adjlist; ///< Sparse adjacency list

    dim1 A;         ///< Per-node excitatory synaptic gain [mV] (length N)
    double par_B;   ///< Inhibitory synaptic gain [mV]
    double par_a;   ///< Excitatory synaptic time constant [1/s]
    double par_b;   ///< Inhibitory synaptic time constant [1/s]
    double par_r;   ///< Sigmoid slope parameter [1/mV]
    double par_v0;  ///< Sigmoid half-maximum potential [mV]
    double par_vmax;///< Maximum firing rate of sigmoid [Hz]
    double coupling;     ///< Global inter-column coupling strength G
    double noise_mu;     ///< Mean of Gaussian afferent input [Hz]
    double noise_sigma;  ///< Standard deviation of additive noise

    dim2 adj;  ///< N×N weight matrix
    dim1 C0;   ///< Per-node intra-column connectivity (slow inhibitory)
    dim1 C1;   ///< Per-node intra-column connectivity (pyramidal)
    dim1 C2;   ///< Per-node intra-column connectivity (fast inhibitory)
    dim1 C3;   ///< Per-node intra-column connectivity (fast inhib. feedback)

    int fix_seed; ///< Seed flag forwarded to rng()

    dim1 times;        ///< Recorded post-transient time array
    dim2 states;       ///< Recorded observable (x1-x2), shape n_record × N
    dim1 initial_state;///< Flat initial state vector (length 6*N)

    // bool ADJ_SET = false;      //check if adjacency matrix is set.
    // bool COUPLING_SET = false; //check if coupling is set

public:
    /**
     * @brief Construct a JR_sde simulator.
     *
     * @param N             Number of network nodes.
     * @param dt            Integration time step [s].
     * @param t_transition  Transient period to discard [s].
     * @param t_final       Total simulation time [s].
     * @param coupling      Global coupling scaling factor G.
     * @param adj           N×N weight matrix of inter-column coupling strengths.
     * @param y             Flat initial state vector (length 6*N).
     * @param A             Per-node excitatory synaptic gain vector [mV].
     * @param B             Inhibitory synaptic gain [mV].
     * @param a             Excitatory synaptic time constant [1/s].
     * @param b             Inhibitory synaptic time constant [1/s].
     * @param r             Sigmoid slope [1/mV].
     * @param v0            Sigmoid half-maximum potential [mV].
     * @param vmax          Maximum firing rate of sigmoid [Hz].
     * @param C0            Per-node intra-column connectivity (slow inhibitory).
     * @param C1            Per-node intra-column connectivity (pyramidal).
     * @param C2            Per-node intra-column connectivity (fast inhibitory).
     * @param C3            Per-node intra-column connectivity (fast inhib. feedback).
     * @param noise_mu      Mean of Gaussian afferent input (background drive) [Hz].
     * @param noise_sigma   Standard deviation of additive noise.
     * @param fix_seed      0 = random seed; non-zero = fixed seed (default 0).
     */
    JR_sde(size_t N,
        double dt,
        double t_transition,
        double t_final,
        double coupling,
        dim2 adj,
        dim1 y,
        dim1 A,
        double B,
        double a,
        double b,
        double r,
        double v0,
        double vmax,
        dim1 C0,
        dim1 C1,
        dim1 C2,
        dim1 C3,
        double noise_mu,
        double noise_sigma,
        int fix_seed=0)
    {
        assert(t_final > t_transition);

        this->A = A;
        par_B = B;
        par_a = a;
        par_b = b;
        par_r = r;
        par_v0 = v0;
        par_vmax = vmax;

        this-> noise_mu = noise_mu;
        this-> noise_sigma = noise_sigma;

        initial_state = y;
        this->N = N;
        this->dt = dt;
        this->t_final = t_final;
        this->t_transition = t_transition;
        this->coupling = coupling;
        this->adj = adj;
        this->fix_seed = fix_seed;
        this->C0 = C0;
        this->C1 = C1;
        this->C2 = C2;
        this->C3 = C3;

        adjlist = adjmat_to_adjlist(adj);

        dimension = y.size();
        num_steps = int(t_final / dt);

        index_transition = int(t_transition / dt);
        size_t buffer_size = num_steps - index_transition; //   int((t_final - t_transition) / dt);

        states.resize(buffer_size);
        for (size_t i = 0; i < buffer_size; ++i)
            states[i].resize(N);
        times.resize(buffer_size);
    }

    /**
     * @brief Sigmoid (firing-rate) transfer function.
     *
     * `sigma(v) = vmax / (1 + exp(r * (v0 - v)))`
     *
     * @param v  Membrane potential [mV].
     * @return   Firing rate [Hz].
     */
    double sigma(const double v)
    {
        return par_vmax / (1 + exp(par_r * (par_v0 - v)));
    }

    /**
     * @brief Evaluate the right-hand side of the JR-SDE system.
     *
     * Computes the deterministic drift for all 6*N state variables.
     * Inter-column coupling is instantaneous (no delays).
     *
     * @param y     Current state vector (length 6*N).
     * @param dxdt  Output derivative vector (length 6*N, pre-allocated).
     * @param t     Current time [s] (unused; kept for API consistency).
     */
    void rhs(const vector<double> &y,
             vector<double> &dxdt,
             const double t)
    {

        double a2 = par_a * par_a;
        double b2 = par_b * par_b;
        double Bb = par_B * par_b;

        size_t N2 = 2 * N;
        size_t N3 = 3 * N;
        size_t N4 = 4 * N;
        size_t N5 = 5 * N;

        for (size_t i = 0; i < N; ++i)
        {
            double coupling_term = 0.0;

            for (size_t j = 0; j < adjlist[i].size(); ++j)
            {
                int k = adjlist[i][j];
                coupling_term += adj[i][k] * sigma(y[k + N] - y[k + N2]);
            }

            dxdt[i] = y[i + N3];
            dxdt[i + N] = y[i + N4];
            dxdt[i + N2] = y[i + N5];
            dxdt[i + N3] = A[i] * par_a * sigma(y[i + N] - y[i + N2]) - 2 * par_a * y[i + N3] - a2 * y[i];
            dxdt[i + N4] = A[i] * par_a * (noise_mu + C1[i] * sigma(C0[i] * y[i]) + coupling * coupling_term) - 2 * par_a * y[i + N4] - a2 * y[i + N];
            dxdt[i + N5] = Bb * C3[i] * sigma(C2[i] * y[i]) - 2 * par_b * y[i + N5] -
                           b2 * y[i + N2];
        }
    }

    /**
     * @brief Perform one Euler-Maruyama step, updating @p y in-place.
     *
     * Additive white noise (scaled by sqrt(dt) * noise_sigma) is applied
     * only to the ẋ1 components (indices [4N, 5N)).
     *
     * @param y  State vector to advance (length 6*N), modified in-place.
     * @param t  Current time [s].
     */
    void euler(dim1 &y, const double t)
    {
        std::normal_distribution<> normal(0, 1);

        size_t n = y.size();
        dim1 dydt(n);
        rhs(y, dydt, t);
        for (size_t i = 0; i < n; ++i)
        {
            if ((i>= (4*N)) && (i<(5*N)))
                y[i] += dydt[i] * dt + sqrt(dt) * noise_sigma * normal(rng(fix_seed));
            else
                y[i] += dydt[i] * dt;
        }
    }

    /**
     * @brief Run the full simulation using Euler-Maruyama integration.
     *
     * Discards the transient period (index_transition steps) and records
     * the EEG proxy (x1 - x2) and time array post-transient.
     */
    void eulerIntegrate()
    {
        size_t N2 = 2 * N;

        dim1 y = initial_state;
        size_t counter = 0;

        for (int step = 0; step < num_steps; ++step)
        {
            double t = step * dt;

            if (step >= index_transition)
            {
                times[counter] = t;

                for (size_t i = 0; i < N; ++i)
                    states[counter][i] = y[i + N] - y[i + N2];
                counter++;
            }
            euler(y, t);
        }
    }

    /**
     * @brief Perform one Heun (predictor-corrector) SDE step, updating @p y in-place.
     *
     * Noise is applied only to the ẋ1 components (indices [4N, 5N)).
     *
     * @param y  State vector to advance (length 6*N), modified in-place.
     * @param t  Current time [s].
     */
    void heun(dim1 &y, const double t)
    {
        std::normal_distribution<> normal(0, 1);

        size_t n = y.size();
        dim1 k1(n);
        dim1 k2(n);
        dim1 tmp(n);
        rhs(y, k1, t);
        for (size_t i = 0; i < n; ++i)
        {
            if ((i>= (4*N)) && (i<(5*N)))
                tmp[i] = y[i] + k1[i] * dt + sqrt(dt) * noise_sigma * normal(rng(fix_seed));
            else
                tmp[i] = y[i] + k1[i] * dt;
        }

        rhs(tmp, k2, t + dt);
        for (size_t i = 0; i < n; ++i)
        {
            if ((i>= (4*N)) && (i<(5*N)))
                y[i] += 0.5 * dt * (k1[i] + k2[i]) + sqrt(dt) * noise_sigma * normal(rng(fix_seed));
            else
                y[i] += 0.5 * dt * (k1[i] + k2[i]);
        }

    }

    /**
     * @brief Run the full simulation using the Heun SDE scheme.
     *
     * Discards the transient period (index_transition steps) and records
     * the EEG proxy (x1 - x2) and time array post-transient.
     */
    void heunIntegrate()
    {
        size_t N2 = 2 * N;

        dim1 y = initial_state;
        size_t counter = 0;

        for (int step = 0; step < num_steps; ++step)
        {
            double t = step * dt;
            if (step >= index_transition)
            {
                times[counter] = t;
                for (size_t i = 0; i < N; ++i)
                    states[counter][i] = y[i + N] - y[i + N2];
                counter++;
            }
            heun(y, t);
        }
    }

    /**
     * @brief Return the recorded EEG proxy (x1 - x2) for all nodes.
     * @return dim2 of shape n_record × N.
     */
    dim2 get_coordinates()
    {
        return states;
    }

    /**
     * @brief Return the recorded post-transient time array.
     * @return dim1 of length n_record = (t_final - t_transition) / dt.
     */
    dim1 get_times()
    {
        return times;
    }
};

#endif
