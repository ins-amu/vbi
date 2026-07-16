/**
 * @file vep.hpp
 * @brief Virtual Epileptic Patient (VEP) model for a network of N brain regions.
 *
 * Implements a simplified two-variable Epileptor formulation used in the
 * Virtual Epileptic Patient pipeline.  Each node has a fast seizure variable
 * x and a slow permittivity variable z.  Stochastic noise and long-range
 * inter-regional coupling (through x) are included.
 *
 * **Model references:**
 * - Jirsa, V.K. et al. (2014). On the nature of seizure dynamics.
 *   *Brain*, 137(8), 2210–2230. https://doi.org/10.1093/brain/awu133
 * - Proix, T. et al. (2014). Permittivity coupling across brain regions
 *   determines seizure recruitment in partial epilepsy.
 *   *Journal of Neuroscience*, 34(45), 15009–15021.
 *
 * **State-vector layout** (flat vector of length 2×N):
 * | Index range | Variable | Description |
 * |-------------|----------|-------------|
 * | [0,   N)    | x        | Fast seizure variable per node |
 * | [N,  2N)    | z        | Slow permittivity variable per node |
 *
 * **Equations:**
 * ```
 *   dx/dt = 1 - x^3 - 2x^2 - z + iext
 *   dz/dt = inv_tau * (4*(x - eta) - z - G * sum_j w_ij*(x_j - x_i))
 * ```
 * Noise enters both x and z components uniformly.
 */

#ifndef bvep_HPP
#define bvep_HPP

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
 * @brief VEP — N brain regions modelled as simplified 2-variable Epileptors.
 *
 * Noise (normal distribution, std = noise_sigma * sqrt(dt)) is applied to
 * both x and z at every integration step.  Supports Euler-Maruyama and Heun
 * schemes selected at runtime via the @p method argument.
 */
class VEP
{
private:
    int nn;          ///< Number of network nodes (brain regions)
    dim1 x;          ///< Scratch state variable (unused field; state is local in integrate())
    dim1 eta;        ///< Per-node epileptogenicity parameter η (seizure threshold)
    double G;        ///< Global inter-regional coupling strength
    dim1 iext;       ///< Per-node external excitatory input
    double dt;       ///< Integration time step [s]
    dim1 times;      ///< Recorded post-transient time array
    double tcut;     ///< Transient period to discard [s]
    double tend;     ///< Total simulation time [s]
    dim2 states;     ///< Recorded x states, shape n_record × N
    int fix_seed;    ///< RNG seed flag forwarded to rng()
    dim2 weights;    ///< N×N inter-regional weight matrix
    string method;   ///< Integration method: "euler" or "heun"
    double inv_tau;  ///< 1 / tau (precomputed permittivity time constant)
    unsigned num_steps;   ///< Total integration steps = tend / dt
    double noise_sigma;   ///< Standard deviation of additive Gaussian noise
    dim1 initial_state;   ///< Flat initial state [x0..xN-1, z0..zN-1]
    vector<vector<unsigned>> adjlist; ///< Sparse adjacency list

public:
    /**
     * @brief Construct a VEP simulator.
     *
     * @param G             Global inter-regional coupling strength.
     * @param iext          Per-node external input vector (length N).
     * @param eta           Per-node epileptogenicity η (length N); controls
     *                      seizure threshold — more negative = more
     *                      epileptogenic.
     * @param dt            Integration time step [s].
     * @param tcut          Transient period to discard [s].
     * @param tend          Total simulation time [s].
     * @param tau           Permittivity time constant for the slow variable z [s].
     * @param noise_sigma   Standard deviation of additive Gaussian noise.
     * @param initial_state Flat initial state [x0..xN-1, z0..zN-1] (length 2N).
     * @param weights       N×N inter-regional structural connectivity matrix.
     * @param fix_seed      RNG seed flag (0 = random, non-zero = fixed).
     * @param method        Integration scheme: "euler" or "heun".
     */
    VEP(
        double G,
        dim1 iext,
        dim1 eta,
        double dt,
        double tcut,
        double tend,
        double tau,
        double noise_sigma,
        dim1 initial_state,
        dim2 weights,
        int fix_seed,
        string method)
    {
        this->G = G;
        this->dt = dt;
        this->nn = weights.size();
        this->iext = iext;
        this->eta = eta;
        this->tcut = tcut;
        this->tend = tend;
        this->method = method;
        this->fix_seed = fix_seed;
        this->weights = weights;
        this->inv_tau = 1.0 / tau;
        this->noise_sigma = noise_sigma;
        this->initial_state = initial_state;
        adjlist = adjmat_to_adjlist(weights);

        unsigned idx_cut = (int)(tcut / dt);
        num_steps = (int)(tend / dt);
        unsigned bufsize = num_steps - idx_cut;

        states.resize(bufsize);
        for (unsigned i = 0; i < bufsize; i++)
        {
            states[i].resize(nn);
        }
        times.resize(bufsize);
    };

    /**
     * @brief Evaluate the VEP right-hand side.
     *
     * For node i:
     * ```
     *   dx/dt = 1 - x[i]^3 - 2*x[i]^2 - z[i] + iext[i]
     *   dz/dt = inv_tau * (4*(x[i] - eta[i]) - z[i] - G * gx[i])
     * ```
     * where `gx[i] = sum_j weights[i][j] * (x[j] - x[i])` is the
     * diffusive inter-regional coupling through the slow variable.
     *
     * @param x     Current state vector [x0..xN-1, z0..zN-1].
     * @param dxdt  Output derivative vector (pre-allocated, length 2N).
     * @param t     Current time [s] (unused; kept for API consistency).
     */
    void rhs(const vector<double> &x, vector<double> &dxdt, const double t)
    {
        (void)t; // Mark as intentionally unused
        for (int i = 0; i < nn; i++)
        {
            double gx = 0.0;
            for (size_t j = 0; j < adjlist[i].size(); ++j)
            {
                int k = adjlist[i][j];
                gx += weights[i][k] * (x[k] - x[i]);
            }
            dxdt[i] = 1.0 - x[i] * x[i] * x[i] - 2.0 * x[i] * x[i] - x[i + nn] + iext[i];
            dxdt[i + nn] = inv_tau * (4.0 * (x[i] - eta[i]) - x[i + nn] - G * gx);
        }
    }

    /**
     * @brief Perform one Euler-Maruyama step, updating @p x in-place.
     *
     * Noise (scale sqrt(dt) * noise_sigma) is applied to all 2N state
     * variables.
     *
     * @param x  State vector (length 2N), modified in-place.
     * @param t  Current time [s].
     */
    void euler_step(vector<double> &x, const double t)
    {
        std::normal_distribution<> normal(0, 1);
        vector<double> dxdt(nn * 2);

        rhs(x, dxdt, t);
        for (int i = 0; i < nn * 2; i++)
        {
            x[i] += dt * dxdt[i] + sqrt(dt) * noise_sigma * normal(rng(fix_seed));
        }
    }

    /**
     * @brief Perform one Heun SDE step, updating @p x in-place.
     *
     * Noise is applied to all 2N state variables in both the predictor and
     * the corrector passes.
     *
     * @param x  State vector (length 2N), modified in-place.
     * @param t  Current time [s].
     */
    void heun_step(vector<double> &x, const double t)
    {
        std::normal_distribution<> normal(0, 1);
        vector<double> dxdt(nn * 2);
        vector<double> xtemp(nn * 2);

        rhs(x, dxdt, t);
        for (int i = 0; i < nn * 2; i++)
        {
            xtemp[i] = x[i] + dt * dxdt[i] + sqrt(dt) * noise_sigma * normal(rng(fix_seed));
        }

        vector<double> dxdt_temp(nn * 2);
        rhs(xtemp, dxdt_temp, t + dt);
        for (int i = 0; i < nn * 2; i++)
        {
            x[i] += 0.5 * dt * (dxdt[i] + dxdt_temp[i]) + sqrt(dt) * noise_sigma * normal(rng(fix_seed));
        }
    }

    /**
     * @brief Run the full simulation.
     *
     * Iterates for @c num_steps steps, discards the transient (tcut), and
     * records the fast variable x (length N) and time at each post-transient
     * step.  The integration method is selected by the @p method field set
     * at construction.
     *
     * @throws std::invalid_argument if the method is not "euler" or "heun".
     */
    void integrate()
    {
        dim1 x = initial_state;
        int idxtcut = (int)(tcut / dt);
        double t = 0.0;
        int counter = 0;

        for (unsigned it = 0; it < num_steps; it++)
        {
            if (it >= static_cast<unsigned>(idxtcut))
            {
                for (int i = 0; i < nn; i++)
                {
                    states[counter][i] = x[i];
                }
                times[counter] = t;
                counter++;
            }

            t += dt;
            if (method == "euler")
                euler_step(x, t);
            else if (method == "heun")
                heun_step(x, t);
            else
                throw std::invalid_argument("Invalid method");
        }
    }

    /**
     * @brief Return the recorded post-transient fast-variable (x) time series.
     * @return dim2 of shape n_record × N.
     */
    dim2 get_states()
    {
        return states;
    }

    /**
     * @brief Return the recorded post-transient time array.
     * @return dim1 of length n_record = (tend - tcut) / dt.
     */
    dim1 get_times()
    {
        return times;
    }
};

#endif
