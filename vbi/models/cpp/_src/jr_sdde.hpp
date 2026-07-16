/**
 * @file jr_sdde.hpp
 * @brief Jansen-Rit neural mass model with stochastic noise and axonal
 *        conduction delays (SDDE — Stochastic Delay Differential Equations).
 *
 * Implements the Jansen-Rit (JR) model for a network of N cortical columns
 * coupled via delayed, weighted connections.  Each column is described by
 * three second-order linear differential operators (excitatory pyramidal,
 * excitatory interneuron, inhibitory interneuron) driven by a sigmoidal
 * firing-rate function.
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
 * | [N,  2N)    | x1       | EPSP  — pyramidal cell output |
 * | [2N, 3N)    | x2       | Fast IPSP — inhibitory interneuron output |
 * | [3N, 4N)    | ẋ0       | Velocity of x0 |
 * | [4N, 5N)    | ẋ1       | Velocity of x1 — **noise applied here** |
 * | [5N, 6N)    | ẋ2       | Velocity of x2 |
 *
 * The EEG proxy (pyramidal-cell PSP) is: `x1 - x2` = `y[i+N] - y[i+2N]`.
 */

#ifndef JANSENRIT_HPP
#define JANSENRIT_HPP

#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <assert.h>
#include <iostream>
#include "utility.hpp"

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1> dim2;
typedef std::vector<float> dim1f;
typedef std::vector<dim1f> dim2f;

/**
 * @brief Jansen-Rit SDDE — N coupled cortical columns with conduction delays.
 *
 * Noise enters only through the ẋ1 (excitatory pyramidal velocity) component,
 * modelling stochastic afferent input.  Integration uses either the
 * Euler-Maruyama or Heun (order-1.5 strong) scheme.
 */
class JR_sdde
{

private:
    int N;              ///< Number of network nodes
    dim1 y0;            ///< Initial state (history buffer seed)
    double dt;          ///< Integration time step [s]
    int dimension;      ///< State-space dimension per node (= 6)
    int num_nodes;      ///< Alias for N
    double maxdelay;    ///< Maximum axonal delay across all edges [s]
    size_t fix_seed;    ///< Seed flag forwarded to rng()
    double coupling;    ///< Global coupling strength G
    double par_vmax;    ///< Sigmoid saturation firing rate [Hz]
    dim2 states;        ///< Recorded observable (x1-x2), shape N × n_record

    dim1 C0; ///< Intra-column connectivity scaling for slow inhibitory population
    dim1 C1; ///< Intra-column connectivity scaling for pyramidal population
    dim1 C2; ///< Intra-column connectivity scaling for fast inhibitory population
    dim1 C3; ///< Intra-column connectivity scaling for fast inhibitory feedback

    double noise_sigma;  ///< Standard deviation of additive white noise
    double noise_mu;     ///< Mean of additive Gaussian input (background drive)
    double par_A;        ///< Excitatory synaptic gain [mV]
    double par_a;        ///< Excitatory synaptic time constant [1/s]
    double par_B;        ///< Inhibitory synaptic gain [mV]
    double par_b;        ///< Inhibitory synaptic time constant [1/s]
    double par_r;        ///< Sigmoid slope parameter [1/mV]
    double par_v0;       ///< Sigmoid half-maximum potential [mV]

    dim1 sti_amp;           ///< Per-node stimulation amplitude vector
    double sti_ti;          ///< Stimulation onset time [s]
    double sti_duration;    ///< Stimulation duration [s]
    double sti_gain = 0.0;  ///< Stimulation gain (0 = no stimulation)
    double _sti_gain = 0.0; ///< Active stimulation gain at current time step

    double t_final;             ///< Total simulation time [s]
    double t_transition;        ///< Transient period to discard [s]
    size_t index_transition;    ///< Time-step index corresponding to t_transition
    long unsigned num_iteration;///< Total number of integration steps

    vector<vector<unsigned>> adjlist; ///< Sparse adjacency list (from adjmat)
    vector<vector<unsigned>> D;       ///< Delay indices D[i][j] = round(delays[i][j]/dt)

    dim1 t_ar;   ///< Full time array (length num_iteration)
    dim1 t_arr;  ///< Recorded time array (post-transient)
    dim2 y;      ///< Full state history, shape (6*N) × num_iteration
    dim2 adj;    ///< Adjacency/weight matrix, shape N×N
    dim2 delays; ///< Axonal delay matrix [s], shape N×N

public:
    int nstart;       ///< Number of warm-up steps to initialise delay buffer
    dim1 sti_vector;  ///< Recorded stimulation gain time series

    /**
     * @brief Construct a JR_sdde simulator.
     *
     * @param dt            Integration time step [s].
     * @param y0            Initial state vector (length 6*N); used to fill
     *                      the delay history buffer.
     * @param adj           N×N weight matrix of inter-column coupling strengths.
     * @param delays        N×N matrix of axonal conduction delays [s].
     * @param coupling      Global coupling scaling factor G.
     * @param dimension     State-space dimension (must be 6).
     * @param A             Excitatory synaptic gain [mV].
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
     * @param sti_amp       Per-node stimulation amplitude (length N).
     * @param sti_gain      Stimulation gain scalar (0 = off).
     * @param sti_ti        Stimulation onset time [s] (must be ≥ t_transition).
     * @param sti_duration  Stimulation duration [s].
     * @param noise_mu      Mean of Gaussian input (background drive) [Hz].
     * @param noise_sigma   Standard deviation of additive noise.
     * @param t_transition  Transient period to discard [s] (default 1.0).
     * @param t_final       Total simulation time [s] (default 10.0).
     * @param noise_seed    0 = random seed; non-zero = fixed seed for reproducibility.
     */
    JR_sdde(double dt,
            dim1 y0,
            dim2 adj,
            dim2 delays,
            double coupling,
            int dimension,
            double A,
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
            dim1 sti_amp,
            double sti_gain,
            double sti_ti,
            double sti_duration,
            double noise_mu,
            double noise_sigma,
            double t_transition = 1.0,
            double t_final = 10.0,
            size_t noise_seed = 0)
    {
        N = num_nodes = adj.size();
        this->dimension = dimension;

        assert(t_final > t_transition);

        this->noise_mu = noise_mu;
        this->noise_sigma = noise_sigma;

        par_A = A;
        par_a = a;
        par_B = B;
        par_b = b;
        par_r = r;
        par_v0 = v0;
        par_vmax = vmax;

        this->dt = dt;
        this->adj = adj;
        this->delays = delays;
        this->t_final = t_final;
        this->fix_seed = noise_seed;
        this->coupling = coupling;
        this->t_transition = t_transition;
        this->C0 = C0;
        this->C1 = C1;
        this->C2 = C2;
        this->C3 = C3;

        assert(y0.size() == (dimension * N));
        prepare_sti(sti_amp, sti_gain, sti_ti, sti_duration);

        {
            maxdelay = 0.0;
            dim1 tmp(N);
            for (size_t i = 0; i < N; ++i)
                tmp[i] = *std::max_element(delays[i].begin(), delays[i].end());
            maxdelay = *std::max_element(tmp.begin(), tmp.end());
        }
        nstart = (std::abs(maxdelay) > dt) ? int(ceil(maxdelay / dt)) : 50;
        num_iteration = int(ceil(t_final / dt)) + nstart;
        index_transition = int(round(t_transition / dt));

        assert((index_transition) < num_iteration); // make sure the simulation is long enough

        // memory allocations -------------------------------------------------
        D.resize(N);
        states.resize(N);
        y.resize(dimension * N);
        t_ar.resize(num_iteration);

        for (size_t i = 0; i < N; ++i)
        {
            D[i].resize(N);
            states[i].resize(num_iteration - index_transition);
        }
        for (int i = 0; i < dimension * N; ++i)
            y[i].resize(num_iteration);

        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                D[i][j] = int(round(delays[i][j] / dt)); // delay indices
        // --------------------------------------------------------------------
        adjlist = adjmat_to_adjlist(adj);

        set_history(y0);
    }

    /**
     * @brief Fill the delay history buffer with a constant initial state.
     *
     * Sets `y[i][j] = hist[i]` for all variables i and all warm-up steps
     * j ∈ [0, nstart], and builds the time array t_ar.
     *
     * @param hist  Constant initial state vector (length 6*N).
     */
    void set_history(const dim1 &hist)
    {
        for (int i = 0; i < num_iteration; ++i)
            t_ar[i] = i * dt;

        // p_x_ar: N x nstart
        for (int i = 0; i < (dimension * N); ++i)
            for (int j = 0; j < nstart + 1; ++j)
                y[i][j] = hist[i];
    }

    /**
     * @brief Validate and store stimulation parameters.
     *
     * If @p sti_gain is effectively zero the stimulation is disabled.
     * Otherwise the amplitude vector size is checked, and the onset must
     * fall after the transient period.
     *
     * @param sti_amp       Per-node amplitude vector (length N).
     * @param sti_gain      Stimulation gain scalar.
     * @param sti_ti        Onset time [s].
     * @param sti_duration  Duration [s].
     */
    void prepare_sti(const dim1 sti_amp, double sti_gain, double sti_ti, double sti_duration)
    {
        if ((sti_amp.size() != N) && (std::abs(sti_gain) > 0.0))
        {
            std::cout << "Stimulation amplitude vector size is not equal to the number of nodes" << std::endl;
            exit(1);
        }
        else if (std::abs(sti_gain) > 1e-10)
        {
            this->sti_amp = sti_amp;
            this->sti_gain = sti_gain;
            this->sti_ti = sti_ti;
            this->sti_duration = sti_duration;
            // assert((sti_ti + sti_duration) <= t_final); // make sure the stimulation duration is not longer than the simulation duration
            assert(sti_ti >= t_transition); // make sure the stimulation starts after the transition period
        }
        else // no stimulation
        {
            this->sti_amp.resize(N);
            this->sti_gain = 0.0;
            this->sti_ti = 0.0;
            this->sti_duration = 0.0;
        }
    }

    /**
     * @brief Sigmoid (firing-rate) transfer function.
     *
     * Converts membrane potential @p v to firing rate:
     * `sigma(v) = vmax / (1 + exp(r * (v0 - v)))`
     *
     * @param v  Membrane potential [mV].
     * @return   Firing rate [Hz].
     */
    double _sigma(const double v)
    {
        return par_vmax / (1 + exp(par_r * (par_v0 - v)));
    }

    /**
     * @brief Evaluate the right-hand side of the JR-SDDE system at time step n.
     *
     * Computes the deterministic drift for all 6*N state variables.  The
     * inter-column coupling term uses delayed values `y[k][n - D[i][k]]`.
     *
     * @param t  Current simulation time [s] (unused directly; kept for API consistency).
     * @param n  Current time-step index into the history array.
     * @return   Derivative vector dxdt of length 6*N.
     */
    dim1 f_sys(const double t,
               const unsigned n)
    {
        dim1 dxdt(dimension * N);

        double a2 = par_a * par_a;
        double b2 = par_b * par_b;
        double Aa = par_A * par_a;
        double Bb = par_B * par_b;

        int N2 = 2 * N;
        int N3 = 3 * N;
        int N4 = 4 * N;
        int N5 = 5 * N;

        for (size_t i = 0; i < N; ++i)
        {
            double coupling_term = 0.0;

            for (size_t j = 0; j < adjlist[i].size(); ++j)
            {
                int k = adjlist[i][j];
                coupling_term += adj[i][k] * _sigma(y[k + N][n - D[i][k]] - y[k + N2][n - D[i][k]]);
            }

            dxdt[i] = y[i + N3][n];
            dxdt[i + N] = y[i + N4][n];
            dxdt[i + N2] = y[i + N5][n];
            dxdt[i + N3] = Aa * _sigma(y[i + N][n] - y[i + N2][n]) - 2 * par_a * y[i + N3][n] - a2 * y[i][n];
            dxdt[i + N4] = Aa * (noise_mu + _sti_gain * sti_amp[i] + C1[i] * _sigma(C0[i] * y[i][n]) + coupling * coupling_term) - 2 * par_a * y[i + N4][n] - a2 * y[i + N][n];
            dxdt[i + N5] = Bb * C3[i] * _sigma(C2[i] * y[i][n]) - 2 * par_b * y[i + N5][n] - b2 * y[i + N2][n];
        }

        return dxdt;
    }

    /**
     * @brief Perform one Euler-Maruyama step.
     *
     * Additive white noise (scaled by sqrt(dt) * noise_sigma) is applied
     * only to the ẋ1 components (indices [4N, 5N)).
     *
     * @param t  Current time [s].
     * @param n  Current time-step index.
     */
    void euler(const double t, const unsigned n)
    {
        std::normal_distribution<> normal(0, 1);
        size_t nc = dimension * num_nodes;
        dim1 dy(nc);

        double coeff = sqrt(dt) * noise_sigma;

        dy = f_sys(t, n);
        for (int i = 0; i < (N * dimension); ++i)
        {
            if ((i >= 4 * N) && (i < 5 * N))
                y[i][n + 1] = y[i][n] + dt * dy[i] + coeff * normal(rng(fix_seed));
            else
                y[i][n + 1] = y[i][n] + dt * dy[i];
        }
    }

    /**
     * @brief Perform one Heun (predictor-corrector) SDE step.
     *
     * Implements the Heun scheme for Itô SDEs:
     *   1. Predictor:  ỹ = y_n + dt*f(t,n) + coeff*dW
     *   2. Corrector:  y_{n+1} = y_n + dt/2*(f(t,n) + f(t+dt,ñ)) + coeff*dW
     *
     * Noise is applied only to the ẋ1 components.
     *
     * @param t  Current time [s].
     * @param n  Current time-step index.
     */
    void heun(const double t, const unsigned n)
    {
        std::normal_distribution<> normal(0, 1);
        size_t nc = dimension * num_nodes;
        dim1 k1(nc);
        dim1 k2(nc);

        double coeff = sqrt(dt) * noise_sigma;
        double half_dt = 0.5 * dt;

        k1 = f_sys(t, n);
        for (int i = 0; i < (N * dimension); ++i)
        {
            if ((i >= 4 * N) && (i < 5 * N))
                y[i][n + 1] = y[i][n] + dt * k1[i] + coeff * normal(rng(fix_seed));
            else
                y[i][n + 1] = y[i][n] + dt * k1[i];
        }

        k2 = f_sys(t + dt, n + 1);
        for (int i = 0; i < (N * dimension); ++i)
        {
            if ((i >= 4 * N) && (i < 5 * N))
                y[i][n + 1] = y[i][n] + half_dt * (k1[i] + k2[i]) + coeff * normal(rng(fix_seed));
            else
                y[i][n + 1] = y[i][n] + half_dt * (k1[i] + k2[i]);
        }
    }

    /**
     * @brief Run the full simulation.
     *
     * Iterates from step @c nstart to @c num_iteration-1.  Stimulation is
     * toggled on/off based on the simulation time relative to sti_ti and
     * sti_duration.  States (x1 - x2) and times are recorded only after the
     * transient period (index_transition).
     *
     * @param method  Integration scheme: "euler" (Euler-Maruyama) or "heun".
     * @throws std::invalid_argument if @p method is not recognised.
     */
    void integrate(const std::string method)
    {
        std::normal_distribution<> normal(0, 1);
        size_t nc = dimension * num_nodes;
        unsigned counter = 0;
        unsigned N2 = 2 * N;
        t_arr.resize(num_iteration - index_transition);
        sti_vector.resize(t_arr.size());

        for (unsigned it = nstart; it < num_iteration - 1; ++it)
        {
            double t = (it - nstart + 1) * dt;

            // stimulation
            if ((t >= sti_ti) && (t <= (sti_ti + sti_duration)))
                _sti_gain = sti_gain;
            else
                _sti_gain = 0.0;

            if (method == "euler")
                euler(t, it);
            else if (method == "heun")
                heun(t, it);
            else
            {
                throw std::invalid_argument("Invalid integration method");
                exit(1);
            }

            if (it >= (index_transition))
            {
                t_arr[counter] = t;
                sti_vector[counter] = _sti_gain;
                for (int j = 0; j < N; ++j)
                    states[j][counter] = y[j + N][it + 1] - y[j + N2][it + 1];
                counter++;
            }
        }
    }

    /**
     * @brief Return the recorded post-transient time array.
     * @return dim1 of length (t_final - t_transition) / dt.
     */
    dim1 get_t()
    {
        return t_arr;
    }

    /**
     * @brief Return the recorded EEG proxy (x1 - x2) for all nodes.
     * @return dim2 of shape N × n_record, where n_record = (t_final - t_transition)/dt.
     */
    dim2 get_y()
    {
        return states;
    }

    /**
     * @brief Return the recorded stimulation gain time series.
     * @return dim1 of length n_record.
     */
    dim1 get_sti_vector()
    {
        return sti_vector;
    }
};

#endif
