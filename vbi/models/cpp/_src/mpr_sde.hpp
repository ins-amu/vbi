/**
 * @file mpr_sde.hpp
 * @brief Montbrió-Pazó-Roxin (MPR) mean-field neural mass model with
 *        stochastic noise and optional BOLD hemodynamic output.
 *
 * Implements the exact mean-field reduction of a network of quadratic
 * integrate-and-fire (QIF) neurons derived by Montbrió et al. (2015).
 * The two collective variables per node are:
 *   - r : population-mean firing rate [spikes/s]
 *   - v : population-mean membrane potential [mV]
 *
 * Optionally computes the BOLD (fMRI) signal via the Balloon-Windkessel
 * hemodynamic model (Friston et al., 2000; Stephan et al., 2007).
 *
 * **Model references:**
 * - Montbrió, E., Pazó, D., & Roxin, A. (2015). Macroscopic description for
 *   networks of spiking neurons. *Physical Review X*, 5(2), 021028.
 *   https://doi.org/10.1103/PhysRevX.5.021028
 * - Friston, K.J. et al. (2000). Nonlinear responses in fMRI: the Balloon
 *   model, Volterra kernels, and other hemodynamics.
 *   *NeuroImage*, 12(4), 466–477.
 *
 * **State-vector layout** (flat vector of length 2×N):
 * | Index range | Variable | Description |
 * |-------------|----------|-------------|
 * | [0,   N)    | r        | Mean firing rate per node [spikes/s] |
 * | [N,  2N)    | v        | Mean membrane potential per node [mV] |
 */

#ifndef MPR_SDE_HPP
#define MPR_SDE_HPP

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
 * @brief Parameters of the Balloon-Windkessel hemodynamic model.
 *
 * Default values are from Friston et al. (2000) / Stephan et al. (2007).
 * K1, K2, K3 are derived quantities computed from the other parameters.
 *
 * @see MPR_sde::bold_step
 */
struct BoldParams
{
    double kappa   = 0.65;  ///< Rate of signal decay [1/s]
    double gamma   = 0.41;  ///< Rate of flow-dependent elimination [1/s]
    double tau     = 0.98;  ///< Haemodynamic transit time [s]
    double alpha   = 0.32;  ///< Grubb's vessel stiffness exponent
    double epsilon = 0.34;  ///< Ratio of intra- to extra-vascular signal
    double Eo      = 0.4;   ///< Resting oxygen extraction fraction
    double TE      = 0.04;  ///< Echo time [s]
    double vo      = 0.08;  ///< Resting blood volume fraction
    double r0      = 25.0;  ///< Slope of intravascular relaxation rate [1/s]
    double theta0  = 40.3;  ///< Frequency offset at the outer surface of vessels [Hz]
    double rtol    = 1e-5;  ///< Relative tolerance (reserved for adaptive integration)
    double atol    = 1e-8;  ///< Absolute tolerance (reserved for adaptive integration)
    double dt_b    = 0.001; ///< BOLD integration sub-step [s]
    double K1      = 4.3 * theta0 * Eo * TE;   ///< BOLD signal coefficient K1 (derived)
    double K2      = epsilon * r0 * Eo * TE;    ///< BOLD signal coefficient K2 (derived)
    double K3      = 1 - epsilon;               ///< BOLD signal coefficient K3 (derived)
    double ialpha  = 1 / alpha;                 ///< 1/alpha (precomputed)

    /** @brief Default constructor — uses literature default values. */
    BoldParams() = default;

    /**
     * @brief Construct with custom hemodynamic parameters.
     *
     * K1, K2, K3, and ialpha are recomputed automatically.
     *
     * @param kappa   Rate of signal decay [1/s].
     * @param gamma   Rate of flow-dependent elimination [1/s].
     * @param tau     Haemodynamic transit time [s].
     * @param alpha   Grubb's vessel stiffness exponent.
     * @param epsilon Ratio of intra- to extra-vascular signal.
     * @param Eo      Resting oxygen extraction fraction.
     * @param TE      Echo time [s].
     * @param vo      Resting blood volume fraction.
     * @param r0      Slope of intravascular relaxation rate [1/s].
     * @param theta0  Frequency offset at outer vessel surface [Hz].
     * @param rtol    Relative tolerance.
     * @param atol    Absolute tolerance.
     */
    BoldParams(double kappa, double gamma, double tau,
               double alpha, double epsilon, double Eo, double TE,
               double vo, double r0, double theta0, double rtol,
               double atol) : kappa(kappa), gamma(gamma), tau(tau),
                              alpha(alpha), epsilon(epsilon), Eo(Eo),
                              TE(TE), vo(vo), r0(r0), theta0(theta0),
                              rtol(rtol), atol(atol)
    {
        K1 = 4.3 * theta0 * Eo * TE;
        K2 = epsilon * r0 * Eo * TE;
        K3 = 1 - epsilon;
        ialpha = 1 / alpha;
    }
};

/**
 * @brief MPR mean-field model with SDE and optional BOLD output.
 *
 * Integrates the Montbrió-Pazó-Roxin equations for N network nodes using
 * the Heun SDE scheme.  Noise amplitudes for r and v are scaled differently
 * (rNoise = sqrt(2*noise_amp), vNoise = sqrt(4*noise_amp)).
 *
 * Optionally records:
 *   - The r,v time series downsampled by @p rv_decimate.
 *   - The BOLD signal sampled at repetition time @p tr.
 */
class MPR_sde
{

private:
    dim1 delta;      ///< Half-width of Lorentzian heterogeneity (per node) [Hz]
    dim1 tau;        ///< Membrane time constant (per node) [ms]
    dim1 eta;        ///< Mean excitability (per node) [mV]
    dim1 J;          ///< Synaptic weight (per node)
    dim1 i_app;      ///< Applied current (per node)
    double dt;       ///< Neural SDE integration time step [s]
    double dt_b;     ///< BOLD sub-step time [s]
    double G;        ///< Global inter-node coupling strength

    double rNoise;     ///< Noise amplitude for firing-rate variable r
    double vNoise;     ///< Noise amplitude for membrane-potential variable v
    double noise_amp;  ///< Raw noise amplitude parameter

    size_t num_nodes;    ///< Number of network nodes
    size_t num_steps;    ///< Total integration steps = t_end / dt
    size_t rv_decimate;  ///< Decimation factor for r,v recording
    size_t idx_cut;      ///< Step index at which recording starts (= t_cut/dt)
    unsigned RECORD_RV;  ///< Flag: record r,v time series (non-zero = yes)
    unsigned RECORD_BOLD;///< Flag: record BOLD signal (non-zero = yes)

    double t_end;   ///< Total simulation time [s]
    double t_cut;   ///< Transient period to discard [s]
    double tr;      ///< BOLD repetition time (sampling interval) [s]
    vector<vector<unsigned>> adjlist; ///< Sparse adjacency list

    dim2 weights;        ///< N×N inter-node weight matrix
    dim1 t_arr;          ///< Temporary time array
    int fix_seed;        ///< Seed flag forwarded to rng()
    dim1 initial_state;  ///< Flat initial state vector [r0..rN-1, v0..vN-1]
    BoldParams bp;       ///< Balloon-Windkessel hemodynamic parameters

public:
    dim2 bold_d; ///< BOLD signal, shape (n_bold_samples) × N
    dim1 bold_t; ///< BOLD time array, length n_bold_samples
    dim2 r_d;    ///< Firing-rate time series (decimated), shape n_rv × (2N)
    dim1 r_t;    ///< Time array for r_d, length n_rv

    /**
     * @brief Construct an MPR_sde simulator.
     *
     * @param dt          Neural SDE time step [s].
     * @param dt_b        BOLD integration sub-step [s].
     * @param rv_decimate Decimation factor for recording r,v (1 = every step).
     * @param weights     N×N inter-node coupling weight matrix.
     * @param initial_state Flat initial state [r0..rN-1, v0..vN-1] (length 2N).
     * @param delta       Per-node Lorentzian half-width (heterogeneity) [Hz].
     * @param tau         Per-node membrane time constant [ms].
     * @param eta         Per-node mean excitability [mV].
     * @param J           Per-node synaptic weight.
     * @param i_app       Per-node applied current.
     * @param noise_amp   Noise amplitude (rNoise=sqrt(2*noise_amp)*sqrt(dt),
     *                    vNoise=sqrt(4*noise_amp)*sqrt(dt)).
     * @param G           Global coupling strength.
     * @param t_end       Total simulation time [s].
     * @param t_cut       Transient period to discard [s].
     * @param tr          BOLD repetition time (sampling interval) [s].
     * @param RECORD_RV   Non-zero to record r,v time series.
     * @param RECORD_BOLD Non-zero to record BOLD signal.
     * @param fix_seed    RNG seed flag (0 = random, non-zero = fixed).
     * @param bp          Balloon-Windkessel hemodynamic parameters.
     */
    MPR_sde(double dt,
            double dt_b,
            size_t rv_decimate,
            dim2 weights,
            dim1 initial_state,
            dim1 delta,
            dim1 tau,
            dim1 eta,
            dim1 J,
            dim1 i_app,
            double noise_amp,
            double G,
            double t_end,
            double t_cut,
            double tr,
            size_t RECORD_RV,
            size_t RECORD_BOLD,
            int fix_seed,
            const BoldParams &bp) : delta(delta), tau(tau), eta(eta), J(J), i_app(i_app),
                                    dt(dt), dt_b(dt_b), G(G),
                                    noise_amp(noise_amp),
                                    rv_decimate(rv_decimate),
                                    RECORD_RV(RECORD_RV), RECORD_BOLD(RECORD_BOLD),
                                    t_end(t_end), t_cut(t_cut), tr(tr),
                                    weights(weights),
                                    fix_seed(fix_seed), initial_state(initial_state), bp(bp)

    {
        assert(t_end > t_cut && "t_end must be greater than t_cut");
        assert(tr > 0);
        assert(rv_decimate > 0);

        num_nodes = weights.size();
        num_steps = int((t_end) / dt);
        idx_cut = int((t_cut) / dt);
        rNoise = sqrt(dt) * sqrt(2 * noise_amp);
        vNoise = sqrt(dt) * sqrt(4 * noise_amp);
        adjlist = adjmat_to_adjlist(weights);
    }

    /**
     * @brief Evaluate the MPR mean-field right-hand side.
     *
     * For node i:
     * ```
     *   dr/dt = (delta[i] / (tau[i]^2 * pi) + 2*r[i]*v[i]) / tau[i]
     *   dv/dt = (v[i]^2 + i_app[i] + eta[i] + J[i]*tau[i]*r[i]
     *            - pi^2*tau[i]^2*r[i]^2 + G*cpl[i]) / tau[i]
     * ```
     * where cpl[i] = sum_j weights[i][j] * r[j].
     *
     * @param x     Current state vector [r0..rN-1, v0..vN-1] (length 2N).
     * @param dxdt  Output derivative vector (length 2N, pre-allocated).
     * @param t     Current time [s] (unused; kept for API consistency).
     */
    void f_mpr(
        const dim1 &x,
        dim1 &dxdt,
        const double t)
    {
        (void)t; // Mark as intentionally unused
        size_t nn = num_nodes;
        double p2 = M_PI * M_PI;

        for (unsigned i = 0; i < nn; i++)
        {
            double cpl = 0;
            for (unsigned j = 0; j < adjlist[i].size(); j++)
            {
                unsigned k = adjlist[i][j];
                cpl += weights[i][k] * x[k];
            }
            dxdt[i] = 1.0 / tau[i] * (delta[i] / (tau[i] * M_PI) + 2 * x[i] * x[i + nn]);
            dxdt[i + nn] = 1.0 / tau[i] * (x[i + nn] * x[i + nn] + i_app[i] + eta[i] + J[i] * tau[i] * x[i] - (p2 * tau[i] * tau[i] * x[i] * x[i]) + G * cpl);
        }
    }

    /**
     * @brief Perform one Heun SDE step for the r,v state vector.
     *
     * Noise amplitudes differ between the r and v components:
     *   - r: rNoise = sqrt(dt * 2 * noise_amp)
     *   - v: vNoise = sqrt(dt * 4 * noise_amp)
     *
     * Firing rate r is clipped to zero if it goes negative (physically
     * meaningless).
     *
     * @param y  State vector [r0..rN-1, v0..vN-1], modified in-place.
     * @param t  Current time [s].
     */
    void heun_step(dim1 &y, const double t)
    {
        std::normal_distribution<> normal(0, 1);

        size_t nn = 2 * num_nodes;
        size_t n = num_nodes;
        dim1 tmp(nn);
        dim1 k1(nn);
        dim1 k2(nn);

        f_mpr(y, k1, t);

        for (size_t i = 0; i < nn; ++i)
            if (i < n)
                tmp[i] = y[i] + dt * k1[i] + rNoise * normal(rng(fix_seed));
            else
                tmp[i] = y[i] + dt * k1[i] + vNoise * normal(rng(fix_seed));

        f_mpr(tmp, k2, t + dt);
        for (size_t i = 0; i < nn; ++i)
        {
            if (i < n)
            {
                y[i] += 0.5 * dt * (k1[i] + k2[i]) + rNoise * normal(rng(fix_seed));
                if (y[i] < 0)
                    y[i] = 0.0;
            }
            else
                y[i] += 0.5 * dt * (k1[i] + k2[i]) + vNoise * normal(rng(fix_seed));
        }
    }

    /**
     * @brief Perform one Balloon-Windkessel BOLD integration sub-step.
     *
     * Integrates the haemodynamic state variables (vasodilatory signal s,
     * CBF f, log-CBF ftilde, log-CBV vtilde, log-dHb qtilde, CBV v, dHb q)
     * using a forward-Euler step of size @p dtt.
     *
     * @param r_in   Firing-rate input to the BOLD model (length N).
     * @param s      Vasodilatory signal (2 × N ping-pong buffer).
     * @param f      Cerebral blood flow (2 × N).
     * @param ftilde Log-normalised CBF (2 × N).
     * @param vtilde Log-normalised CBV (2 × N).
     * @param qtilde Log-normalised deoxyhaemoglobin (2 × N).
     * @param v      Cerebral blood volume (2 × N).
     * @param q      Deoxyhaemoglobin content (2 × N).
     * @param dtt    Sub-step size [s].
     */
    void bold_step(
        const dim1 &r_in,
        dim2 &s,
        dim2 &f,
        dim2 &ftilde,
        dim2 &vtilde,
        dim2 &qtilde,
        dim2 &v,
        dim2 &q,
        const double dtt)
    {
        unsigned n = num_nodes;
        // double dtt = dt_b;
        // dim1 fv(n, 0.0);
        // dim1 ff(n, 0.0);

        for (unsigned i = 0; i < n; i++)
        {
            s[1][i] = s[0][i] + dtt * (r_in[i] - bp.kappa * s[0][i] - bp.gamma * (f[0][i] - 1));
            f[0][i] = std::max(f[0][i], 1.0);
            ftilde[1][i] = ftilde[0][i] + dtt * (s[0][i] / f[0][i]);
            double fv = pow(v[0][i], bp.ialpha);
            vtilde[1][i] = vtilde[0][i] + dtt * ((f[0][i] - fv) / (bp.tau * v[0][i]));
            q[0][i] = std::max(q[0][i], 0.01);
            double ff = (1 - pow((1 - bp.Eo), 1.0 / f[0][i])) / bp.Eo;
            qtilde[1][i] = qtilde[0][i] + dtt * ((f[0][i] * ff - fv * q[0][i] / v[0][i]) / (bp.tau * q[0][i]));
            f[1][i] = exp(ftilde[1][i]);
            v[1][i] = exp(vtilde[1][i]);
            q[1][i] = exp(qtilde[1][i]);
            f[0][i] = f[1][i];
            s[0][i] = s[1][i];
            ftilde[0][i] = ftilde[1][i];
            vtilde[0][i] = vtilde[1][i];
            qtilde[0][i] = qtilde[1][i];
            v[0][i] = v[1][i];
            q[0][i] = q[1][i];
        }
    }

    /**
     * @brief Run the full simulation.
     *
     * Advances the neural state with Heun SDE steps and, if RECORD_BOLD is
     * set, drives the Balloon-Windkessel model to produce a BOLD signal
     * sampled every @p tr seconds.  r,v data are recorded every @p rv_decimate
     * steps (if RECORD_RV is set).
     *
     * Output is stored in the public members bold_d, bold_t, r_d, r_t.
     */
    void integrate()
    {
        unsigned n = num_nodes;
        double r_period = dt * 10; // we extend time 10 times
        unsigned b_decimate = (int)(std::round(tr / r_period));
        double dtt = r_period / 1000.0; // in seconds

        size_t nt = (int)(t_end / dt);
        dim1 rv_current = initial_state;

        if (RECORD_RV)
        {
            r_d.resize((int)(nt / rv_decimate), dim1(2 * n, 0.0));
            r_t.resize((int)(nt / rv_decimate), 0.0);
        }

        dim2 s(2, dim1(n, 0.0));
        dim2 f(2, dim1(n, 0.0));
        dim2 ftilde(2, dim1(n, 0.0));
        dim2 vtilde(2, dim1(n, 0.0));
        dim2 qtilde(2, dim1(n, 0.0));
        dim2 v(2, dim1(n, 0.0));
        dim2 q(2, dim1(n, 0.0));
        dim2 vv((int(nt / b_decimate)), dim1(n, 0.0));
        dim2 qq((int(nt / b_decimate)), dim1(n, 0.0));

        if (RECORD_BOLD)
        {
            bold_d.resize((int)(nt / b_decimate), dim1(n, 0.0));
            bold_t.resize((int)(nt / b_decimate), 0.0);
        }

        s[0] = dim1(n, 1.0);
        f[0] = dim1(n, 1.0);
        v[0] = dim1(n, 1.0);
        q[0] = dim1(n, 1.0);

        for (unsigned itr = 0; itr < nt - 1; ++itr)
        {
            double t_current = itr * dt;
            heun_step(rv_current, t_current);

            if (RECORD_RV)
            {
                if (((itr % rv_decimate) == 0) && ((itr / rv_decimate) < r_d.size()))
                {
                    unsigned idx = itr / rv_decimate;
                    r_d[idx] = rv_current;
                    r_t[idx] = t_current;
                }
            }
            if (RECORD_BOLD)
            {
                bold_step(rv_current, s, f, ftilde, vtilde, qtilde, v, q, dtt);

                if (((itr % b_decimate) == 0) && ((itr / b_decimate) < bold_d.size()))
                {
                    unsigned idx = itr / b_decimate;
                    vv[idx] = v[1];
                    qq[idx] = q[1];
                    bold_t[idx] = t_current;
                    {
                        if (std::isnan(qq[idx][0]))
                        {
                            std::cout << "nan found! " << "\n";
                            break;
                        }
                    }
                }
            }
        }

        for (unsigned i = 0; i < bold_d.size(); i++)
        {
            for (unsigned j = 0; j < n; ++j)
                bold_d[i][j] = bp.vo * (bp.K1 * (1 - qq[i][j]) + bp.K2 * (1 - qq[i][j] / vv[i][j]) + bp.K3 * (1 - vv[i][j]));
        }
    }

    /** @brief Return the BOLD signal time series (shape n_bold × N). */
    dim2 get_bold_d()
    {
        return bold_d;
    }
    /** @brief Return the BOLD sampling time array (length n_bold). */
    dim1 get_bold_t()
    {
        return bold_t;
    }

    /** @brief Return the decimated r,v time series (shape n_rv × 2N). */
    dim2 get_r_d()
    {
        return r_d;
    }
    /** @brief Return the time array for the r,v recording (length n_rv). */
    dim1 get_r_t()
    {
        return r_t;
    }
};

#endif
