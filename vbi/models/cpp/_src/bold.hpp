/**
 * @file bold.hpp
 * @brief Balloon-Windkessel hemodynamic models for converting neural activity
 *        to BOLD (fMRI) signals.
 *
 * Provides two variants of the Balloon-Windkessel model:
 *   - **BOLD_2D**: 2-state model (vasodilatory signal s, CBF f); a simplified
 *     linearised approximation suitable for fast forward simulation.
 *   - **BOLD_4D**: 4-state model (s, f, CBV v, dHb q); the full nonlinear
 *     Balloon model for higher-fidelity BOLD predictions.
 *
 * **Model references:**
 * - Buxton, R.B. et al. (1998). Dynamics of blood flow and oxygenation changes
 *   during brain activation: the Balloon model. *MRM*, 39(6), 855–864.
 * - Friston, K.J. et al. (2000). Nonlinear responses in fMRI: the Balloon
 *   model, Volterra kernels, and other hemodynamics.
 *   *NeuroImage*, 12(4), 466–477.
 * - Stephan, K.E. et al. (2007). Comparing hemodynamic models with DCM.
 *   *NeuroImage*, 38(3), 387–401.
 *
 * Both classes expose a single-step `integrate()` method that advances the
 * hemodynamic state by one time step @p dt_b and returns the BOLD signal
 * contribution for that step.  They are designed to be called from within a
 * neural simulation loop.
 */

#ifndef BOLD_HPP
#define BOLD_HPP

#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <assert.h>
#include <iostream>
#include "utility.hpp"

typedef std::vector<double> dim1;
typedef std::vector<dim1> dim2;
typedef std::vector<float> dim1f;
typedef std::vector<dim1f> dim2f;

using std::string;
using std::vector;

/**
 * @brief Simplified 2-state Balloon-Windkessel BOLD model.
 *
 * State vector per node (2×N flat vector):
 * | Index range | Variable | Description |
 * |-------------|----------|-------------|
 * | [0,   N)    | s        | Vasodilatory signal |
 * | [N,  2N)    | f        | Log-normalised cerebral blood flow (CBF) |
 *
 * The BOLD output is a linear function of (f - 1).
 *
 * Supports Euler (`eulerDeterministic`) and Heun (`heunDeterministic`)
 * integration methods, selected by the @p integration_method string.
 */
class BOLD_2D
{
private:
    size_t N;         ///< Number of network nodes
    double dt_b;      ///< Integration time step [s]
    double PAR_rho;   ///< Resting oxygen extraction fraction ρ
    double PAR_e;     ///< Neural efficacy ε (scales neural→vascular drive)
    double PAR_taus;  ///< Signal decay time constant τ_s [s]
    double PAR_tauf;  ///< Auto-regulation time constant τ_f [s]
    double PAR_k1;    ///< BOLD signal scaling coefficient k1
    double PAR_eps;   ///< Ratio of intra- to extra-vascular signal ε
    double inv_taus;  ///< 1 / τ_s (precomputed)
    double inv_tauf;  ///< 1 / τ_f (precomputed)
    std::string integration_method; ///< "eulerDeterministic" or "heunDeterministic"

public:
    /**
     * @brief Construct a BOLD_2D model.
     *
     * @param N                  Number of network nodes.
     * @param dt                 Integration time step [s].
     * @param integration_method "eulerDeterministic" or "heunDeterministic"
     *                           (default "heunDeterministic").
     * @param rho                Resting oxygen extraction fraction (default 0.8).
     * @param e                  Neural efficacy (default 0.02).
     * @param taus               Signal decay time constant [s] (default 0.8).
     * @param tauf               Auto-regulation time constant [s] (default 0.4).
     * @param k1                 BOLD scaling coefficient (default 5.6).
     * @param eps                Intra/extra-vascular ratio (default 0.5).
     */
    BOLD_2D(
        size_t N,
        double dt,
        std::string integration_method = "heunDeterministic",
        double rho = 0.8,
        double e = 0.02,
        double taus = 0.8,
        double tauf = 0.4,
        double k1 = 5.6,
        double eps = 0.5) : N(N),
                            dt_b(dt),
                            PAR_e(e),
                            PAR_k1(k1),
                            PAR_eps(eps),
                            PAR_rho(rho),
                            PAR_taus(taus),
                            PAR_tauf(tauf)

    {
        assert(N != 0);
        inv_taus = 1.0 / PAR_taus;
        inv_tauf = 1.0 / PAR_tauf;

        this->integration_method = integration_method;
    }

    /**
     * @brief Evaluate the 2-state BOLD right-hand side.
     *
     * For each node i:
     * ```
     *   ds/dt  = eps * vec_in[i + index] - s[i] / taus - (f[i] - 1) / tauf
     *   df/dt  = s[i]
     * ```
     * where @p index selects which component of @p vec_in drives the model
     * (0 for r, num_nodes for v in MPR-type models).
     *
     * @param x       Current state [s0..sN-1, f0..fN-1] (length 2N).
     * @param dxdt    Output derivative (length 2N, pre-allocated).
     * @param t       Current time [s] (unused).
     * @param vec_in  Neural activity vector driving the BOLD model.
     * @param index   Offset into @p vec_in selecting the relevant component.
     * @return        Reference to @p dxdt.
     */
    dim1 bold_derivate(const dim1 &x,
                       dim1 &dxdt,
                       const double t,
                       const dim1 &vec_in,
                       const size_t index)
    {

        for (size_t i = 0; i < N; ++i)
        {
            dxdt[i] = vec_in[i + index] * PAR_eps - inv_taus * x[i] - inv_tauf * (x[i + N] - 1.0);
            dxdt[i + N] = x[i];
        }
        return dxdt;
    }

    /**
     * @brief Advance the BOLD state by one Euler step.
     *
     * @param y0        Current state [s, f] (length 2N), modified in-place.
     * @param t         Current time [s].
     * @param vec_in    Neural activity vector.
     * @param num_nodes Number of nodes (= N).
     * @param index     Offset into @p vec_in.
     */
    void eulerDeterministic(dim1 &y0,
                            const double t,
                            const dim1 &vec_in,
                            const size_t num_nodes,
                            const size_t index)
    {
        size_t n = y0.size();
        dim1 dydt(n);
        bold_derivate(y0, dydt, t, vec_in, index);

        for (size_t i = 0; i < n; ++i)
            y0[i] += dydt[i] * dt_b;
    }

    /**
     * @brief Advance the BOLD state by one Heun step.
     *
     * @param y0        Current state [s, f] (length 2N), modified in-place.
     * @param t         Current time [s].
     * @param vec_in    Neural activity vector.
     * @param num_nodes Number of nodes (= N).
     * @param index     Offset into @p vec_in.
     */
    void heunDeterministic(dim1 &y0,
                           const double t,
                           const dim1 &vec_in,
                           const size_t num_nodes,
                           const size_t index)
    {
        size_t n = y0.size();
        dim1 k1(n);
        dim1 k2(n);
        dim1 tmp(n);

        bold_derivate(y0, k1, t, vec_in, index);
        for (size_t i = 0; i < n; ++i)
            tmp[i] = y0[i] + dt_b * k1[i];
        bold_derivate(tmp, k2, t + dt_b, vec_in, index);
        for (size_t i = 0; i < n; ++i)
            y0[i] += 0.5 * dt_b * (k1[i] + k2[i]);
    }

    /**
     * @brief Advance the hemodynamic state and return the BOLD signal.
     *
     * Updates @p y0 in-place and returns the BOLD signal vector:
     * `BOLD[i] = (100 / rho) * e * k1 * (f[i] - 1)`
     *
     * @param y0        Current state (length 2N), modified in-place.
     * @param t         Current time [s].
     * @param vec_in    Neural activity driving the BOLD model.
     * @param num_nodes Number of nodes.
     * @param component "r" (index = 0) or "v" (index = num_nodes).
     * @return          BOLD signal vector (length N).
     */
    dim1 integrate(dim1 &y0,
                   const double t,
                   const dim1 &vec_in,
                   const size_t num_nodes,
                   const std::string component)
    {
        size_t index = (component == "v") ? num_nodes : 0;

        dim1 out(N);

        if (integration_method == "eulerDeterministic")
            eulerDeterministic(y0, t, vec_in, num_nodes, index);

        else if (integration_method == "heunDeterministic")
            heunDeterministic(y0, t, vec_in, num_nodes, index);

        else
        {
            printf("unknow integration method; \n");
            exit(EXIT_FAILURE);
        }

        double coef = (100.0 / PAR_rho) * PAR_e * PAR_k1;

        for (size_t i = 0; i < N; ++i)
            out[i] = coef * (y0[i + N] - 1.0);

        return out;
    }
};

/**
 * @brief Full 4-state nonlinear Balloon-Windkessel BOLD model.
 *
 * State vector per node (4×N flat vector):
 * | Index range | Variable | Description |
 * |-------------|----------|-------------|
 * | [0,   N)    | s        | Vasodilatory signal |
 * | [N,  2N)    | f        | Log-normalised CBF |
 * | [2N, 3N)    | v        | Log-normalised cerebral blood volume (CBV) |
 * | [3N, 4N)    | q        | Log-normalised deoxyhaemoglobin content |
 *
 * The BOLD output is the nonlinear sum:
 * `BOLD = V0 * (K1*(1-q) + K2*(1-q/v) + K3*(1-v))`
 *
 * Supports Euler (`eulerDeterministic`) and Heun (`heunDeterministic`)
 * integration methods.
 */
class BOLD_4D
{

private:
    size_t N;         ///< Number of network nodes
    double dt_b;      ///< Integration time step [s]
    double PAR_alpha; ///< Grubb's vessel stiffness exponent α
    double PAR_tauo;  ///< Haemodynamic transit time τ_o [s]
    double PAR_taus;  ///< Signal decay time constant τ_s [s]
    double PAR_tauf;  ///< Auto-regulation time constant τ_f [s]
    double PAR_k1;    ///< BOLD signal coefficient K1 (derived)
    double PAR_k2;    ///< BOLD signal coefficient K2 (derived)
    double PAR_k3;    ///< BOLD signal coefficient K3 (derived)
    double PAR_E0;    ///< Resting oxygen extraction fraction E0
    double PAR_V0;    ///< Resting blood volume fraction V0
    double PAR_TE;    ///< Echo time TE [s]
    double PAR_eps;   ///< Ratio of intra- to extra-vascular signal ε
    double PAR_nu0;   ///< Frequency offset at outer vessel surface ν0 [Hz]
    double PAR_r0;    ///< Slope of intravascular relaxation rate r0 [1/s]
    std::string integration_method; ///< "eulerDeterministic" or "heunDeterministic"

    double inv_taus;  ///< 1 / τ_s (precomputed)
    double inv_tauf;  ///< 1 / τ_f (precomputed)
    double inv_tauo;  ///< 1 / τ_o (precomputed)
    double inv_alpha; ///< 1 / α   (precomputed)

public:
    /**
     * @brief Construct a BOLD_4D model.
     *
     * K1, K2, K3, and the inverse time constants are computed automatically
     * from the given parameters.
     *
     * @param N                  Number of network nodes.
     * @param dt                 Integration time step [s].
     * @param integration_method "eulerDeterministic" or "heunDeterministic"
     *                           (default "heunDeterministic").
     * @param alpha  Grubb's stiffness exponent (default 0.32).
     * @param tauo   Transit time [s] (default 0.98).
     * @param taus   Signal decay time [s] (default 1.54).
     * @param tauf   Auto-regulation time [s] (default 1.44).
     * @param E0     Resting oxygen extraction fraction (default 0.4).
     * @param V0     Resting blood volume fraction [%] (default 4.0).
     * @param TE     Echo time [s] (default 0.04).
     * @param nu0    Frequency offset [Hz] (default 40.3).
     * @param r0     Intravascular relaxation rate slope [1/s] (default 25.0).
     * @param eps    Intra/extra-vascular ratio (default 0.5).
     * @param RBM    Regional BOLD model variant (currently only 1 is supported).
     */
    BOLD_4D(
        size_t N,
        double dt,
        std::string integration_method = "heunDeterministic",
        double alpha = 0.32,
        double tauo = 0.98,
        double taus = 1.54,
        double tauf = 1.44,
        double E0 = 0.4,
        double V0 = 4.0,
        double TE = 0.04,
        double nu0 = 40.3,
        double r0 = 25.0,
        double eps = 0.5,
        size_t RBM = 1) : N(N),
                          dt_b(dt),
                          PAR_alpha(alpha),
                          PAR_tauo(tauo),
                          PAR_taus(taus),
                          PAR_tauf(tauf),
                          PAR_E0(E0),
                          PAR_V0(V0),
                          PAR_TE(TE),
                          PAR_nu0(nu0),
                          PAR_r0(r0),
                          PAR_eps(eps)
    {

        assert(N != 0);
        inv_taus = 1.0 / PAR_taus;
        inv_tauf = 1.0 / PAR_tauf;
        inv_tauo = 1.0 / PAR_tauo;
        inv_alpha = 1.0 / PAR_alpha;
        this->integration_method = integration_method;
        if (RBM == 1)
        {
            PAR_k1 = 7.0 * PAR_E0;
            PAR_k2 = 2.0 * PAR_E0;
            PAR_k3 = 1.0 - PAR_eps;
        }
        else // TODO
        {
            printf("not implemented!");
            exit(EXIT_FAILURE);
        }
    }

    /**
     * @brief Evaluate the 4-state BOLD right-hand side.
     *
     * For each node i:
     * ```
     *   ds/dt  = vec_in[i+index] - s[i]/taus - (f[i]-1)/tauf
     *   df/dt  = s[i]
     *   dv/dt  = (f[i] - v[i]^(1/alpha)) / (tauo * v[i])
     *   dq/dt  = (f[i]*(1-(1-E0)^(1/f[i]))/E0 - v[i]^(1/alpha)*q[i]/v[i]) / (tauo*q[i])
     * ```
     * Variables are stored in log-space so that v,q remain positive.
     *
     * @param x       Current state [s, f, v, q] (length 4N).
     * @param dxdt    Output derivative (length 4N, pre-allocated).
     * @param t       Current time [s] (unused).
     * @param vec_in  Neural activity vector.
     * @param index   Offset into @p vec_in.
     * @return        Reference to @p dxdt.
     */
    dim1 bold_derivate(const dim1 &x,
                       dim1 &dxdt,
                       const double t,
                       const dim1 &vec_in,
                       const size_t index)
    {
        size_t n2 = 2 * N;
        size_t n3 = 3 * N;
        double PAR_E1 = 1.0 - PAR_E0;
        for (size_t i = 0; i < N; ++i)
        {
            dxdt[i] = vec_in[i + index] - inv_taus * x[i] - inv_tauf * (x[i + N] - 1.0);
            dxdt[i + N] = x[i];
            dxdt[i + n2] = inv_tauo * (x[i + N] - pow(x[i + n2], inv_alpha));
            dxdt[i + n3] = inv_tauo * ((x[i + N] * (1. - pow(PAR_E1, (1.0 / x[i + N]))) / PAR_E0) -
                                       (pow(x[i + n2], inv_alpha)) * (x[i + n3] / x[i + n2]));
        }
        return dxdt;
    }

    /**
     * @brief Advance the 4-state BOLD state by one Euler step.
     *
     * @param y0        Current state (length 4N), modified in-place.
     * @param t         Current time [s].
     * @param vec_in    Neural activity vector.
     * @param num_nodes Number of nodes.
     * @param index     Offset into @p vec_in.
     */
    void eulerDeterministic(dim1 &y0,
                            const double t,
                            const dim1 &vec_in,
                            const size_t num_nodes,
                            const size_t index)
    {
        size_t n = y0.size();
        dim1 dydt(n);
        bold_derivate(y0, dydt, t, vec_in, index);

        for (size_t i = 0; i < n; ++i)
            y0[i] += dydt[i] * dt_b;
    }

    /**
     * @brief Advance the 4-state BOLD state by one Heun step.
     *
     * @param y0        Current state (length 4N), modified in-place.
     * @param t         Current time [s].
     * @param vec_in    Neural activity vector.
     * @param num_nodes Number of nodes.
     * @param index     Offset into @p vec_in.
     */
    void heunDeterministic(dim1 &y0,
                           const double t,
                           const dim1 &vec_in,
                           const size_t num_nodes,
                           const size_t index)
    {
        size_t n = y0.size();
        dim1 k1(n);
        dim1 k2(n);
        dim1 tmp(n);

        bold_derivate(y0, k1, t, vec_in, index);
        for (size_t i = 0; i < n; ++i)
            tmp[i] = y0[i] + dt_b * k1[i];
        bold_derivate(tmp, k2, t + dt_b, vec_in, index);
        for (size_t i = 0; i < n; ++i)
            y0[i] += 0.5 * dt_b * (k1[i] + k2[i]);
    }

    /**
     * @brief Advance the hemodynamic state and return the BOLD signal.
     *
     * Updates @p y0 in-place and computes the nonlinear BOLD signal:
     * `BOLD[i] = V0 * (K1*(1-q[i]) + K2*(1-q[i]/v[i]) + K3*(1-v[i]))`
     *
     * @param y0        Current state (length 4N), modified in-place.
     * @param t         Current time [s].
     * @param vec_in    Neural activity vector.
     * @param num_nodes Number of nodes.
     * @param component "r" (index = 0) or "v" (index = num_nodes).
     * @return          BOLD signal vector (length N).
     */
    dim1 integrate(dim1 &y0,
                   const double t,
                   const dim1 &vec_in,
                   const size_t num_nodes,
                   const std::string component)
    {
        size_t index = (component == "v") ? num_nodes : 0;
        size_t n2 = 2 * N;
        size_t n3 = 3 * N;

        dim1 out(N);
        if (integration_method == "eulerDeterministic")
            eulerDeterministic(y0, t, vec_in, num_nodes, index);

        else if (integration_method == "heunDeterministic")
            heunDeterministic(y0, t, vec_in, num_nodes, index);

        else
        {
            printf("unknown integration method.\n");
            exit(EXIT_FAILURE);
        }

        // nonlinear
        for (size_t i = 0; i < N; ++i)
            out[i] = PAR_V0 * (PAR_k1 * (1.0 - y0[i + n3]) + PAR_k2 * (1.0 - (y0[i + n3] / y0[i + n2])) + PAR_k3 * (1.0 - y0[i + n2]));

        //         // linear
        //         // for (size_t i = 0; i < N; ++i)
        //         //     out[i] = PAR_V0 * ((PAR_k1 + PAR_k2) * (1.0 - y0[i + n3]) + (PAR_k3 - PAR_k2) * (1.0 - y0[i + n2]));

        return out;
    }
};

#endif
