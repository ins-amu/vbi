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

class BOLD_2D
{
private:
    size_t N;
    double dt_b;
    double PAR_rho;
    double PAR_e;
    double PAR_taus;
    double PAR_tauf;
    double PAR_k1;
    double PAR_eps;
    double inv_taus;
    double inv_tauf;
    std::string integration_method;

public:
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
    // ------------------------------------------------------------------------
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
    // ------------------------------------------------------------------------
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
    // ------------------------------------------------------------------------
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
    // ------------------------------------------------------------------------
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

class BOLD_4D
{

private:
    size_t N;
    double dt_b;
    double PAR_alpha;
    double PAR_tauo;
    double PAR_taus;
    double PAR_tauf;
    double PAR_k1;
    double PAR_k2;
    double PAR_k3;
    double PAR_E0;
    double PAR_V0;
    double PAR_TE;
    double PAR_eps;
    double PAR_nu0;
    double PAR_r0;
    std::string integration_method;

    double inv_taus;
    double inv_tauf;
    double inv_tauo;
    double inv_alpha;

public:
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
    //     // ------------------------------------------------------------------------
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
    //     // ------------------------------------------------------------------------
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
    // ------------------------------------------------------------------------
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
    //     // ------------------------------------------------------------------------
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
