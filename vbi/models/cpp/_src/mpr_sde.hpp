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

struct BoldParams
{
    double kappa = 0.65;
    double gamma = 0.41;
    double tau = 0.98;
    double alpha = 0.32;
    double epsilon = 0.34;
    double Eo = 0.4;
    double TE = 0.04;
    double vo = 0.08;
    double r0 = 25.0;
    double theta0 = 40.3;
    double rtol = 1e-5;
    double atol = 1e-8;
    double dt_b = 0.001;
    double K1 = 4.3 * theta0 * Eo * TE;
    double K2 = epsilon * r0 * Eo * TE;
    double K3 = 1 - epsilon;
    double ialpha = 1 / alpha;

    BoldParams() = default;

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

class MPR_sde
{

private:
    dim1 delta;
    dim1 tau;
    dim1 eta;
    dim1 J;
    dim1 i_app;
    double dt;
    double dt_b;
    double G;

    double rNoise;
    double vNoise;
    double noise_amp;

    size_t num_nodes;
    size_t num_steps;
    size_t rv_decimate;
    size_t idx_cut;
    unsigned RECORD_RV;
    unsigned RECORD_BOLD;

    double t_end;
    double t_cut;
    double tr; // TR: repetition time [time interval for bold sampling]
    vector<vector<unsigned>> adjlist;

    dim2 weights;
    dim1 t_arr;
    int fix_seed;
    dim1 initial_state;
    BoldParams bp;

public:
    dim2 bold_d;
    dim1 bold_t;
    dim2 r_d;
    dim1 r_t;

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
            const BoldParams &bp) : dt(dt), dt_b(dt_b), G(G), delta(delta),
                                    tau(tau), eta(eta), J(J), t_end(t_end),
                                    t_cut(t_cut), tr(tr), RECORD_RV(RECORD_RV),
                                    RECORD_BOLD(RECORD_BOLD), fix_seed(fix_seed),
                                    weights(weights), initial_state(initial_state),
                                    rv_decimate(rv_decimate), noise_amp(noise_amp), i_app(i_app)

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

    void f_mpr(
        const dim1 &x,
        dim1 &dxdt,
        const double t)
    {
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

        for (int i = 0; i < n; i++)
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
                if (((itr % rv_decimate) == 0) && ((int)(itr / rv_decimate) < r_d.size()))
                {
                    unsigned idx = (int)(itr / rv_decimate);
                    r_d[idx] = rv_current;
                    r_t[idx] = t_current;
                }
            }
            if (RECORD_BOLD)
            {
                bold_step(rv_current, s, f, ftilde, vtilde, qtilde, v, q, dtt);

                if (((itr % b_decimate) == 0) && ((int)(itr / b_decimate) < bold_d.size()))
                {
                    unsigned idx = (int)(itr / b_decimate);
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
            for (int j = 0; j < n; ++j)
                bold_d[i][j] = bp.vo * (bp.K1 * (1 - qq[i][j]) + bp.K2 * (1 - qq[i][j] / vv[i][j]) + bp.K3 * (1 - vv[i][j]));
        }
    }

    dim2 get_bold_d()
    {
        return bold_d;
    }
    dim1 get_bold_t()
    {
        return bold_t;
    }

    dim2 get_r_d()
    {
        return r_d;
    }
    dim1 get_r_t()
    {
        return r_t;
    }
};

#endif
