#ifndef WW_SDE_HPP
#define WW_SDE_HPP

#include <cmath>
#include <fenv.h>
#include <vector>
#include <random>
#include <string>
#include <assert.h>
#include <iostream>
#include <fstream>
#include "utility.hpp"
// #include "bold.hpp"

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<unsigned> dim1I;
typedef std::vector<std::vector<double>> dim2;
typedef std::vector<std::vector<unsigned>> dim2I;

class WW_sde
{
private:
    int N;
    double dt;
    double dt_bold;
    double G;
    double a;
    double b;
    double d;
    double gamma;
    double tau_s;
    double inv_tau_s;
    double J_N;
    dim1 sigma;
    dim1 w;
    dim1 I_o;
    dim2 weights;
    dim2I adjlist;
    int fix_seed;
    int decimate_fmri;
    int decimate_ts;
    double t_end;
    double t_cut;

    dim2 states;
    dim1 times;
    dim1 initial_state;
    dim2 d_fmri;
    dim1 t_fmri;
    int RECORD_TS;
    int RECORD_FMRI;

public:
    WW_sde(
        int N,
        double dt,
        double dt_bold,
        double G,
        double t_cut,
        double t_end,
        dim1 y0,
        dim2 weights,
        double a,
        double b,
        double d,
        double gamma,
        double tau_s,
        dim1 w,
        double J_N,
        dim1 I_o,
        dim1 sigma,
        int decimate_fmri = 1,
        int decimate_ts = 1,
        int RECORD_TS = 1,
        int RECORD_FMRI = 1,
        int fix_seed = 0) : dt(dt), G(G)
    {
        assert(t_end > t_cut);
        this->decimate_fmri = decimate_fmri;
        this->decimate_ts = decimate_ts;
        this->t_end = t_end;
        this->t_cut = t_cut;
        this->N = N;
        this->dt_bold = dt_bold;
        this->weights = weights;
        this->a = a;
        this->b = b;
        this->d = d;
        this->gamma = gamma;
        this->tau_s = tau_s;
        this->w = w;
        this->J_N = J_N;
        this->I_o = I_o;
        this->sigma = sigma;
        this->fix_seed = fix_seed;
        this->RECORD_TS = RECORD_TS;
        this->RECORD_FMRI = RECORD_FMRI;
        inv_tau_s = 1.0 / tau_s;

        initial_state = y0;

        assert(decimate_ts > 0);
        assert(N == weights.size());
        adjlist = adjmat_to_adjlist(weights);
    }

    double H(const double x)
    {
        return (a * x - b) / (1.0 - exp(-d * (a * x - b)));
    }

    void f_ww(const dim1 &S, dim1 &dSdt, const double t)
    {
        double x = 0.0;
        // dim1 coupling = matvec(weights, S);
        for (int i; i < N; ++i)
        {
            double coupling = 0.0;
            for (int j : adjlist[i])
                coupling += weights[i][j] * S[j];
            x = w[i] * J_N * S[i] + I_o[i] + G * J_N * coupling;
            dSdt[i] = -S[i] * inv_tau_s + (1.0 - S[i]) * H(x) * gamma;
        }
    }

    void heun_sde_step(dim1 &y, const double t)
    {
        std::normal_distribution<> normal(0, 1);

        dim1 tmp(N);
        dim1 k0(N);
        dim1 k1(N);

        f_ww(y, k0, t);
        for (int i = 0; i < N; ++i)
            tmp[i] = y[i] + dt * k0[i] + sigma[i] * normal(rng(fix_seed));
        f_ww(tmp, k1, t + dt);
        for (int i = 0; i < N; ++i)
            y[i] += 0.5 * dt * (k0[i] + k1[i]) + sigma[i] * normal(rng(fix_seed));
    }

    void euler_sde_step(dim1 &y, const double t)
    {
        std::normal_distribution<> normal(0, 1);

        dim1 k0(N);
        f_ww(y, k0, t);
        for (int i = 0; i < N; ++i)
            y[i] += dt * k0[i] + sigma[i] * normal(rng(fix_seed));
    }

    dim1 bw_ode(const dim1 &xin,
                const dim1 &x)
    {
        double beta = 0.65;
        double gamma = 0.41;
        double tau = 0.98;
        double alpha = 0.33;
        double p_constant = 0.34;
        dim1 dfdt(4 * N);
        int N2 = 2 * N;
        int N3 = 3 * N;
        double inv_tau = 1.0 / tau;
        double inv_alpha = 1.0 / alpha;

        for (size_t i = 0; i < N; ++i)
        {
            dfdt[i] = xin[i] - beta * x[i] - gamma * (x[i + N] - 1.0);
            dfdt[i + N] = x[i];
            dfdt[i + N2] = inv_tau * (x[i + N] - pow(x[i + N2], inv_alpha));
            dfdt[i + N3] = inv_tau * (x[i + N] / p_constant * (1 - pow((1 - p_constant), (1 / x[i + N]))) - (x[i + N3] / x[i + N2]) * pow(x[i + N2], (1 / alpha)));
        }
        return dfdt;
    }

    void bw_ode_step(const dim1 &xin, dim1 &x, const double dt_bold)
    {
        int n = x.size();
        dim1 dfdt = bw_ode(xin, x);
        for (size_t i = 0; i < n; ++i)
            x[i] = x[i] + dt_bold * dfdt[i];
    }

    dim1 get_ybold(const dim1 &y)
    {
        dim1 yb(N);
        double p_costant = 0.34;
        double v_0 = 0.02;
        double k_1 = 4.3 * 28.265 * 3 * 0.0331 * p_costant;
        double k_2 = 0.47 * 110 * 0.0331 * p_costant;
        double k_3 = 0.53;
        double coef = 100 / p_costant * v_0;
        int N3 = 3 * N;
        int N2 = 2 * N;

        for (int i = 0; i < N; ++i)
        {
            yb[i] = coef * (k_1 * (1 - y[i + N3]) + k_2 * (1 - y[i + N3] / y[i + N2]) + k_3 * (1 - y[i + N2]));
        }
        return yb;
    }

    void integrate()
    {
        dim1 k0(N);
        dim1 k1(N);
        dim1 x0 = initial_state;

        size_t ii = 0;

        dim1 y0(4 * N, 1.0);
        for (int i = 0; i < N; ++i)
            y0[i] = 0.0;

        size_t nt = (size_t)(t_end / dt);
        size_t n_steps = (size_t)((nt) / decimate_ts);

        times.resize(n_steps);
        states.resize(n_steps);
        for (size_t i = 0; i < n_steps; ++i)
            states[i].resize(N);

        t_fmri.reserve(nt / decimate_fmri);
        d_fmri.reserve(nt / decimate_fmri);

        for (size_t itr = 1; itr <= nt; ++itr)
        {
            double t = itr * dt;
            double t_bold = itr * dt_bold;
            euler_sde_step(x0, t);
            bw_ode_step(x0, y0, dt_bold);

            if (RECORD_TS)
            {
                if (itr % decimate_ts == 0)
                {

                    times[ii] = t;
                    states[ii] = x0;
                    ii++;
                }
            }
            if (RECORD_FMRI)
            {
                if ((itr % decimate_fmri) == 0)
                {
                    dim1 yb = get_ybold(y0);
                    d_fmri.push_back(yb);
                    t_fmri.push_back(t_bold);

                    int info = find_nan(yb);
                    if (info == -1)
                    {
                        d_fmri.clear();
                        t_fmri.clear();
                        break;
                    }
                }
            }
        }
    }

    dim2 get_states()
    {
        return states;
    }
    dim1 get_times()
    {
        return times;
    }
    dim2 get_d_fmri()
    {
        return d_fmri;
    }
    dim1 get_t_fmri()
    {
        return t_fmri;
    }
};

#endif // WW_SDE_HPP
