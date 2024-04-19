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
#include "bold.hpp"

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
    double G;
    double a;
    double b;
    double d;
    double gamma;
    double tau_s;
    double w;
    double J_N;
    double I_o;
    double sigma;
    dim2 weights;
    dim2I adjlist;
    int fix_seed;
    int decimate;
    int record_step;
    double t_end;
    double t_cut;

    BOLD_4D boldObj;

    dim2 states;
    dim1 times;
    dim1 initial_state;
    dim2 d_fmri;
    dim1 t_fmri;
    int RECORD_TS;
    int RECORD_FMRI;
    int SPARSE;

public:
    WW_sde(
        int N,
        double dt,
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
        double w,
        double J_N,
        double I_o,
        double sigma,
        int decimate = 1,
        int record_step = 1,
        int RECORD_TS = 1,
        int RECORD_FMRI = 1,
        int SPARSE = 0,
        int fix_seed = 0) : dt(dt), G(G), boldObj(N, dt)
    {
        assert(t_end > t_cut);
        this->decimate = decimate;
        this->record_step = record_step;
        this->t_end = t_end;
        this->t_cut = t_cut;
        this->N = N;
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
        this->SPARSE = SPARSE;

        initial_state = y0;

        assert(record_step > 0);
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
        double inv_tau_s = 1.0 / tau_s;
        // dim1 coupling = matvec(weights, S);

        for (int i; i < N; ++i)
        {
            double coupling = 0.0;
            for (int j: adjlist[i])
                coupling += weights[i][j] * S[j];
            x = w * J_N * S[i] + I_o + G * J_N * coupling;
            dSdt[i] = -S[i] * inv_tau_s + (1.0 - S[i]) * H(x) * gamma;
        }
    }

    void heun_sde(dim1 &y, const double t)
    {
        std::normal_distribution<> normal(0, 1);

        dim1 tmp(N);
        dim1 k0(N);
        dim1 k1(N);

        f_ww(y, k0, t);
        for (int i = 0; i < N; ++i)
            tmp[i] = y[i] + dt * k0[i] + sigma * normal(rng(fix_seed));
        f_ww(tmp, k1, t + dt);
        for (int i = 0; i < N; ++i)
            y[i] += 0.5 * dt * (k0[i] + k1[i]) + sigma * normal(rng(fix_seed));
    }

    void Integrate()
    {
        dim1 k0(N);
        dim1 k1(N);
        dim1 x0 = initial_state;

        size_t ii = 0;

        dim1 y0(4 * N, 1.0);
        for (int i = 0; i < N; ++i)
            y0[i] = 0.0;

        size_t nt = (size_t)(t_end / dt);
        size_t ncut = (size_t)(t_cut / dt);
        size_t n_steps = (size_t)((nt - ncut) / record_step);

        times.resize(n_steps);
        states.resize(n_steps);
        for (size_t i = 0; i < n_steps; ++i)
            states[i].resize(N);

        t_fmri.reserve(nt / decimate);
        d_fmri.reserve(nt / decimate);

        for (size_t itr = 1; itr <= nt; ++itr)
        {
            double t = itr * dt;
            heun_sde(x0, t);
            if (RECORD_TS)
            {
                if ((itr % record_step == 0) && (t > t_cut))
                {

                    times[ii] = t;
                    states[ii] = x0;
                    ii++;
                }
            }
            if (RECORD_FMRI)
            {
                if ((itr % decimate) == 0)
                {
                    dim1 yb = boldObj.integrate(y0, t, x0, N, "r");
                    d_fmri.push_back(yb);
                    t_fmri.push_back(t);

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
