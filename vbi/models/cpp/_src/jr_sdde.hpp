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

class JR_sdde
{

private:
    int N;
    dim1 y0;
    double dt;
    int dimension;
    int num_nodes;
    double maxdelay;
    size_t fix_seed;
    double coupling;
    double par_vmax;
    dim2 states;

    dim1 C0;
    dim1 C1;
    dim1 C2;
    dim1 C3;

    double noise_sigma;
    double noise_mu;
    double par_A;
    double par_a;
    double par_B;
    double par_b;
    double par_r;
    double par_v0;

    dim1 sti_amp;
    double sti_ti;
    double sti_duration;
    double sti_gain = 0.0;
    double _sti_gain = 0.0;

    double t_final;
    double t_transition;
    size_t index_transition;
    long unsigned num_iteration;

    vector<vector<unsigned>> adjlist;
    vector<vector<unsigned>> D;
    // vector<vector<unsigned>> plag;

    dim1 t_ar;
    dim1 t_arr;
    dim2 y;
    dim2 adj;
    dim2 delays;

public:
    int nstart;
    dim1 sti_vector;

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

    void set_history(const dim1 &hist)
    {
        for (int i = 0; i < num_iteration; ++i)
            t_ar[i] = i * dt;

        // p_x_ar: N x nstart
        for (int i = 0; i < (dimension * N); ++i)
            for (int j = 0; j < nstart + 1; ++j)
                y[i][j] = hist[i];
    }

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

    // ------------------------------------------------------------------------
    double _sigma(const double v)
    {
        return par_vmax / (1 + exp(par_r * (par_v0 - v)));
    }

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
    // ------------------------------------------------------------------------

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
    // ------------------------------------------------------------------------

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
    // ------------------------------------------------------------------------

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

    dim1 get_t()
    {
        return t_arr;
    }

    dim2 get_y()
    {
        return states;
    }
    dim1 get_sti_vector()
    {
        return sti_vector;
    }
};

#endif
