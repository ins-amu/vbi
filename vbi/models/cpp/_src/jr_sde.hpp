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

class JR_sde
{
private:
    size_t N;
    double dt;
    double t_initial;
    double t_final;
    double t_transition;
    size_t dimension;
    size_t num_steps;
    size_t index_transition;
    vector<vector<unsigned>> adjlist;

    dim1 A;
    double par_a;
    double par_B;
    double par_b;
    double par_r;
    double par_v0;
    double par_vmax;
    double coupling;
    double noise_mu;
    double noise_sigma;

    dim2 adj;
    dim1 C0;
    dim1 C1;
    dim1 C2;
    dim1 C3;

    int fix_seed;

    dim1 times;
    dim2 states;
    dim1 initial_state;

    // bool ADJ_SET = false;      //check if adjacency matrix is set.
    // bool COUPLING_SET = false; //check if coupling is set

public:
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
    // ------------------------------------------------------------------------
    double sigma(const double v)
    {
        return par_vmax / (1 + exp(par_r * (par_v0 - v)));
    }
    // ------------------------------------------------------------------------
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
    // ------------------------------------------------------------------------
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
    // ------------------------------------------------------------------------
    void eulerIntegrate()
    {
        size_t n = dimension;
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

    void heunIntegrate()
    {
        size_t n = dimension;
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

    // ------------------------------------------------------------------------
    dim2 get_coordinates()
    {
        return states;
    }
    // ------------------------------------------------------------------------
    dim1 get_times()
    {
        return times;
    }
};

#endif
