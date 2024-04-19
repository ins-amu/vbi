#ifndef KM_SDE_HPP
#define KM_SDE_HPP

#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <assert.h>
#include <iostream>
#include <omp.h>
#include "utility.hpp"

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1> dim2;
typedef std::vector<float> dim1f;
typedef std::vector<dim1f> dim2f;
typedef std::vector<size_t> dim1i;
typedef std::vector<dim1i> dim2i;
typedef std::vector<std::vector<unsigned>> dim2I;

class KM_sde
{
    /*
    Kuramoto model with stochastic noise
    solve the SDE using heun's stochastic integration scheme.
    */

private:
    double dt;
    double t_initial;
    double t_transition;
    double t_end;
    double G;
    double noise_amp;
    dim1 theta;
    dim1 omega;
    dim2 alpha;
    dim2 weights;
    int num_nodes;
    dim2 Theta;
    dim1 times;
    dim2I adjlist;
    size_t num_steps;
    size_t num_steps_transition;
    size_t fix_seed;
    size_t buffer_size;

public:
    KM_sde(double dt,
           double t_initial,
           double t_transition,
           double t_end,
           double G,
           double noise_amp,
           dim1 theta,
           dim1 omega,
           dim2 alpha,
           dim2 weights,
           size_t fix_seed = 0,
           size_t num_threads = 1
           )
    {
        num_nodes = theta.size();
        this->dt = dt;
        this->t_initial = t_initial;
        this->t_transition = t_transition;
        this->t_end = t_end;
        this->G = G;
        this->theta = theta;
        this->omega = omega;
        this->alpha = alpha;
        this->weights = weights;
        this->noise_amp = noise_amp;
        this->fix_seed = fix_seed;

        omp_set_num_threads(num_threads);

        adjlist = adjmat_to_adjlist(weights);
        num_steps = int((t_end - t_initial) / dt);
        num_steps_transition = int((t_transition - t_initial) / dt);
        buffer_size = num_steps - num_steps_transition;
        Theta.resize(buffer_size);
        for (size_t i = 0; i < buffer_size; i++)
        {
            Theta[i].resize(num_nodes);
        }
        times.resize(buffer_size);
    }

    void rhs_f(dim1 &theta, dim1 &dtheta, const double /*t*/)
    {
        double sumj = 0.0;
#pragma omp parallel for reduction(+ : sumj)
        for (int i = 0; i < num_nodes; i++)
        {
            sumj = 0.0;
            for (int j = 0; j < adjlist[i].size(); j++)
            {
                int k = adjlist[i][j];
                sumj += weights[i][k] * sin(theta[k] - theta[i] - alpha[i][k]);
            }
            dtheta[i] = omega[i] + G * sumj;
        }
    }

    void heun(dim1 &y, const double t)
    {
        std::normal_distribution<> normal(0, 1);

        size_t nn = num_nodes;
        dim1 tmp(nn);
        dim1 k1(nn);
        dim1 k2(nn);

        rhs_f(y, k1, t);

        for (size_t i = 0; i < nn; ++i)
            tmp[i] = y[i] + dt * k1[i] + noise_amp * normal(rng(fix_seed));

        rhs_f(tmp, k2, t + dt);
        for (size_t i = 0; i < nn; ++i)
            y[i] += 0.5 * dt * (k1[i] + k2[i]) + noise_amp * normal(rng(fix_seed));

    }

    void IntegrateHeun()
    {
        for (size_t i = 0; i < num_steps_transition; i++)
        {
            heun(theta, t_initial);
            t_initial += dt;
        }

        for (size_t i = 0; i < buffer_size; i++)
        {
            heun(theta, t_initial);
            Theta[i] = theta;
            times[i] = t_initial;
            t_initial += dt;
        }
    }

    dim1 get_times()
    {
        return times;
    }

    dim2 get_theta()
    {
        return Theta;
    }
};

#endif
