#ifndef bvep_HPP
#define bvep_HPP

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

class VEP
{
private:
    int nn;
    dim1 x;
    dim1 eta;
    double G;
    dim1 iext;
    double dt;
    dim1 times;
    double tcut;
    double tend;
    dim2 states;
    int fix_seed;
    dim2 weights;
    string method;
    double inv_tau;
    unsigned num_steps;
    double noise_sigma;
    dim1 initial_state;
    vector<vector<unsigned>> adjlist;

public:
    VEP(
        double G,
        dim1 iext,
        dim1 eta,
        double dt,
        double tcut,
        double tend,
        double tau,
        double noise_sigma,
        dim1 initial_state,
        dim2 weights,
        int fix_seed,
        string method)
    {
        this->G = G;
        this->dt = dt;
        this->nn = weights.size();
        this->iext = iext;
        this->eta = eta;
        this->tcut = tcut;
        this->tend = tend;
        this->method = method;
        this->fix_seed = fix_seed;
        this->weights = weights;
        this->inv_tau = 1.0 / tau;
        this->noise_sigma = noise_sigma;
        this->initial_state = initial_state;
        adjlist = adjmat_to_adjlist(weights);

        unsigned idx_cut = (int)(tcut / dt);
        num_steps = (int)(tend / dt);
        unsigned bufsize = num_steps - idx_cut;

        states.resize(bufsize);
        for (int i = 0; i < bufsize; i++)
        {
            states[i].resize(nn);
        }
        times.resize(bufsize);
    };

    void rhs(const vector<double> &x, vector<double> &dxdt, const double t)
    {
        for (int i = 0; i < nn; i++)
        {
            double gx = 0.0;
            for (size_t j = 0; j < adjlist[i].size(); ++j)
            {
                int k = adjlist[i][j];
                gx += weights[i][k] * (x[k] - x[i]);
            }
            dxdt[i] = 1.0 - x[i] * x[i] * x[i] - 2.0 * x[i] * x[i] - x[i + nn] + iext[i];
            dxdt[i + nn] = inv_tau * (4.0 * (x[i] - eta[i]) - x[i + nn] - G * gx);
        }
    }

    void euler_step(vector<double> &x, const double t)
    {
        std::normal_distribution<> normal(0, 1);
        vector<double> dxdt(nn * 2);

        rhs(x, dxdt, t);
        for (int i = 0; i < nn * 2; i++)
        {
            x[i] += dt * dxdt[i] + sqrt(dt) * noise_sigma * normal(rng(fix_seed));
        }
    }

    void heun_step(vector<double> &x, const double t)
    {
        std::normal_distribution<> normal(0, 1);
        vector<double> dxdt(nn * 2);
        vector<double> xtemp(nn * 2);

        rhs(x, dxdt, t);
        for (int i = 0; i < nn * 2; i++)
        {
            xtemp[i] = x[i] + dt * dxdt[i] + sqrt(dt) * noise_sigma * normal(rng(fix_seed));
        }

        vector<double> dxdt_temp(nn * 2);
        rhs(xtemp, dxdt_temp, t + dt);
        for (int i = 0; i < nn * 2; i++)
        {
            x[i] += 0.5 * dt * (dxdt[i] + dxdt_temp[i]) + sqrt(dt) * noise_sigma * normal(rng(fix_seed));
        }
    }

    void integrate()
    {
        dim1 x = initial_state;
        int idxtcut = (int)(tcut / dt);
        double t = 0.0;
        int counter = 0;

        for (int it = 0; it < num_steps; it++)
        {
            if (it >= idxtcut)
            {
                for (int i = 0; i < nn; i++)
                {
                    states[counter][i] = x[i];
                }
                times[counter] = t;
                counter++;
            }

            t += dt;
            if (method == "euler")
                euler_step(x, t);
            else if (method == "heun")
                heun_step(x, t);
            else
                throw std::invalid_argument("Invalid method");
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
};

#endif