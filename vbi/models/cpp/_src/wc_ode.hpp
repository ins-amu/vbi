#ifndef WC_ODE_HPP_
#define WC_ODE_HPP_

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
typedef std::vector<unsigned int> dim1I;
typedef std::vector<dim1I> dim2I;


class WC_ode
{
private:
    int N;
    double dt;
    double c_ee;
    double c_ei;
    double c_ie;
    double c_ii;
    double tau_e;
    double tau_i;
    double a_e;
    double a_i;
    double b_e;
    double b_i;
    double c_e;
    double c_i;
    double theta_i;
    double theta_e;
    double r_e;
    double r_i;
    double k_e;
    double k_i;
    double alpha_e;
    double alpha_i;
    double g_e;
    double g_i;
    double inv_tau_e;
    double inv_tau_i;
    size_t num_steps;
    size_t index_cut;
    dim1 P;
    dim1 Q;
    dim1 x0;
    dim2 weights;
    dim2I adjlist;

    int fix_seed;
    double t_end;
    double t_cut;

    vector<float> times;
    vector<double> initial_state;
    vector<vector<float>> states;

public:
    WC_ode(
        int N,
        double dt,
        dim1 P,
        dim1 Q,
        dim1 x0,
        dim2 weights,
        double t_end = 300.0,
        double t_cut = 0.0,
        double c_ee = 16.0,
        double c_ei = 12.0,
        double c_ie = 15.0,
        double c_ii = 3.0,
        double tau_e = 8.0,
        double tau_i = 8.0,
        double a_e = 1.3,
        double a_i = 2.0,
        double b_e = 4.0,
        double b_i = 3.7,
        double c_e = 1.0,
        double c_i = 1.0,
        double theta_i = 0.0,
        double theta_e = 0.0,
        double r_e = 1.0,
        double r_i = 1.0,
        double k_e = 0.994,
        double k_i = 0.999,
        double alpha_e = 1.0,
        double alpha_i = 1.0,
        double g_e = 0.0,
        double g_i = 0.0,
        int fix_seed = 0)
    {
        this->N = N;
        this->dt = dt;
        this->c_ee = c_ee;
        this->c_ei = c_ei;
        this->c_ie = c_ie;
        this->c_ii = c_ii;
        this->tau_e = tau_e;
        this->tau_i = tau_i;
        this->a_e = a_e;
        this->a_i = a_i;
        this->b_e = b_e;
        this->b_i = b_i;
        this->c_e = c_e;
        this->c_i = c_i;
        this->theta_i = theta_i;
        this->theta_e = theta_e;
        this->r_e = r_e;
        this->r_i = r_i;
        this->k_e = k_e;
        this->k_i = k_i;
        this->alpha_e = alpha_e;
        this->alpha_i = alpha_i;
        this->g_e = g_e;
        this->g_i = g_i;
        this->weights = weights;
        this->P = P;
        this->Q = Q;
        this->x0 = x0;

        inv_tau_e = 1.0 / tau_e;
        inv_tau_i = 1.0 / tau_i;

        adjlist = adjmat_to_adjlist(weights);
        num_steps = int(t_end / dt);
        index_cut = int(t_cut / dt);
        size_t buffer_size = num_steps - index_cut;
        times.resize(buffer_size);
        states.resize(buffer_size);
        for (size_t i = 0; i < buffer_size; ++i)
            states[i].resize(2 * N);
    }

    double sigmoid(const double x,
                   const double a,
                   const double b,
                   const double c)
    {
        return c / (1.0 + exp(-a * (x - b)));
    }

    void rhs(const dim1 &y,
             dim1 &dydt,
             const double dt)
    {
        dim1 lc_e(N);
        dim1 lc_i(N);
        double thr = 1e-6;
        if (std::abs(g_e) > thr)
            lc_e = matvec(weights, y, 0);
        if (std::abs(g_i) > thr)
            lc_i = matvec(weights, y, N);

        for (int i = 0; i < N; ++i)
        {
            double x_e = alpha_e * (c_ee * y[i] - c_ei * y[i + N] + P[i] - theta_e + g_e * lc_e[i]);
            double x_i = alpha_i * (c_ie * y[i] - c_ii * y[i + N] + Q[i] - theta_i + g_i * lc_i[i]);
            double s_e = sigmoid(x_e, a_e, b_e, c_e);
            double s_i = sigmoid(x_i, a_i, b_i, c_i);
            dydt[i] = inv_tau_e * (-y[i] + (k_e - r_e * y[i]) * s_e);
            dydt[i + N] = inv_tau_i * (-y[i + N] + (k_i - r_i * y[i + N]) * s_i);
        }
    }

    void euler_step(dim1 &y, const double dt)
    {

        dim1 dydt(2 * N);
        rhs(y, dydt, dt);
        for (int i = 0; i < 2 * N; ++i)
            y[i] += dt * dydt[i];
    }

    void eulerIntegrate()
    {
        size_t nc = 2 * N;
        dim1 dy(nc);
        dim1 y(nc);

        y = x0;
        for (size_t i = 0; i < index_cut; ++i)
            euler_step(y, dt);

        size_t ind = 0;
        for (size_t i = index_cut; i < num_steps; ++i)
        {
            euler_step(y, dt);
            times[ind] = i * dt;
            for (size_t j = 0; j < 2 * N; ++j)
                states[ind][j] = y[j];
            ind++;
        }
    }

    void heun_step(dim1 &y, const double dt)
    {
        dim1 dydt(2 * N);
        dim1 tmp(2 * N);
        rhs(y, dydt, dt);
        for (size_t i = 0; i < 2 * N; ++i)
            tmp[i] = y[i] + dt * dydt[i];
        rhs(tmp, dydt, dt);
        for (size_t i = 0; i < 2 * N; ++i)
            y[i] += 0.5 * dt * (dydt[i] + dydt[i]);
    }

    void heunIntegrate()
    {
        size_t nc = 2 * N;
        dim1 dy(nc);
        dim1 y(nc);

        y = x0;
        for (size_t i = 0; i < index_cut; ++i)
            heun_step(y, dt);

        size_t ind = 0;
        for (size_t i = index_cut; i < num_steps; ++i)
        {
            heun_step(y, dt);
            times[ind] = i * dt;
            for (size_t j = 0; j < 2 * N; ++j)
                states[ind][j] = y[j];
            ind++;
        }
    }

    void rk4_step(dim1 &y, const double dt)
    {
        dim1 dydt(2 * N);
        dim1 tmp(2 * N);
        dim1 k1(2 * N);
        dim1 k2(2 * N);
        dim1 k3(2 * N);
        dim1 k4(2 * N);
        double dt_over_6 = dt / 6.0;

        rhs(y, k1, dt);
        for (size_t i = 0; i < 2 * N; ++i)
            tmp[i] = y[i] + 0.5 * dt * k1[i];
        rhs(tmp, k2, dt);
        for (size_t i = 0; i < 2 * N; ++i)
            tmp[i] = y[i] + 0.5 * dt * k2[i];
        rhs(tmp, k3, dt);
        for (size_t i = 0; i < 2 * N; ++i)
            tmp[i] = y[i] + dt * k3[i];
        rhs(tmp, k4, dt);
        for (size_t i = 0; i < 2 * N; ++i)
            y[i] += dt_over_6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    void rk4Integrate()
    {
        size_t nc = 2 * N;
        dim1 dy(nc);
        dim1 y(nc);

        y = x0;
        for (size_t i = 0; i < index_cut; ++i)
            rk4_step(y, dt);

        size_t ind = 0;
        for (size_t i = index_cut; i < num_steps; ++i)
        {
            rk4_step(y, dt);
            times[ind] = i * dt;
            for (size_t j = 0; j < 2 * N; ++j)
                states[ind][j] = y[j];
            ind++;
        }
    }

    vector<vector<float>> get_states()
    {
        return states;
    }

    vector<float> get_times()
    {
        return times;
    }
};

#endif // WC_ODE_HPP_
