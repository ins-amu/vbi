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
#include "bold.hpp"

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1> dim2;
typedef std::vector<float> dim1f;
typedef std::vector<dim1f> dim2f;

class MPR_sde
{
private:
    double delta;
    dim1 iapp;
    double tau;
    dim1 eta;
    double dt;
    double J;
    double G;
    double alpha_sc;
    double beta_sc;
    size_t decimate;
    BOLD_4D boldObj;

    double rNoise;
    double vNoise;

    size_t num_nodes;
    size_t num_steps;
    size_t record_step;
    size_t index_transition;
    size_t RECORD_AVG;
    int APPLY_STIMULATION; // to apply stimulation

    double t_final;
    double t_initial;
    double t_transition;
    vector<vector<unsigned>> adjlist;
    vector<vector<unsigned>> adjlist_A;
    vector<vector<unsigned>> adjlist_B;

    dim2 adj;
    dim2 adj_A;
    dim2 adj_B;
    dim1 times;
    int fix_seed;
    dim1 initial_state;
    vector<int> sti_indices; // indices of nodes to be stimulated
    double sti_amplitude;

public:
    dim2 bold_buffer;
    dim1 time_buffer;

    MPR_sde(double dt,
        double dt_bold,
        size_t decimate,
        dim1 y,
        dim2 adj,
        dim2 adj_A,
        dim2 adj_B,
        double G,
        dim1 eta,
        double alpha_sc,
        double beta_sc,
        dim1 sti_indices_,
        double J = 14.5,
        double tau = 1.0,
        double delta = 0.7,
        double i_app = 0.0,
        int apply_sti = 0, // to apply stimulation
        double noise_amp = 0.35,
        size_t record_step = 10,
        double t_initial = 0.0,
        double t_transition = 10.0,
        double t_final = 4000.0,
        size_t RECORD_AVG = 0,
        int fix_seed = 0) : dt(dt),
                            G(G),
                            eta(eta),
                            J(J),
                            tau(tau),
                            delta(delta),
                            boldObj(int(y.size() / 2), dt_bold)
    {

        assert(t_final > t_initial);
        assert(t_final > t_transition);

        this->adj = adj;
        this->adj_A = adj_A;
        this->adj_B = adj_B;
        this->alpha_sc = alpha_sc;
        this->beta_sc = beta_sc;
        this->t_final = t_final;
        this->t_initial = t_initial;
        this->record_step = record_step;
        this->t_transition = t_transition;
        this->decimate = decimate;
        this->RECORD_AVG = RECORD_AVG;
        this->APPLY_STIMULATION = apply_sti;
        this->fix_seed = fix_seed;
        // this->sti_indices = sti_indices;
        initial_state = y;

        if (sti_indices_.size() > 0)
        {
            sti_indices.resize(sti_indices_.size());
            for (int i=0; i<sti_indices_.size(); i++)
                sti_indices[i] = int(sti_indices_[i]);
        }
        else
            sti_indices.resize(0);

        num_nodes = adj.size();
        num_steps = int((t_final - t_initial) / dt);
        index_transition = int(t_transition / dt);

        // consistent with the TVB
        rNoise = sqrt(dt) * sqrt(2 * noise_amp);
        vNoise = sqrt(dt) * sqrt(2 * 2 * noise_amp);

        adjlist = adjmat_to_adjlist(adj);
        adjlist_A = adjmat_to_adjlist(adj_A);
        adjlist_B = adjmat_to_adjlist(adj_B);

        // preaparing input current vector if stimulation is applied
        iapp.resize(num_nodes);
        std::fill(this->iapp.begin(), this->iapp.end(), 0.0);
        if (apply_sti)
            sti_amplitude = i_app;
    }
    // ------------------------------------------------------------------------
    void derivative(
        const vector<double> &x,
        vector<double> &dxdt,
        const double t)
    {
        size_t N = num_nodes;

        double rtau = 1.0 / tau;
        double tau2 = tau * tau;
        double PI2 = M_PI * M_PI;
        double delta_over_tau_pi = delta / (tau * M_PI);
        double J_tau = J * tau;

        for (size_t i = 0; i < N; ++i)
        {
            double CP0 = 0;
            double CP1 = 0;
            double CP2 = 0;
            for (size_t j = 0; j < adjlist[i].size(); ++j)
            {
                int k = adjlist[i][j];
                CP0 += adj[i][k] * x[k];
            }
            if (std::abs(alpha_sc) > 1e-10)
            {
                for (size_t j = 0; j < adjlist_A[i].size(); ++j)
                {
                    int k = adjlist_A[i][j];
                    CP1 += adj_A[i][k] * x[k];
                }
            }
            if (std::abs(beta_sc) > 1e-10)
            {
                for (size_t j = 0; j < adjlist_B[i].size(); ++j)
                {
                    int k = adjlist_B[i][j];
                    CP2 += adj_B[i][k] * x[k];
                }
            }

            dxdt[i] = rtau * (delta_over_tau_pi + 2 * x[i] * x[i + N]);
            dxdt[i + N] = rtau * (x[i + N] * x[i + N] + eta[i] + iapp[i] + J_tau * x[i] - (PI2 * tau2 * x[i] * x[i]) + G * CP0 - alpha_sc * CP1 - beta_sc * CP2);
        }
    }
    // ------------------------------------------------------------------------
    void heun(dim1 &y, const double t)
    {
        std::normal_distribution<> normal(0, 1);

        size_t nn = 2 * num_nodes;
        size_t N = num_nodes;
        dim1 tmp(nn);
        dim1 k1(nn);
        dim1 k2(nn);

        derivative(y, k1, t);

        for (size_t i = 0; i < nn; ++i)
            if (i < N)
                tmp[i] = y[i] + dt * k1[i] + rNoise * normal(rng(fix_seed));
            else
                tmp[i] = y[i] + dt * k1[i] + vNoise * normal(rng(fix_seed));

        derivative(tmp, k2, t + dt);
        for (size_t i = 0; i < nn; ++i)
        {
            if (i < N)
            {
                y[i] += 0.5 * dt * (k1[i] + k2[i]) + rNoise * normal(rng(fix_seed));
                if (y[i] < 0)
                    y[i] = 0.0;
            }
            else
                y[i] += 0.5 * dt * (k1[i] + k2[i]) + vNoise * normal(rng(fix_seed));
        }
    }
    // ------------------------------------------------------------------------
    void heunStochasticIntegrate()
    {
        size_t nn = 2 * num_nodes;
        size_t N = num_nodes;
        dim1 k1(nn);
        dim1 k2(nn);
        dim1 tmp(nn);
        dim1 y = initial_state;

        size_t counter = 0;
        size_t buffer_counter = 0;

        //---------------------------------------------------------------------
        dim1 y_tmp(4 * N, 1.0);
        for (size_t i = 0; i < N; ++i)
            y_tmp[i] = 0.0;
        // Stimulation parameters ---------------------------------------------
        int sti_interval = int(1.0 / dt * decimate);        // interval between stimulation
        int sti_duration = int(sti_interval / record_step); // stimulus duration
        int sti_aux = 0;                                    // auxulary variable for counting the stimulus duration
        bool sti_state = false;

        for (int itr = 1; itr < num_steps + 1; ++itr)
        {
            double t = itr * dt;

            if (APPLY_STIMULATION) // apply stimulation
            {
                if ((itr % sti_interval) == 0)
                {
                    fill_vector(iapp, sti_indices, sti_amplitude);
                    sti_aux = sti_duration;
                    sti_state = true;
                }

                if (sti_aux > 0)
                    --sti_aux;
                else if ((sti_aux == 0) && (sti_state))
                {
                    fill(iapp.begin(), iapp.end(), 0.0);
                    sti_state = false;
                }
            } // end of apply stimulation

            heun(y, t); // integrate one step

            // Apply BOLD and record after a transiant period
            // if (((itr % record_step) == 0) && (itr > index_transition))
            if (((itr % record_step) == 0) && (itr > 1))
            {

                dim1 y_bold = boldObj.integrate(y_tmp, t, y, num_nodes, "r"); // update y_tmp and y_bold

                if (((buffer_counter % decimate) == 0) && (buffer_counter != 0))
                {
                    bold_buffer.push_back(y_bold);
                    time_buffer.push_back(t);


                    // check for NAN and Break if True
                    {
                        size_t ind = y_bold.size() - 1;
                        if (std::isnan(y_bold[ind]))
                        {
                            std::cout << "nan found! "
                                      << "\n";
                            bold_buffer.clear();
                            break;
                        }
                    }

                    buffer_counter = 0;
                }
                buffer_counter++;
            }
        }
    }

    dim2 get_bold()
    {
        return bold_buffer;
    }
    dim1 get_time()
    {
        return time_buffer;
    }
};

#endif