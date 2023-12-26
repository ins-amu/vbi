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

class Stuart_Landau
{

private:
    int N;
    dim1 y0;
    double G;
    double dt;
    int nstart;
    dim1 omega;
    int num_nodes;
    double rNoise;
    double vNoise;
    double maxdelay;
    size_t fix_seed;
    int PAR_a;

    double t_end;
    double t_initial;
    double t_transition;
    size_t index_transition;
    long unsigned num_iteration;

    vector<vector<size_t>> adjlist;
    vector<vector<unsigned>> D;
    vector<vector<unsigned>> plag;

    dim1 t_ar;
    dim2 y;
    dim2 adj;
    dim2 DL;

public:
    Stuart_Landau(double dt,
                  dim1 y0,
                  dim2 adj,
                  dim2 DL,
                  double G,
                  double a,
                  dim1 omega,
                  double sigma_r,
                  double sigma_v,
                  double t_initial = 0.0,
                  double t_transition = 10.0,
                  double t_end = 400.0,
                  size_t fix_seed = 0)
    {

        assert(t_end > t_initial);
        assert(t_end > t_transition);
        assert(t_transition >= t_initial);

        PAR_a = a;
        this->G = G;
        this->dt = dt;
        this->DL = DL;
        this->adj = adj;
        this->omega = omega;
        this->t_end = t_end;
        this->t_initial = t_initial;
        this->t_transition = t_transition;

        num_iteration = int((t_end - t_initial) / dt);
        index_transition = int(t_transition / dt);
        N = num_nodes = adj.size();
        assert(y0.size() == (2 * N));

        {
            maxdelay = 0.0;
            dim1 tmp(N);
            for (size_t i = 0; i < N; ++i)
                tmp[i] = *std::max_element(DL[i].begin(), DL[i].end());
            maxdelay = *std::max_element(tmp.begin(), tmp.end());
        }
        nstart = (std::abs(maxdelay) > dt) ? int(maxdelay / dt) : 1;

        assert((nstart + 1 + index_transition) < num_iteration); // make sure the simulation is long enough

        // memory allocations -------------------------------------------------
        D.resize(N);
        y.resize(2 * N);
        plag.resize(N);
        t_ar.resize(num_iteration + nstart + 1);

        for (size_t i = 0; i < N; ++i)
        {
            D[i].resize(N);
            plag[i].resize(N);
            y[i].resize(num_iteration + nstart + 1);
            y[i + N].resize(num_iteration + nstart + 1);
            for (size_t j = 0; j < N; ++j)
            {
                plag[i][j] = nstart;
                D[i][j] = int(floor(DL[i][j] / dt)); // delay indices
            }
        }
        // --------------------------------------------------------------------

        rNoise = sqrt(dt) * sigma_r;
        vNoise = sqrt(dt) * sigma_v;
        adjlist = adjmat_to_adjlist(adj);
        set_history(y0);
    }

    void set_history(const dim1 &hist)
    {
        for (int i = 0; i < (nstart + 1); i++)
            t_ar[i] = -(nstart - i) / (double)nstart * maxdelay;
        for (int i = nstart; i < nstart + num_iteration + 1; ++i)
            t_ar[i] = (i - nstart + 1) * dt;

        // p_x_ar: N x nstart
        for (int i = 0; i < (2 * N); ++i)
            for (int j = 0; j < (nstart + 1); ++j)
                y[i][j] = hist[i];
    }

    dim1 f_sys(vector<double> &K,
               const double t,
               const long unsigned n,
               vector<vector<unsigned>> &plag)
    {
        dim1 dxdt(2 * N);
        dim1 x(N);
        dim1 v(N);

        for (int i = 0; i < N; ++i)
        {
            x[i] = y[i][n] + K[i];
            v[i] = y[i + N][n] + K[i + N];
        }

        for (size_t i = 0; i < N; ++i)
        {
            double coupling_x = 0.0;
            double coupling_v = 0.0;
            for (size_t j = 0; j < adjlist[i].size(); ++j)
            {
                int k = adjlist[i][j];
                coupling_x += adj[i][k] * (y[k][n - D[i][k]] - x[i]);
                coupling_v += adj[i][k] * (y[k + N][n - D[i][k]] - v[i]);
            }

            double x2v2 = x[i] * x[i] + v[i] * v[i];
            dxdt[i] = x[i] * (PAR_a - x2v2) - v[i] * omega[i] + G * coupling_x;
            dxdt[i + N] = v[i] * (PAR_a - x2v2) + x[i] * omega[i] + G * coupling_v;
        }

        return dxdt;
    }

    dim2 integrate_heun()
    {
        std::normal_distribution<> normal(0, 1);
        size_t nn = 2 * num_nodes;
        dim1 k0(nn);
        dim1 k1(nn);
        dim1 k2(nn);
        dim1 p(nn);

        for (size_t it = nstart; it < nstart + num_iteration; ++it)
        {
            k1 = f_sys(k0, 0, it, plag); // 0 -> t[it]

            for (int i = 0; i < N; ++i)
            {
                k1[i] = k1[i] * dt;
                k1[i + N] = k1[i + N] * dt;
                p[i] = y[i][it] + k1[i] + rNoise * normal(rng(fix_seed));
                p[i + N] = y[i + N][it] + k1[i + N] + vNoise * normal(rng(fix_seed));
            }

            k2 = f_sys(p, 0, it + 1, plag); // t_ar[it + 1]

            for (int i = 0; i < N; ++i)
            {
                k2[i] = k2[i] * dt + rNoise * normal(rng(fix_seed));
                k2[i + N] = k2[i + N] * dt + vNoise * normal(rng(fix_seed));
                y[i][it + 1] = y[i][it] + 0.5 * (k1[i] + k2[i]);
                y[i + N][it + 1] = y[i + N][it] + 0.5 * (k1[i + N] + k2[i + N]);
            }
        }

        return y;
    }

    dim2 integrate_euler()
    {
        std::normal_distribution<> normal(0, 1);
        size_t nn = 2 * num_nodes;
        dim1 dy(nn);
        dim1 k0(nn);
        for (size_t it = nstart; it < nstart + num_iteration; ++it)
        {
            dy = f_sys(k0, 0, it, plag);
            for (int i = 0; i < N; ++i)
            {
                
                y[i][it + 1] = y[i][it] + dt * dy[i] + rNoise * normal(rng(fix_seed));
                y[i + N][it + 1] = y[i + N][it] + dt * dy[i + N] + vNoise * normal(rng(fix_seed));
            }
        }
        return y;
    }

    dim1 get_time()
    {
        return t_ar;
    }

    dim2 get_y()
    {
        return y;
    }
};
