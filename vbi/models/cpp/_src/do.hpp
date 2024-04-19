#include <vector>
#include <string>
#include <assert.h>
#include <iostream>

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1> dim2;

class DO
{
private:
    double dt;
    double PAR_a;
    double PAR_b;

    size_t dimension;
    size_t num_steps;
    double t_final;
    double t_initial;

    dim1 times;
    dim2 states;
    dim1 initial_state;

public:
    DO(double dt,    // time step
       double a, // parameter
       double b,
       double t_initial,
       double t_final, // simulation time
       dim1 y          // state vector
       ) : dt(dt), PAR_a(a), PAR_b(b)
    {
        assert(t_final >= t_initial);
        this->t_final = t_final;
        this->t_initial = t_initial;
        initial_state = y;

        dimension = y.size();
        num_steps = int((t_final - t_initial) / dt);

        states.resize(num_steps);
        for (size_t i = 0; i < num_steps; ++i)
            states[i].resize(dimension);
        times.resize(num_steps);
    }

    void derivative(const vector<double> &x,
             vector<double> &dxdt,
             const double t)
    {
        dxdt[0] = x[0] - x[0] * x[1] - PAR_a * x[0] * x[0];
        dxdt[1] = x[0] * x[1] - x[1] - PAR_b * x[1] * x[1];
    }

    void eulerIntegrate()
    {
        size_t n = dimension;
        dim1 dxdt(n);
        states[0] = initial_state;
        times[0] = t_initial;
        dim1 y = initial_state;

        for (int step = 1; step < num_steps; ++step)
        {
            double t = step * dt;
            euler(y, t);

            states[step] = y;
            times[step] = t_initial + t;
        }
    }

    void euler(dim1& y, const double t)
    {
        size_t n = y.size();
        dim1 dydt(n);
        derivative(y, dydt, t);
        for (size_t i = 0; i < n; ++i)
            y[i] += dydt[i] * dt;
    }

    // void heunIntegrate()
    // {
    //     size_t n = dimension;
    //     dim1 dxdt(n), dydt(n), f(n);
    //     states[0] = initial_state;
    //     times[0] = t_initial;
    //     dim1 y = initial_state;

    //     for (int step = 1; step < num_steps; ++step)
    //     {
    //         double t = step * dt;
    //         heun(y, t);

    //         states[step] = y;
    //         times[step] = t_initial + t;
    //     }
    // }

    // void heun(dim1& y, const double t)
    // {
    //     size_t n = y.size();
    //     dim1 dydt(n), f(n);
    //     derivative(y, dydt, t);
    //     for (size_t i = 0; i < n; ++i)
    //         f[i] = y[i] + dydt[i] * dt;

    //     derivative(f, dydt, t + dt);
    //     for (size_t i = 0; i < n; ++i)
    //         y[i] += 0.5 * (dydt[i] + dydt[i]) * dt;
    // }

    void rk4Integrate()
    {
        size_t n = dimension;

        states[0] = initial_state;
        times[0] = t_initial;
        dim1 y = initial_state;

        for (int step = 1; step < num_steps; ++step)
        {
            double t = step * dt;
            rk4(y, t);
            states[step] = y;
            times[step] = t_initial + t;
        }
    }

    void rk4(dim1&y, const double t)
    {

        size_t n = y.size();
        dim1 k1(n), k2(n), k3(n), k4(n);
        dim1 f(n);
        double c_dt = 1.0 / 6.0 * dt;

        derivative(y, k1, t);
        for (int i = 0; i < n; i++)
            f[i] = y[i] + 0.5 * dt * k1[i];

        derivative(f, k2, t);
        for (int i = 0; i < n; i++)
            f[i] = y[i] + 0.5 * dt * k2[i];

        derivative(f, k3, dt);
        for (int i = 0; i < n; i++)
            f[i] = y[i] + dt * k3[i];

        derivative(f, k4, t);
        for (int i = 0; i < n; i++)
            y[i] += (k1[i] + 2.0 * (k2[i] + k3[i]) + k4[i]) * c_dt;
    }

    dim2 get_coordinates()
    {
        return states;
    }
    dim1 get_times()
    {
        return times;
    }
};
