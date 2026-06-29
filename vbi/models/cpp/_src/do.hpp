/**
 * @file do.hpp
 * @brief Damped Oscillator (DO) — a 2D competitive (Lotka-Volterra-like)
 *        nonlinear oscillator for testing and benchmarking VBI inference.
 *
 * The model describes two interacting populations (x, y) with self-damping
 * and cross-inhibition, making it a useful toy system for validating the
 * inference pipeline before applying it to more complex brain models.
 *
 * **Equations:**
 * ```
 *   dx/dt = x - x*y - a*x^2
 *   dy/dt = x*y - y - b*y^2
 * ```
 * where @p a and @p b are intraspecific competition (self-damping) parameters.
 *
 * **State vector:** [x, y] (length 2).
 *
 * Supports Euler and fourth-order Runge-Kutta (RK4) integration.
 */

#include <vector>
#include <string>
#include <assert.h>
#include <iostream>

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1> dim2;

/**
 * @brief Damped Oscillator — 2D competitive nonlinear ODE for benchmarking.
 *
 * Used as a simple testbed for the VBI inference pipeline (simulation-based
 * inference of parameters a and b).
 */
class DO
{
private:
    double dt;       ///< Integration time step [s]
    double PAR_a;    ///< Self-damping coefficient for population x
    double PAR_b;    ///< Self-damping coefficient for population y

    size_t dimension;  ///< State-space dimension (= 2)
    size_t num_steps;  ///< Total integration steps = (t_final - t_initial) / dt
    double t_final;    ///< Simulation end time [s]
    double t_initial;  ///< Simulation start time [s]

    dim1 times;        ///< Time array of length num_steps
    dim2 states;       ///< State trajectory, shape num_steps × 2
    dim1 initial_state;///< Initial state [x0, y0]

public:
    /**
     * @brief Construct a DO simulator.
     *
     * @param dt        Integration time step [s].
     * @param a         Self-damping coefficient for x.
     * @param b         Self-damping coefficient for y.
     * @param t_initial Simulation start time [s].
     * @param t_final   Simulation end time [s].
     * @param y         Initial state vector [x0, y0] (length 2).
     */
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

    /**
     * @brief Evaluate the system right-hand side.
     *
     * ```
     *   dxdt[0] = x - x*y - a*x^2
     *   dxdt[1] = x*y - y - b*y^2
     * ```
     *
     * @param x     Current state [x, y].
     * @param dxdt  Output derivative vector (pre-allocated, length 2).
     * @param t     Current time [s] (unused; kept for API consistency).
     */
    void derivative(const vector<double> &x,
             vector<double> &dxdt,
             const double t)
    {
        dxdt[0] = x[0] - x[0] * x[1] - PAR_a * x[0] * x[0];
        dxdt[1] = x[0] * x[1] - x[1] - PAR_b * x[1] * x[1];
    }

    /**
     * @brief Run the full simulation using the Euler scheme.
     *
     * Records the full state trajectory in @c states and the time array
     * in @c times.
     */
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

    /**
     * @brief Perform one Euler step, updating @p y in-place.
     * @param y  State vector [x, y], modified in-place.
     * @param t  Current time [s].
     */
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

    /**
     * @brief Run the full simulation using the RK4 scheme.
     *
     * Records the full state trajectory in @c states and the time array
     * in @c times.
     */
    void rk4Integrate()
    {
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

    /**
     * @brief Perform one fourth-order Runge-Kutta step, updating @p y in-place.
     * @param y  State vector [x, y], modified in-place.
     * @param t  Current time [s].
     */
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

    /**
     * @brief Return the full state trajectory.
     * @return dim2 of shape num_steps × 2.
     */
    dim2 get_coordinates()
    {
        return states;
    }

    /**
     * @brief Return the time array.
     * @return dim1 of length num_steps.
     */
    dim1 get_times()
    {
        return times;
    }
};
