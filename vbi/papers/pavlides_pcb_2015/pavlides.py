import os
import os.path
import numpy as np
from numpy import pi
from os.path import join
import matplotlib.pyplot as plt
from jitcdde import jitcdde, y, t
from symengine import sin, cos, Symbol, symarray, exp
import warnings

warnings.filterwarnings("ignore")


class Pav:
    """
    This class represents Wilson-Cowan model for Parkinson's disease.

    Reference:
    - Pavlides, A., Hogan, S.J. and Bogacz, R., 2015. Computational models describing possible mechanisms for generation of excessive beta oscillations in Parkinson's disease. PLoS computational biology, 11(12), p.e1004609.
    """

    def __init__(self, par: dict = {}):

        _par = self.get_default_params()
        _par.update(par)
        for item in _par.items():
            if item[0] not in _par["control"]:
                setattr(self, item[0], item[1])
                
        self.control_pars = []
        if len(_par["control"]) > 0:    
            for i in _par["control"]:    
                value = Symbol(i)
                setattr(self, i, value)
                self.control_pars.append(value)

        if not "modulename" in par.keys():
            self.modulename = "pav"
        os.makedirs(self.output, exist_ok=True)

    def get_default_params(self):
        """
        Return a dictionary of default parameters for the model.
        """
        par = {
            "control": "",  # list of control parameters
            "verbose": False,  # print compilation information
            "openmp": False,  # use openmp
            "output": "output",  # output directory
            "initial_state": None,  # initial state of the system
            "t_end": 1000.0,  # end time of the simulation
            "t_cut": 0.0,  # cut time of the simulation
            "seed": None,  # seed for random number generator
            "n_components": 4,  # number of components in the system
            "interval": 0.1,  # interval for saving the state of the system
            
            "Tsg": 6.0,  # ms delay between subthalamic and globus pallidus
            "Tgs": 6.0,  # ms delay between globus pallidus and subthalamic
            "Tgg": 4.0,  # ms delay between globus pallidus and globus pallidus
            "Tcs": 5.5,  # ms delay between cortex and subthalamic
            "Tsc": 21.5,  # ms delay between subthalamic and cortex
            "Tcc": 4.65,  # ms delay between cortex and cortex
            
            "taus": 12.8,  # ms time constant for subthalamic
            "taug": 20.0,  # ms time constant for globus pallidus
            "taue": 11.59,  # ms time constant for excitatory neurons
            "taui": 13.02,  # ms time constant for inhibitory neurons
            
            "Ms": 300.0/1000,  # spk/ms maximum firing rate of subthalamic
            "Mg": 400.0/1000,  # spk/ms maximum firing rate of globus pallidus
            "Me": 75.77/1000,  # spk/ms maximum firing rate of excitatory neurons
            "Mi": 205.72/1000,  # spk/ms maximum firing rate of inhibitory neurons
            
            "Bs": 10.0/1000,  # spk/ms baseline firing rate of subthalamic
            "Bg": 20.0/1000,  # spk/ms baseline firing rate of globus pallidus
            "Be": 17.85/1000,  # spk/ms population firing rate of excitatory neurons
            "Bi": 9.87/1000,  # spk/ms population firing rate of inhibitory neurons
            
            "C": 172.18/1000,  # spk/s external input to cortex
            "Str": 8.46/1000,  # spk/s external input to striatum
            
            "wgs": 1.33,  # synaptic weight from globus pallidus to subthalamic
            "wsg": 4.87,  # synaptic weight from subthalamic to globus pallidus
            "wgg": 0.53,  # synaptic weight from globus pallidus to globus pallidus
            "wcs": 9.97,  # synaptic weight from cortex to subthalamic
            "wsc": 8.93,  # synaptic weight from subthalamic to cortex
            "wcc": 6.17,  # synaptic weight from cortex to cortex
        }
        return par


    def sys_eqs(self):
        
        inS = self.wcs * y(2, t - self.Tcs) - self.wgs * y(1, t - self.Tgs)
        inG = self.wsg * y(0, t - self.Tsg) - self.wgg * y(1, t - self.Tgg) - self.Str
        inE = -self.wsc * y(0, t - self.Tsc) - self.wcc * y(3, t - self.Tcc) + self.C
        inI = self.wcc * y(2, t - self.Tcc)
        
        yield ((self.Ms/((1+exp(-4*inS/self.Ms)*((self.Ms-self.Bs)/self.Bs)))) - y(0))*(1/self.taus)
        yield ((self.Mg/((1+exp(-4*inG/self.Mg)*((self.Mg-self.Bg)/self.Bg)))) - y(1))*(1/self.taug)
        yield ((self.Me/((1+exp(-4*inE/self.Me)*((self.Me-self.Be)/self.Be)))) - y(2))*(1/self.taue)
        yield ((self.Mi/((1+exp(-4*inI/self.Mi)*((self.Mi-self.Bi)/self.Bi)))) - y(3))*(1/self.taui)
        

    def compile(self, **kwargs):
        control_pars = self.control_pars if len(self.control_pars) > 0 else ()
        I = jitcdde(
            self.sys_eqs,
            n=self.n_components,
            verbose=self.verbose,
            control_pars=control_pars,
        )
        I.compile_C(omp=self.openmp, **kwargs)
        I.save_compiled(overwrite=True, destination=join(self.output, self.modulename))

    def set_initial_state(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        initial_state = np.zeros(4)
        return initial_state

    def run(
        self,
        par=[],
        disc="step_on",
        step=0.001,
        propagations=1,
        min_distance=1e-5,
        max_step=None,
        shift_ratio=1e-4,
        **integrator_params
    ):
        """
        integrate the system of equations and return the
        computed state of the system after integration and times

        Parameters
        ------------

        par : list
            values of control parameters in order of appearance in `control`
        disc : str
            type of discontinuities handling. The default value is blind
                - step_on [step_on_discontinuities]
                - blind   [integrate_blindly]
                - adjust  [adjust_diff]
        step : float
            argument for integrate_blindly aspired step size. The actual step size may be slightly adapted to make it divide the integration time. If `None`, `0`, or otherwise falsy, the maximum step size as set with `max_step` of `set_integration_parameters` is used.

        propagations : int
            argument for step_on_discontinuities:  how often the discontinuity has to propagate to before it's considered smoothed.
        min_distance : float
                argument for step_on_discontinuities: If two required steps are closer than this, they will be treated as one.
        max_step : float
            argument for step_on_discontinuities: Retired parameter. Steps are now automatically adapted.
        shift_ratio : float
            argument for adjust_diff. Performs a zero-amplitude (backwards) `jump` whose `width` is `shift_ratio` times the distance to the previous anchor into the past. See the documentation of `jump` for the caveats of this and see `discontinuities` for more information on why you almost certainly need to use this or an alternative way to address initial discontinuities.

        Return : dict(t, x)
            - **t** times
            - **x** coordinates.
        """

        if self.initial_state is None:
            self.initial_state = self.set_initial_state(self.seed)

        I = jitcdde(
            self.sys_eqs,
            n=4,
            control_pars=self.control_pars,
            module_location=join(self.output, self.modulename + ".so"),
        )
        I.set_integration_parameters(**integrator_params)
        I.constant_past(self.initial_state, time=0.0)

        if disc == "blind":
            I.integrate_blindly(self.initial_state, step=step)
        elif disc == "step_on":
            I.step_on_discontinuities(
                propagations=propagations, min_distance=min_distance, max_step=max_step
            )
        else:
            I.adjust_diff(shift_ratio=shift_ratio)

        if len(self.control_pars) > 0:
            I.set_parameters(par)
        tcut = max(self.t_cut, I.t)
        times = tcut + np.arange(0, self.t_end - tcut, self.interval)

        x = np.zeros((len(times), self.n_components))
        for i in range(len(times)):
            x[i, :] = I.integrate(times[i])

        return {"t": times, "x": x}

if __name__ == "__main__":

    par = {"control": "",
           "output": "output"
           }
    ode = Pav(par=par)
    ode.compile()
    data = ode.run(disc="step_on")
    times = data["t"]
    x = data["x"]
    
    print("Times: ", times.shape)
    print("Coordinates: ", x.shape)
    plt.plot(times, x)
    plt.legend(["Subthalamic", "Globus Pallidus", "Excitatory", "Inhibitory"])
    plt.show()    
