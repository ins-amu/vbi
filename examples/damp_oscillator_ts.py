import numpy as np 
import matplotlib.pyplot as plt
from vbi.models.cpp.damp_oscillator import DO_cpp



parameters = {
        "a": 0.1,
        "b": 0.05,
        "dt": 0.01,
        "t_start": 0,
        "method": "rk4",
        "t_end": 100.0,
        "t_transition": 20,
        "output": "output",
        "initial_state": [0.5, 1.0],
    }

ode = DO_cpp(parameters)
print(ode())

sol = ode.simulate()
t = sol["t"]
x = sol["x"]

plt.style.use("ggplot")
plt.plot(t, x[:, 0], label='$\\theta$')
plt.plot(t, x[:, 1], label='$\omega$')
plt.xlabel("t")
plt.ylabel("x")
plt.legend()
plt.tight_layout()
plt.show()

