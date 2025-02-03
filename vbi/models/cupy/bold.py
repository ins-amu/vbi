import numpy as np


class BoldStephan2008:

    def __init__(self, par: dict = {}) -> None:

        self._par = self.get_default_parameters()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for key, value in self._par.items():
            setattr(self, key, value)

    def _prepare(self, nn, ns, xp, n_steps, bold_decimate):
        s = xp.zeros((2, nn, ns), dtype=self.dtype)
        f = xp.zeros((2, nn, ns), dtype=self.dtype)
        ftilde = xp.zeros((2, nn, ns), dtype=self.dtype)
        vtilde = xp.zeros((2, nn, ns), dtype=self.dtype)
        qtilde = xp.zeros((2, nn, ns), dtype=self.dtype)
        v = xp.zeros((2, nn, ns), dtype=self.dtype)
        q = xp.zeros((2, nn, ns), dtype=self.dtype)
        vv = np.zeros((n_steps // bold_decimate, nn, ns), dtype="f")
        qq = np.zeros((n_steps // bold_decimate, nn, ns), dtype="f")
        s[0] = 1
        f[0] = 1
        v[0] = 1
        q[0] = 1
        ftilde[0] = 0
        vtilde[0] = 0
        qtilde[0] = 0

        return {
            "s": s,
            "f": f,
            "ftilde": ftilde,
            "vtilde": vtilde,
            "qtilde": qtilde,
            "v": v,
            "q": q,
            "vv": vv,
            "qq": qq,
        }

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")

    def get_default_parameters(self):

        theta0 = 41.0
        Eo = 0.42
        TE = 0.05
        epsilon = 0.36
        r0 = 26.0
        k1 = 4.3 * theta0 * Eo * TE
        k2 = epsilon * r0 * Eo * TE
        k3 = 1 - epsilon

        par = {
            "kappa": 0.7,
            "gamma": 0.5,
            "tau": 1.0,
            "alpha": 0.35,
            "epsilon": epsilon,
            "Eo": Eo,
            "TE": TE,
            "vo": 0.09,
            "r0": r0,
            "theta0": theta0,
            "rtol": 1e-6,
            "atol": 1e-9,
            "k1": k1,
            "k2": k2,
            "k3": k3,
        }
        return par

    def bold_step(self, r_in, s, f, ftilde, vtilde, qtilde, v, q, dt, P):

        kappa, gamma, alpha, tau, Eo = P
        ialpha = 1 / alpha

        s[1] = s[0] + dt * (r_in - kappa * s[0] - gamma * (f[0] - 1))
        f[0] = np.clip(f[0], 1, None)
        ftilde[1] = ftilde[0] + dt * (s[0] / f[0])
        fv = v[0] ** ialpha  # outflow
        vtilde[1] = vtilde[0] + dt * ((f[0] - fv) / (tau * v[0]))
        q[0] = np.clip(q[0], 0.01, None)
        ff = (1 - (1 - Eo) ** (1 / f[0])) / Eo  # oxygen extraction
        qtilde[1] = qtilde[0] + dt * ((f[0] * ff - fv * q[0] / v[0]) / (tau * q[0]))

        f[1] = np.exp(ftilde[1])
        v[1] = np.exp(vtilde[1])
        q[1] = np.exp(qtilde[1])

        f[0] = f[1]
        s[0] = s[1]
        ftilde[0] = ftilde[1]
        vtilde[0] = vtilde[1]
        qtilde[0] = qtilde[1]
        v[0] = v[1]
        q[0] = q[1]


class BoldTVB:

    def __init__(self):
        pass
