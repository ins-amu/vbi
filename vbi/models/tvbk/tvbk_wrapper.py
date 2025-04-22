import numpy as np
import collections
from vbi.models.tvbk.utils import prepare_vec, setup_connectivity

try:
    import tvbk as m

    TVBK_AVAILABLE = True
except ImportError:
    TVBK_AVAILABLE = False


class MPR:

    MPRTheta = collections.namedtuple(
        typename="MPRTheta", field_names="tau I Delta J eta G".split(" ")
    )
    num_svar = 2  # number of state variables
    num_parm = 6  # number of parameters

    def __init__(self, par: dict = {}) -> None:

        self._par = self.get_default_parameters()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.items():
            setattr(self, item[0], item[1])

        self.mpr_default_theta = self.MPRTheta(
            tau=self._par["tau"],
            I=self._par["I"],
            Delta=self._par["Delta"],
            J=self._par["J"],
            eta=self._par["eta"],
            G=self._par["G"],
        )

    def __str__(self) -> str:
        return f"MPR model with parameters: {self._par}"

    def get_default_parameters(self) -> dict:
        return {
            "tau": 1.0,
            "Delta": 1.0,
            "I": 0.0,
            "J": 15.0,
            "eta": -5.0,
            "G": 1.0,
            "num_batch": 1,
            "horizon": 256,
            "width": 8,
            "dt": 0.01,
            "dtype": np.float32,
            "weights": None,
            "delays": None,
            "num_node": None,
            "noise_amp": None, 
            "num_time": 1000,
            "decimate_rv": 10,
            "RECORD_RV": True,
            "RECORD_BOLD": False, # TODO: Add BOLD recording
        }

    def check_parameters(self, par):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")

    
    def initialize_buffers(self):

        total_volume = self.num_batch * self.num_node * self.horizon * self.width
        self.cx = m.Cx8s(self.num_node, self.horizon, self.num_batch)
        buf_val = (
            np.r_[: 1.0 : 1j * total_volume]
            .reshape(self.num_batch, self.num_node, self.horizon, self.width)
            .astype(self.dtype)
            * 4.0
        )
        self.cx.buf[:] = buf_val
        self.cx.cx1[:] = self.cx.cx2[:] = 0.0
        

    def prepare_input(self):

        width = self.width
        num_batch = self.num_batch
        num_parm = self.num_parm
        
        assert self.weights is not None, "weights must be provided"
        self.weights = np.array(self.weights)
        num_node = self.num_node = self.weights.shape[0]
        
        self.G = prepare_vec(self.G, num_batch, self.dtype)
        self.J = prepare_vec(self.J, num_batch, self.dtype)
        self.I = prepare_vec(self.I, num_batch, self.dtype)
        self.eta = prepare_vec(self.eta, num_batch, self.dtype)
        self.tau = prepare_vec(self.tau, num_batch, self.dtype)
        self.Delta = prepare_vec(self.Delta, num_batch, self.dtype)        
        self.noise_amp = np.array(self.noise_amp, self.dtype)
        self.p = np.zeros((num_batch, num_node, num_parm, width), self.dtype)
        
        self.p[:, :, 0, :] = self.tau
        self.p[:, :, 1, :] = self.I
        self.p[:, :, 2, :] = self.Delta
        self.p[:, :, 3, :] = self.J
        self.p[:, :, 4, :] = self.eta
        self.p[:, :, 5, :] = self.G
        
        self.initialize_buffers()
        self.conn = setup_connectivity(self.weights, self.delays)
        

    def run(self):
        
        self.prepare_input()

        x = np.zeros((self.num_batch, self.num_svar, self.num_node, self.width), self.dtype)
        y = np.zeros_like(x)
        z = np.zeros((self.num_batch, self.num_svar, 8), self.dtype)+ self.noise_amp
        seed = np.zeros((self.num_batch, 8, 4), np.uint64)
        num_samples = self.num_time // self.decimate_rv + 1
        
        if self.RECORD_RV:
            trace_c = np.zeros(
                (
                    num_samples,
                    self.num_batch,
                    self.num_svar,
                    self.num_node,
                    self.width,
                )
            )

        for i in range(num_samples):
            if self.RECORD_RV:
                m.step_mpr(
                    self.cx,
                    self.conn,
                    x,
                    y,
                    z, 
                    self.p,
                    i * self.decimate_rv,
                    self.decimate_rv,
                    self.dt,
                    seed
                )
            if self.RECORD_RV:
                trace_c[i] = x
            
            # TODO: Calculate BOLD signal
            if self.RECORD_BOLD:
                pass
                # add BOLD signal calculation pytorch code here
            
        
        return {
            "rv_t": ...,
            "rv_d": trace_c, 
            "fmri_t": ...,
            "fmri_d": None,
            }
        
