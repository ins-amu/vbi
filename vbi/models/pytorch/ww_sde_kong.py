from os.path import join
from copy import copy
from tqdm import tqdm
import numpy as np
import torch
import time
import math
import csv
import vbi
import gc


class BoldParams:
    def __init__(self, par={}):

        self._par = self.get_params()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

    def check_parameters(self, par: dict):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")

    def get_params(self):

        p_costant = 0.34
        k_1 = 4.3 * 28.265 * 3 * 0.0331 * p_costant
        k_2 = 0.47 * 110 * 0.0331 * p_costant
        par = {
            "beta": 0.65,
            "gamma": 0.41,
            "tau": 0.98,
            "alpha": 0.33,
            "v_0": 0.02,
            "p_constant": p_costant,
            "k_1": k_1,
            "k_2": k_2,
            "k_3": 0.53,
        }

        return par


class WW_SDE_KONG:

    def __init__(self, par={}, bold_par={}, path=None):

        if path is None:
            path = join(vbi.__file__)
            path = path.replace("__init__.py", "")
            path = join(path, "models/pytorch/data")
        self.input_path = path
        self._par = self.get_default_params()
        self.valid_parameters = list(self._par.keys())
        self.check_parameters(par)
        self._par.update(par)

        for item in self._par.keys():
            setattr(self, item, self._par[item])

        BP = BoldParams(bold_par)
        self.bp = BP._par

        self.myelin_data, self.gradient_data = self.get_myelin_gradient()
        self.nn = self.n_node = self.myelin_data.shape[0]
        # dim = self.n_node * 3 + 1

    def get_default_params(self):

        engine = "cpu"
        self.device = "cuda" if engine == "gpu" else "cpu"

        data = np.load(join(self.input_path, "default_parameters.npz"))
        self.weights = self.get_sc(device=self.device)

        nn = self.weights.shape[0]
        inp = data["input_para"][:, 0]
        w = inp[:nn]
        I0 = inp[nn : 2 * nn]
        g_true = inp[2 * nn]
        s = inp[2 * nn + 1 :]

        return {
            "G": g_true,
            "J": 0.2609,
            "w": w,
            "s": s,
            "I0": I0,
            "a": 270.0,
            "b": 108.0,
            "d": 0.154,
            "tau_s": 0.1,
            "gamma_s": 0.641,
            "t_end": 5.0 * 60.0,
            "t_cut": 2.0 * 60.0,
            "tr": 0.72,
            "dt": 0.01,
            "n_sim": 1,
            "weights": self.weights,
            "engine": engine,
            "device": self.device,
            "dtype": torch.float64,
        }

    def f_mfm(self, y_t):

        x = (
            self.J * self.w * y_t
            + self.J * self.G * torch.mm(self.weights, y_t)
            + self.I0
        )
        # Population firing rate
        H = (self.a * x - self.b) / (1 - torch.exp(-self.d * (self.a * x - self.b)))
        # Synaptic activity
        dy = -1 / self.tau_s * y_t + self.gamma_s * (1 - y_t) * H
        return dy

    def f_rfMRI(self, y_t, F):
        """
        This fucntion is to implement the hemodynamic model

        parameters
        ----------
        y_t: torch.Tensor
            N*M matrix represents synaptic gating variable
            N is the number of ROI
            M is the number of candidate parameter sets
        F:  torch.Tensor
            Hemodynamic activity variables

        Returns
        -------
        dF: torch.Tensor
            Derivatives of hemodynamic activity variables
        """
        beta = self.bp["beta"]
        gamma = self.bp["gamma"]
        tau = self.bp["tau"]
        alpha = self.bp["alpha"]
        p_constant = self.bp["p_constant"]
        n_nodes = y_t.shape[0]
        n_set = y_t.shape[1]

        # Calculate derivatives
        if self.engine == "gpu":
            dF = torch.zeros((n_nodes, n_set, 4), dtype=self.dtype).cuda()
        else:
            dF = torch.zeros((n_nodes, n_set, 4), dtype=self.dtype)

        dF[:, :, 0] = y_t - beta * F[:, :, 0] - gamma * (F[:, :, 1] - 1)
        dF[:, :, 1] = F[:, :, 0]
        dF[:, :, 2] = 1 / tau * (F[:, :, 1] - F[:, :, 2] ** (1 / alpha))
        dF[:, :, 3] = (
            1
            / tau
            * (
                F[:, :, 1] / p_constant * (1 - (1 - p_constant) ** (1 / F[:, :, 1]))
                - F[:, :, 3] / F[:, :, 2] * F[:, :, 2] ** (1 / alpha)
            )
        )
        return dF

    def run(self):
        """
        Function used to generate the simulated BOLD signal using mean field model
        and hemodynamic model
        Each parameter set is ussed to simulated multiple times to get stable
        result
        """

        nn = self.nn
        ns = self.n_sim
        dt = self.dt
        n_dup = 1
        dtype = self.dtype
        engine = self.engine
        dt_tensor = torch.tensor(self.dt, dtype=dtype)

        self.prepare_input()

        p_costant = self.bp["p_constant"]
        v_0 = self.bp["v_0"]
        k_1 = self.bp["k_1"]
        k_2 = self.bp["k_2"]
        k_3 = self.bp["k_3"]

        if self.engine == "gpu":
            k_p = torch.arange(0.0, self.t_end + dt, dt).cuda()
        else:
            k_p = torch.arange(0.0, self.t_end + dt, dt)

        nt_samples = k_p.shape[0]
        device = "cuda" if engine == "gpu" else "cpu"
        y_t = torch.zeros((nn, ns), dtype=dtype, device=device)
        d_y = torch.zeros((nn, ns), dtype=dtype, device=device)
        f_mat = torch.ones((nn, ns, 4), dtype=dtype, device=device)
        z_t = torch.zeros((nn, ns), dtype=dtype, device=device)
        f_t = torch.ones((nn, ns), dtype=dtype, device=device)
        v_t = torch.ones((nn, ns), dtype=dtype, device=device)
        q_t = torch.ones((nn, ns), dtype=dtype, device=device)

        f_mat[:, :, 0] = z_t
        y_t[:, :] = 0.001
        # Wiener process
        w_coef = self.s / math.sqrt(0.001)
        if w_coef.shape[0] == 1:
            w_coef = w_coef.repeat(nn, 1)

        d_w = torch.sqrt(dt_tensor) * torch.randn(
            n_dup, nn, nt_samples + 1000, dtype=dtype, device=device
        )

        y_bold = torch.zeros(
            (nn, ns, int(nt_samples / (self.tr / dt) + 1)),
            dtype=torch.float32,
            device=device,
        )

        # Warm up
        for i in range(1000):
            d_y = self.f_mfm(y_t)
            noise_level = (
                d_w[:, :, i].repeat(1, 1, ns).contiguous().view(-1, nn)
            )  # repeat the noise level for all simulations (ns) at one time step
            y_t = y_t + d_y * dt + w_coef * torch.transpose(noise_level, 0, 1)

        # Main body: calculation
        count = 0
        for i in tqdm(range(nt_samples)):
            d_y = self.f_mfm(y_t)
            noise_level = d_w[:, :, i + 1000].repeat(1, 1, ns).contiguous().view(-1, nn)
            y_t = y_t + d_y * dt + w_coef * torch.transpose(noise_level, 0, 1)
            d_f = self.f_rfMRI(y_t, f_mat)
            f_mat = f_mat + d_f * dt
            z_t, f_t, v_t, q_t = torch.chunk(f_mat, 4, dim=2)
            y_bold_temp = (
                100
                / p_costant
                * v_0
                * (k_1 * (1 - q_t) + k_2 * (1 - q_t / v_t) + k_3 * (1 - v_t))
            )

            y_bold[:, :, count] = y_bold_temp[:, :, 0]
            count = count + ((i + 1) % (self.tr / dt) == 0) * 1

        # Downsampling
        cut_index = int(self.t_cut / self.tr)

        t = k_p[cut_index + 1 : y_bold.shape[2]]
        if t.is_cuda:
            t = t.cpu().numpy()

        if engine == "gpu":
            y_bold_cpu = y_bold.cpu()
            y_bold_cpu = y_bold_cpu[:, :, cut_index + 1 : y_bold.shape[2]]
            del y_bold
            torch.cuda.empty_cache()
            gc.collect()
            return {"t": t, "x": y_bold_cpu.numpy()}

        return {
            "t": t,
            "x": y_bold[:, :, cut_index + 1 : y_bold.shape[2]].numpy(),
        }

    def get_sc(self, device: str = "cpu", dtype=torch.float64):
        sc_mat_raw = csv_matrix_read(join(self.input_path, "input", "sc_train.csv"))
        sc_mat = sc_mat_raw / sc_mat_raw.max() * 0.2
        sc_mat = torch.from_numpy(sc_mat).type(dtype)

        if device == "cuda":
            sc_mat = sc_mat.cuda()
        return sc_mat

    def get_myelin_gradient(self):
        myelin_data = csv_matrix_read(join(self.input_path, "input", "myelin.csv"))
        myelin_data = myelin_data[:, 0]
        gradient_data = csv_matrix_read(
            join(self.input_path, "input", "rsfc_gradient.csv")
        )
        gradient_data = gradient_data[:, 0]
        return myelin_data, gradient_data

    def check_parameters(self, par: dict):
        for key in par.keys():
            if key not in self.valid_parameters:
                raise ValueError(f"Invalid parameter {key:s} provided.")

    def prepare_input(self):

        if isinstance(self.weights, np.ndarray):
            self.weights = torch.from_numpy(self.weights)
        if self.weights.dtype != self.dtype:
            self.weights = self.weights.type(self.dtype)

        if (self.engine == "gpu") and (not is_cuda(self.weights)):
            self.weights = self.weights.cuda()

        self.w = to_vector_2d(
            self.w, self.nn, self.n_sim, dtype=self.dtype, engine=self.engine
        )

        self.I0 = to_vector_2d(
            self.I0, self.nn, self.n_sim, dtype=self.dtype, engine=self.engine
        )
        self.s = to_vector_2d(
            self.s, self.nn, self.n_sim, dtype=self.dtype, engine=self.engine
        )
        self.G = to_vector(self.G, self.n_sim, dtype=self.dtype, engine=self.engine)
        self.J = to_vector(self.J, self.n_sim, dtype=self.dtype, engine=self.engine)


def to_vector(x, ns, dtype=torch.float64, engine="cpu"):
    """
    Converts the input `x` to a tensor of specified size and type.

    Parameters
    ------------------
    x : array-like or torch.Tensor
        The input data to be converted to a tensor.
    ns : int
        The size to which the tensor should be repeated if `x` is a single element.
    dtype : torch.dtype, optional
        The desired data type of the tensor. Default is `torch.float64`.
    engine : str, optional
        The computation engine to use, either `"cpu"` or `"gpu"`. Default is `"cpu"`.

    Returns
    ------------------
    torch.Tensor
        The converted tensor with the specified size and type.

    Raises
    ------------------
    AssertionError
        If the size of `x` is not 1 and does not match `n`.
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    if x.ndim == 0:  # scalar
        x = x.repeat(1, ns)
    elif x.ndim == 1:
        assert x.size(0) == ns, f"input size must be 1 or {ns}"
        x = x.view(1, ns)
    if engine == "gpu":
        x = x.cuda()
    return x


def to_vector_2d(x, nn, ns, dtype=torch.float64, engine="cpu"):
    """
    Convert input `x` to a tensor of specified size and type.

    Parameters
    ------------------
    x : array-like or torch.Tensor
        The input data to be converted to a tensor.
    nn : int
        The number of nodes.
    ns : int
        The number of simulations.
    dtype : torch.dtype, optional
        The desired data type of the tensor. Default is `torch.float64`.
    engine : str, optional
        The computation engine to use, either `"cpu"` or `"gpu"`. Default is `"cpu"`.

    Returns
    ------------------
    torch.Tensor
        The converted tensor with the specified size and type.
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=dtype)
    if x.ndim == 0:  # scalar
        x = x.repeat(nn, ns)
    elif x.ndim == 1:
        assert x.size(0) == nn, f"input size must be 1 or {nn}"
        x = x.view(nn, 1).repeat(1, ns)

    if engine == "gpu":
        x = x.cuda()
    return x


# =============================================================================
### Functions for heterogenous parameter estimation corresponding to :
### Kong, et.al., 2021. Nature communications, 12(1), p.6373.


def csv_matrix_read(filename):
    """
    This function is used to read csv file into a numpy array
    Args:
        filename:  input csv file
    Returns:
        out_array: output numpy array
    """

    csv_file = open(filename, "r")
    read_handle = csv.reader(csv_file)
    out_list = []
    R = 0
    for row in read_handle:
        out_list.append([])
        for col in row:
            out_list[R].append(float(col))
        R = R + 1
    out_array = np.array(out_list)
    csv_file.close()
    return out_array


def is_cuda(tensor: torch.Tensor):
    if not isinstance(tensor, torch.Tensor):
        print("not a tensor")
        return False
    return tensor.is_cuda


def get_ranges():
    return {
        "w": np.array([[0.3, 0.7], [-0.3, -0.04], [0.0, 0.06]]),
        "i": np.array([[0.2, 0.32], [-0.006, 0.02], [-0.02, -0.003]]),
        "g": np.array([1.0, 10.0]),
        "s": np.array([[0.004, 0.006], [-0.001, 0.001], [0.0, 0.0008]]),
    }


def get_prior_limits():

    ranges = get_ranges()
    prior_min = np.hstack(
        [ranges["w"][:, 0], ranges["i"][:, 0], ranges["g"][0], ranges["s"][:, 0]]
    )
    prior_max = np.hstack(
        [ranges["w"][:, 1], ranges["i"][:, 1], ranges["g"][1], ranges["s"][:, 1]]
    )

    return prior_min, prior_max


def sample_prior(My, Gr):
    rand = np.random.uniform
    ranges = get_ranges()
    wrange = ranges["w"]
    irange = ranges["i"]
    grange = ranges["g"]
    srange = ranges["s"]

    f = lambda x, y, z: x * My + y * Gr + z

    abc_w = [rand(*wrange[0]), rand(*wrange[1]), rand(*wrange[2])]
    abc_i = [rand(*irange[0]), rand(*irange[1]), rand(*irange[2])]
    abc_s = [rand(*srange[0]), rand(*srange[1]), rand(*srange[2])]
    g = rand(*grange)

    w = f(*abc_w)
    i = f(*abc_i)
    s = f(*abc_s)

    return w, i, s, g, abc_w, abc_i, abc_s


def get_search_range(nn):
    """
    Returns the search range for the given number of nodes.
    """
    wrange = [0, 1]
    irange = [0, 0.5]
    grange = [1, 10]
    srange = [0.0005, 0.01]

    d = 3 * nn + 1
    search_range = np.zeros((d, 2))
    search_range[:nn, :] = wrange
    search_range[nn : 2 * nn, :] = irange
    search_range[2 * nn, :] = grange
    search_range[2 * nn + 1 :, :] = srange

    return search_range


def check_range(w, ii, g, s):
    aw, bw, cw = w
    ai, bi, ci = ii
    asig, bsig, csig = s

    wrange = np.array([[0.3, 0.7], [-0.3, -0.04], [0.0, 0.06]])
    irange = np.array([[0.2, 0.32], [-0.006, 0.02], [-0.02, -0.003]])
    srange = np.array([[0.004, 0.006], [-0.001, 0.0001], [0.0, 0.0008]])
    grange = np.array([5, 7])

    if aw < wrange[0, 0] or aw > wrange[0, 1]:
        return False
    if bw < wrange[1, 0] or bw > wrange[1, 1]:
        return False
    if cw < wrange[2, 0] or cw > wrange[2, 1]:
        return False
    if ai < irange[0, 0] or ai > irange[0, 1]:
        return False
    if bi < irange[1, 0] or bi > irange[1, 1]:
        return False
    if ci < irange[2, 0] or ci > irange[2, 1]:
        return False
    if asig < srange[0, 0] or asig > srange[0, 1]:
        return False
    if bsig < srange[1, 0] or bsig > srange[1, 1]:
        return False
    if csig < srange[2, 0] or csig > srange[2, 1]:
        return False
    if g < grange[0] or g > grange[1]:
        return False
    return True


def get_cmatrix(myelin_data, gradient_data):

    nn = myelin_data.shape[0]
    return np.vstack((myelin_data, gradient_data, np.ones(nn))).T


def get_invcc(cmatrix):
    invcc = np.linalg.inv(cmatrix.T @ cmatrix) @ cmatrix.T
    return invcc


def sample_prior(search_range, dim, cmatrix, invcc):
    init_para = np.zeros(dim)

    init_para = (
        np.random.uniform(0, 1, dim) * (search_range[:, 1] - search_range[:, 0])
        + search_range[:, 0]
    )
    nn = cmatrix.shape[0]
    w = invcc @ init_para[:nn]
    i = invcc @ init_para[nn : 2 * nn]
    g = init_para[2 * nn]
    s = invcc @ init_para[2 * nn + 1 :]
    return init_para, w, i, g, s


def get_init(myelin_data, gradient_data, highest_order, init_para):
    """
    This function is implemented to calculate the initial parametrized
    coefficients
    """

    n_node = myelin_data.shape[0]
    amatrix = np.zeros((n_node, highest_order + 1))
    bmatrix = np.zeros((n_node, highest_order + 1))
    for i in range(highest_order + 1):
        amatrix[:, i] = myelin_data ** (i)
        bmatrix[:, i] = gradient_data ** (i)
    cmatrix = np.hstack((amatrix, bmatrix[:, 1 : highest_order + 1]))
    para = np.linalg.inv(cmatrix.T @ cmatrix) @ cmatrix.T @ init_para
    return para, cmatrix


def make_input_para(theta, cmatrix, dim):
    input_para = np.zeros(dim)
    nn = cmatrix.shape[0]
    input_para[:nn] = cmatrix @ theta[:3]
    input_para[nn : nn * 2] = cmatrix @ theta[3:6]
    input_para[nn * 2] = theta[6]
    input_para[nn * 2 + 1 :] = cmatrix @ theta[7:]
    return input_para
