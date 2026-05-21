<%doc>
Mako template for the pybind11 bindings.
Context: module_name, cpp_filename, n_sv, n_nodes, n_params, n_noise_sv
</%doc>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "${cpp_filename}"

namespace py = pybind11;

PYBIND11_MODULE(${module_name}, m) {
    m.doc() = "VBI auto-generated C++ simulation module";

    m.def("n_sv",     []() { return vbi_gen::kNumSv;     });
    m.def("n_nodes",  []() { return vbi_gen::kNumNodes;  });
    m.def("n_params", []() { return vbi_gen::kNumParams; });
    m.def("n_cvar",   []() { return vbi_gen::kNumCvar;   });

    m.def("run_simulation",
        [](py::array_t<double,  py::array::c_style | py::array::forcecast> initial_state,
           py::array_t<double,  py::array::c_style | py::array::forcecast> weights,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> idelays,
           int            horizon,
           py::array_t<double,  py::array::c_style | py::array::forcecast> params,
           double         coup_a,
           double         coup_b,
           bool           has_delays,
           py::array_t<double,  py::array::c_style | py::array::forcecast> noise_data,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> noise_sv_indices,
           int            n_steps,
           int            record_every,
           int            t_cut_steps,
           py::array_t<double,  py::array::c_style | py::array::forcecast> stim_data,
           bool           has_stimulus)
        -> py::array_t<double>
        {
            const double*  noise_ptr = (noise_data.size() > 0) ? noise_data.data() : nullptr;
            const int*     nidx_ptr  = reinterpret_cast<const int*>(noise_sv_indices.data());
            int            n_noise   = static_cast<int>(noise_sv_indices.size());
            const double*  stim_ptr  = has_stimulus ? stim_data.data() : nullptr;

            std::vector<double> raw;
            {
                py::gil_scoped_release release;
                raw = vbi_gen::run_simulation(
                    initial_state.data(),
                    weights.data(),
                    idelays.data(),
                    horizon,
                    params.data(),
                    coup_a, coup_b,
                    has_delays,
                    noise_ptr, nidx_ptr, n_noise,
                    n_steps, record_every, t_cut_steps,
                    stim_ptr, has_stimulus);
            }

            // Reshape to (n_record, n_sv, n_nodes)
            std::size_t n_sv    = vbi_gen::kNumSv;
            std::size_t n_nodes = vbi_gen::kNumNodes;
            std::size_t n_rec   = raw.size() / (n_sv * n_nodes);

            py::array_t<double> out(
                {static_cast<py::ssize_t>(n_rec),
                 static_cast<py::ssize_t>(n_sv),
                 static_cast<py::ssize_t>(n_nodes)});
            std::copy(raw.begin(), raw.end(), out.mutable_data());
            return out;
        },
        py::arg("initial_state"),
        py::arg("weights"),
        py::arg("idelays"),
        py::arg("horizon"),
        py::arg("params"),
        py::arg("coup_a"),
        py::arg("coup_b"),
        py::arg("has_delays"),
        py::arg("noise_data"),
        py::arg("noise_sv_indices"),
        py::arg("n_steps"),
        py::arg("record_every")   = 1,
        py::arg("t_cut_steps")    = 0,
        py::arg("stim_data")      = py::array_t<double>(),
        py::arg("has_stimulus")   = false);
}
