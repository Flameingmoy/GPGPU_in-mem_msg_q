#include <pybind11/pybind11.h>
#include "gpuqueue/queue.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() = "gpuqueue core bindings (scaffold)";

  m.def("version", []() { return gpuqueue::version(); },
        "Return version string of the gpuqueue core");

  m.def("init", &gpuqueue::init, py::arg("device") = 0,
        "Initialize the gpuqueue core for the specified CUDA device");

  m.def("shutdown", &gpuqueue::shutdown,
        "Shutdown the gpuqueue core and release resources");
}
