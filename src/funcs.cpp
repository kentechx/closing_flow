#include <pybind11/stl_bind.h>
#include "per_face_prin_curvature.h"

namespace py = pybind11;
PYBIND11_MODULE(funcs, m){
    m.doc() = "closing flow functions";
    m.def("per_face_prin_curvature", &per_face_prin_curvature, "per_face_prin_curvature");
}
