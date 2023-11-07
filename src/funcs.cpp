#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include "per_face_prin_curvature.h"

namespace py = pybind11;
PYBIND11_MODULE(funcs, m){
    m.doc() = "closing flow functions";
    m.def("per_face_prin_curvature", [](const Eigen::MatrixXd& V, const Eigen::MatrixXi& F){
        Eigen::MatrixXd PD1, PD2;
        Eigen::VectorXd PC1, PC2;
        per_face_prin_curvature(V, F, PD1, PD2, PC1, PC2);
        return std::make_tuple(PD1, PD2, PC1, PC2);
    }, "compute principal curvature per face");
}
