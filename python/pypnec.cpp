#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <nec_ceres.h>
#include <pnec_ceres.h>

namespace py = pybind11;

// #define STRINGIFY(x) #x
// #define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

Eigen::MatrixXd mat2(Eigen::MatrixXd &mat) {
  std::cout << mat << std::endl;
  return mat * 2;
}

std::vector<Eigen::MatrixXd> matrices(std::vector<Eigen::MatrixXd> mats) {
  std::vector<Eigen::MatrixXd> results;
  for (const auto &mat : mats) {
    std::cout << mat << std::endl;
    results.push_back(mat * 2);
  }
  return results;
}

Eigen::Matrix4d pyceres(std::vector<Eigen::Vector3d> host_bvs,
                        std::vector<Eigen::Vector3d> target_bvs,
                        std::vector<Eigen::Matrix3d> host_covariances,
                        std::vector<Eigen::Matrix3d> target_covariances,
                        Eigen::Matrix4d init_pose, double regularization) {
  pnec::optimization::PNECCeres optimizer;
  Sophus::SE3d sp_init_pose(init_pose);
  optimizer.InitValues(Eigen::Quaterniond(sp_init_pose.rotationMatrix()),
                       sp_init_pose.translation());
  optimizer.Optimize(host_bvs, target_bvs, host_covariances, target_covariances,
                     regularization);

  return optimizer.Result().matrix();
}

Eigen::Matrix4d pyceresnec(std::vector<Eigen::Vector3d> host_bvs,
                           std::vector<Eigen::Vector3d> target_bvs,
                           Eigen::Matrix4d init_pose) {
  pnec::optimization::NECCeres optimizer;
  Sophus::SE3d sp_init_pose(init_pose);
  optimizer.InitValues(Eigen::Quaterniond(sp_init_pose.rotationMatrix()),
                       sp_init_pose.translation());
  optimizer.Optimize(host_bvs, target_bvs);

  return optimizer.Result().matrix();
}

PYBIND11_MODULE(pypnec, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("add", &add, "A function that adds two numbers");
  m.def("mat2", &mat2, "A function prints out and multiplies by two");
  m.def("matrices", &matrices, "A function prints out and multiplies by two");
  m.def("pyceres", &pyceres, "A function prints out and multiplies by two");
  m.def("pyceresnec", &pyceresnec,
        "A function prints out and multiplies by two");
}