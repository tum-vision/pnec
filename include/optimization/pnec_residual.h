/**
 BSD 3-Clause License

 This file is part of the PNEC project.
 https://github.com/tum-vision/pnec

 Copyright (c) 2022, Dominik Muhle.
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OPTIMIZATION_PNEC_RESIDUAL_H_
#define OPTIMIZATION_PNEC_RESIDUAL_H_

#include <math.h>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include <ceres/ceres.h>

#include "common.h"

namespace pnec {
namespace residual {

struct PNECResidualHost {
  PNECResidualHost(Eigen::Vector3d bv_1, Eigen::Vector3d bv_2,
                   Eigen::Matrix3d cov, double regularization)
      : bv_1_{bv_1}, bv_2_{bv_2}, cov_{cov}, regularization_{regularization} {}
  template <typename T>
  bool operator()(const T *const theta_ptr, const T *const phi_ptr,
                  const T *const orientation_ptr, T *residual_ptr) const {
    // map the pointers to Eigen quaternions for easier use
    const Eigen::Matrix<T, 3, 1> translation(
        std::sin(theta_ptr[0]) * std::cos(phi_ptr[0]),
        std::sin(theta_ptr[0]) * std::sin(phi_ptr[0]), std::cos(theta_ptr[0]));
    Eigen::Map<const Eigen::Quaternion<T>> orientation(orientation_ptr);
    const Eigen::Matrix<T, 3, 3> rotation_m = orientation.toRotationMatrix();
    const Eigen::Matrix<double, 3, 3> bv_1_hat =
        pnec::common::SkewFromVector(rotation_m * bv_1_);

    residual_ptr[0] =
        (translation.transpose() * bv_1_.cross(rotation_m * bv_2_))(0, 0) /
        std::sqrt((translation.transpose() * bv_1_hat * cov_ *
                   bv_1_hat.transpose() * translation)(0, 0) +
                  regularization_);
    return true;
  }

private:
  const Eigen::Matrix<double, 3, 1> bv_1_;
  const Eigen::Matrix<double, 3, 1> bv_2_;
  const Eigen::Matrix<double, 3, 3> cov_;
  double regularization_;
};

struct PNECResidualTarget {
  PNECResidualTarget(Eigen::Vector3d bv_1, Eigen::Vector3d bv_2,
                     Eigen::Matrix3d cov, double regularization)
      : bv_1_{bv_1}, bv_2_{bv_2}, cov_{cov}, regularization_{regularization} {}
  template <typename T>
  bool operator()(const T *const theta_ptr, const T *const phi_ptr,
                  const T *const orientation_ptr, T *residual_ptr) const {
    // map the pointers to Eigen quaternions for easier use
    const Eigen::Matrix<T, 3, 1> translation(
        std::sin(theta_ptr[0]) * std::cos(phi_ptr[0]),
        std::sin(theta_ptr[0]) * std::sin(phi_ptr[0]), std::cos(theta_ptr[0]));
    Eigen::Map<const Eigen::Quaternion<T>> orientation(orientation_ptr);
    const Eigen::Matrix<T, 3, 3> rotation_m = orientation.toRotationMatrix();
    const Eigen::Matrix<double, 3, 3> bv_1_hat =
        pnec::common::SkewFromVector(bv_1_);

    residual_ptr[0] =
        (translation.transpose() * bv_1_.cross(rotation_m * bv_2_))(0, 0) /
        std::sqrt((translation.transpose() * bv_1_hat * rotation_m * cov_ *
                   rotation_m.transpose() * bv_1_hat.transpose() *
                   translation)(0, 0) +
                  regularization_);
    return true;
  }

private:
  const Eigen::Matrix<double, 3, 1> bv_1_;
  const Eigen::Matrix<double, 3, 1> bv_2_;
  const Eigen::Matrix<double, 3, 3> cov_;
  double regularization_;
};

struct PNECSymmetrical {
  PNECSymmetrical(Eigen::Vector3d bv_1, Eigen::Vector3d bv_2,
                  Eigen::Matrix3d cov_1, Eigen::Matrix3d cov_2,
                  double regularization)
      : bv_1_{bv_1}, bv_2_{bv_2}, cov_1_{cov_1}, cov_2_{cov_2},
        regularization_{regularization} {}
  template <typename T>
  bool operator()(const T *const theta_ptr, const T *const phi_ptr,
                  const T *const orientation_ptr, T *residual_ptr) const {
    // map the pointers to Eigen quaternions for easier use
    const Eigen::Matrix<T, 3, 1> translation(
        std::sin(theta_ptr[0]) * std::cos(phi_ptr[0]),
        std::sin(theta_ptr[0]) * std::sin(phi_ptr[0]), std::cos(theta_ptr[0]));
    Eigen::Map<const Eigen::Quaternion<T>> orientation(orientation_ptr);
    const Eigen::Matrix<T, 3, 3> rotation_m = orientation.toRotationMatrix();
    const Eigen::Matrix<double, 3, 3> bv_1_hat =
        pnec::common::SkewFromVector(bv_1_);
    const Eigen::Matrix<double, 3, 3> rot_bvs_2_hat =
        pnec::common::SkewFromVector(rotation_m * bv_2_);

    residual_ptr[0] =
        (translation.transpose() * bv_1_.cross(rotation_m * bv_2_))(0, 0) /
        std::sqrt((translation.transpose() *
                   (bv_1_hat * rotation_m * cov_2_ * rotation_m.transpose() *
                        bv_1_hat.transpose() +
                    rot_bvs_2_hat * cov_1_ * rot_bvs_2_hat.transpose()) *
                   translation)(0, 0) +
                  regularization_);
    return true;
  }

private:
  const Eigen::Matrix<double, 3, 1> bv_1_;
  const Eigen::Matrix<double, 3, 1> bv_2_;
  const Eigen::Matrix<double, 3, 3> cov_1_;
  const Eigen::Matrix<double, 3, 3> cov_2_;
  double regularization_;
};

} // namespace residual
} // namespace pnec

#endif // OPTIMIZATION_PNEC_RESIDUAL_H_