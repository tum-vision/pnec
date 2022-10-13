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

#include "pnec_ceres.h"

#include "pnec_residual.h"

namespace pnec {
namespace optimization {

PNECCeres::PNECCeres() {
  orientation_ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  theta_ = 0.0;
  phi_ = 0.0;
  options_ = ceres::Solver::Options();
}

PNECCeres::PNECCeres(const Sophus::SE3d &init,
                     const ceres::Solver::Options &options)
    : options_{options} {
  orientation_ = init.unit_quaternion();
  pnec::common::AnglesFromVec(init.translation(), theta_, phi_);
}

PNECCeres::PNECCeres(const Eigen::Quaterniond &orientation, double theta,
                     double phi, const ceres::Solver::Options &options)
    : orientation_{orientation}, theta_{theta}, phi_{phi} {}

PNECCeres::PNECCeres(const Eigen::Quaterniond &orientation,
                     const Eigen::Vector3d &translation,
                     const ceres::Solver::Options &options)
    : orientation_{orientation}, options_{options} {
  pnec::common::AnglesFromVec(translation, theta_, phi_);
}

PNECCeres::~PNECCeres() {}

void PNECCeres::Optimize(const std::vector<Eigen::Vector3d> &bvs_1,
                         const std::vector<Eigen::Vector3d> &bvs_2,
                         const std::vector<Eigen::Matrix3d> &covs,
                         double regularization,
                         pnec::common::NoiseFrame noise_frame) {

  ceres::Problem problem;
  std::vector<double *> orientation_d =
      std::vector<double *>{orientation_.coeffs().data()};

  if (noise_frame == pnec::common::Host) {
    for (size_t i = 0; i < bvs_1.size(); i++) {

      ceres::CostFunction *cost_function =
          new ceres::NumericDiffCostFunction<pnec::residual::PNECResidualHost,
                                             ceres::CENTRAL, 1, 1, 1, 4>(
              new pnec::residual::PNECResidualHost(bvs_1[i], bvs_2[i], covs[i],
                                                   regularization));
      problem.AddResidualBlock(cost_function, nullptr, &theta_, &phi_,
                               orientation_.coeffs().data());
    }
  } else {
    for (size_t i = 0; i < bvs_1.size(); i++) {

      ceres::CostFunction *cost_function =
          new ceres::NumericDiffCostFunction<pnec::residual::PNECResidualTarget,
                                             ceres::CENTRAL, 1, 1, 1, 4>(
              new pnec::residual::PNECResidualTarget(bvs_1[i], bvs_2[i],
                                                     covs[i], regularization));
      problem.AddResidualBlock(cost_function, nullptr, &theta_, &phi_,
                               orientation_.coeffs().data());
    }
  }
  ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
  // std::cout << "here" << std::endl;

  problem.SetManifold(orientation_.coeffs().data(), quaternion_manifold);
  // problem.SetParameterization(orientation_.coeffs().data(),
  //                             new ceres::EigenQuaternionParameterization);

  ceres::Solve(options_, &problem, &summary_);
}

void PNECCeres::Optimize(const std::vector<Eigen::Vector3d> &bvs_1,
                         const std::vector<Eigen::Vector3d> &bvs_2,
                         const std::vector<Eigen::Matrix3d> &covs_1,
                         const std::vector<Eigen::Matrix3d> &covs_2,
                         double regularization) {

  ceres::Problem problem;
  std::vector<double *> orientation_d =
      std::vector<double *>{orientation_.coeffs().data()};

  for (size_t i = 0; i < bvs_1.size(); i++) {
    ceres::CostFunction *cost_function =
        new ceres::NumericDiffCostFunction<pnec::residual::PNECSymmetrical,
                                           ceres::CENTRAL, 1, 1, 1, 4>(
            new pnec::residual::PNECSymmetrical(bvs_1[i], bvs_2[i], covs_1[i],
                                                covs_2[i], regularization));
    problem.AddResidualBlock(cost_function, nullptr, &theta_, &phi_,
                             orientation_.coeffs().data());
  }

  // if (noise_frame == pnec::common::Host) {
  //   for (size_t i = 0; i < bvs_1.size(); i++) {

  //     ceres::CostFunction *cost_function =
  //         new
  //         ceres::NumericDiffCostFunction<pnec::residual::PNECResidualHost,
  //                                            ceres::CENTRAL, 1, 1, 1, 4>(
  //             new pnec::residual::PNECResidualHost(bvs_1[i], bvs_2[i],
  //             covs[i],
  //                                                  regularization));
  //     problem.AddResidualBlock(cost_function, nullptr, &theta_, &phi_,
  //                              orientation_.coeffs().data());
  //   }
  // } else {
  //   for (size_t i = 0; i < bvs_1.size(); i++) {

  //     ceres::CostFunction *cost_function =
  //         new
  //         ceres::NumericDiffCostFunction<pnec::residual::PNECResidualTarget,
  //                                            ceres::CENTRAL, 1, 1, 1, 4>(
  //             new pnec::residual::PNECResidualTarget(bvs_1[i], bvs_2[i],
  //                                                    covs[i],
  //                                                    regularization));
  //     problem.AddResidualBlock(cost_function, nullptr, &theta_, &phi_,
  //                              orientation_.coeffs().data());
  //   }
  // }
  ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
  // std::cout << "here" << std::endl;

  problem.SetManifold(orientation_.coeffs().data(), quaternion_manifold);
  // problem.SetParameterization(orientation_.coeffs().data(),
  //                             new ceres::EigenQuaternionParameterization);

  ceres::Solve(options_, &problem, &summary_);
}

void PNECCeres::InitValues(const Eigen::Quaterniond orientation, double theta,
                           double phi) {
  orientation_ = orientation;
  theta_ = theta;
  phi_ = phi;
}

void PNECCeres::InitValues(const Sophus::SE3d &init) {
  orientation_ = init.unit_quaternion();
  pnec::common::AnglesFromVec(init.translation(), theta_, phi_);
}

void PNECCeres::InitValues(const Eigen::Quaterniond &orientation,
                           const Eigen::Vector3d &translation) {
  orientation_ = orientation;
  pnec::common::AnglesFromVec(translation, theta_, phi_);
}

void PNECCeres::SetOptions(const ceres::Solver::Options &options) {
  options_ = options;
}

Eigen::Matrix3d PNECCeres::Orientation() const {
  return orientation_.normalized().toRotationMatrix();
}

Eigen::Vector3d PNECCeres::Translation() const {
  return Eigen::Vector3d(std::sin(theta_) * std::cos(phi_),
                         std::sin(theta_) * std::sin(phi_), std::cos(theta_));
}

Sophus::SE3d PNECCeres::Result() const {
  return Sophus::SE3d(orientation_.normalized().toRotationMatrix(),
                      Eigen::Vector3d(std::sin(theta_) * std::cos(phi_),
                                      std::sin(theta_) * std::sin(phi_),
                                      std::cos(theta_)));
}

} // namespace optimization
} // namespace pnec
