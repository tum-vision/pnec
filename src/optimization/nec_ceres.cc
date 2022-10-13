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

#include "nec_ceres.h"

#include <boost/log/trivial.hpp>

#include "common.h"
#include "nec_residual.h"

namespace pnec {
namespace optimization {

NECCeres::NECCeres() {
  orientation_ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  theta_ = 0.0;
  phi_ = 0.0;
  options_ = ceres::Solver::Options();
}

NECCeres::NECCeres(const Sophus::SE3d &init,
                   const ceres::Solver::Options &options)
    : options_{options} {
  orientation_ = init.unit_quaternion();
  pnec::common::AnglesFromVec(init.translation(), theta_, phi_);
}

NECCeres::NECCeres(const Eigen::Quaterniond &orientation, double theta,
                   double phi, const ceres::Solver::Options &options)
    : orientation_{orientation}, theta_{theta}, phi_{phi} {}

NECCeres::NECCeres(const Eigen::Quaterniond &orientation,
                   const Eigen::Vector3d &translation,
                   const ceres::Solver::Options &options)
    : orientation_{orientation}, options_{options} {
  pnec::common::AnglesFromVec(translation, theta_, phi_);
}

NECCeres::~NECCeres() {}

void NECCeres::Optimize(const std::vector<Eigen::Vector3d> &bvs_1,
                        const std::vector<Eigen::Vector3d> &bvs_2) {

  ceres::Problem problem;
  std::vector<double *> orientation_d =
      std::vector<double *>{orientation_.coeffs().data()};

  for (size_t i = 0; i < bvs_1.size(); i++) {

    ceres::CostFunction *cost_function =
        new ceres::NumericDiffCostFunction<pnec::residual::NECResidual,
                                           ceres::CENTRAL, 1, 1, 1, 4>(
            new pnec::residual::NECResidual(bvs_1[i], bvs_2[i]));
    problem.AddResidualBlock(cost_function, nullptr, &theta_, &phi_,
                             orientation_.coeffs().data());
  }

  ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
  // std::cout << "here" << std::endl;

  problem.SetManifold(orientation_.coeffs().data(), quaternion_manifold);
  // problem.SetParameterization(orientation_.coeffs().data(),
  //                             new ceres::EigenQuaternionParameterization);

  ceres::Solve(options_, &problem, &summary_);

  BOOST_LOG_TRIVIAL(trace) << summary_.BriefReport() << "\n";
  BOOST_LOG_TRIVIAL(trace) << summary_.FullReport() << "\n";
}

void NECCeres::InitValues(const Eigen::Quaterniond orientation, double theta,
                          double phi) {
  orientation_ = orientation;
  theta_ = theta;
  phi_ = phi;
}

void NECCeres::InitValues(const Sophus::SE3d &init) {
  orientation_ = init.unit_quaternion();
  pnec::common::AnglesFromVec(init.translation(), theta_, phi_);
}

void NECCeres::InitValues(const Eigen::Quaterniond &orientation,
                          const Eigen::Vector3d &translation) {
  orientation_ = orientation;
  pnec::common::AnglesFromVec(translation, theta_, phi_);
}

void NECCeres::SetOptions(const ceres::Solver::Options &options) {
  options_ = options;
}

Eigen::Matrix3d NECCeres::Orientation() const {
  return orientation_.normalized().toRotationMatrix();
}

Eigen::Vector3d NECCeres::Translation() const {
  return Eigen::Vector3d(std::sin(theta_) * std::cos(phi_),
                         std::sin(theta_) * std::sin(phi_), std::cos(theta_));
}

Sophus::SE3d NECCeres::Result() const {
  return Sophus::SE3d(orientation_.normalized().toRotationMatrix(),
                      Eigen::Vector3d(std::sin(theta_) * std::cos(phi_),
                                      std::sin(theta_) * std::sin(phi_),
                                      std::cos(theta_)));
}

} // namespace optimization
} // namespace pnec
