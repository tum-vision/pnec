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

#ifndef OPTIMIZATION_PNEC_CERES_H_
#define OPTIMIZATION_PNEC_CERES_H_

#include "sophus/se3.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "common.h"

namespace pnec {
namespace optimization {
class PNECCeres {
public:
  PNECCeres();
  PNECCeres(const Sophus::SE3d &init,
            const ceres::Solver::Options &options = ceres::Solver::Options());
  PNECCeres(const Eigen::Quaterniond &orientation, double theta, double phi,
            const ceres::Solver::Options &options = ceres::Solver::Options());
  PNECCeres(const Eigen::Quaterniond &orientation,
            const Eigen::Vector3d &translation,
            const ceres::Solver::Options &options = ceres::Solver::Options());
  ~PNECCeres();

  void Optimize(const std::vector<Eigen::Vector3d> &bvs_1,
                const std::vector<Eigen::Vector3d> &bvs_2,
                const std::vector<Eigen::Matrix3d> &covs, double regularization,
                pnec::common::NoiseFrame noise_frame = pnec::common::Target);

  void Optimize(const std::vector<Eigen::Vector3d> &bvs_1,
                const std::vector<Eigen::Vector3d> &bvs_2,
                const std::vector<Eigen::Matrix3d> &covs_1,
                const std::vector<Eigen::Matrix3d> &covs_2,
                double regularization);

  void InitValues(const Eigen::Quaterniond orientation, double theta,
                  double phi);
  void InitValues(const Sophus::SE3d &init);
  void InitValues(const Eigen::Quaterniond &orientation,
                  const Eigen::Vector3d &translation);

  void SetOptions(const ceres::Solver::Options &options);

  Eigen::Matrix3d Orientation() const;
  Eigen::Vector3d Translation() const;
  Sophus::SE3d Result() const;

private:
  Eigen::Quaterniond orientation_;
  double theta_;
  double phi_;
  ceres::Solver::Options options_;
  ceres::Solver::Summary summary_;
};

} // namespace optimization
} // namespace pnec

#endif // OPTIMIZATION_PNEC_CERES_H_