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

#ifndef REL_POSE_ESTIMATION_PNEC_H_
#define REL_POSE_ESTIMATION_PNEC_H_

#include "opengv/types.hpp"
#include "sophus/se3.hpp"

#include "pnec_config.h"
#include "timing.h"

namespace pnec {
namespace rel_pose_estimation {

class PNEC {
public:
  PNEC(const pnec::rel_pose_estimation::Options &options);
  ~PNEC();

  Sophus::SE3d Solve(const opengv::bearingVectors_t &bvs1,
                     const opengv::bearingVectors_t &bvs2,
                     const std::vector<Eigen::Matrix3d> &projected_covs,
                     const Sophus::SE3d &initial_pose);

  Sophus::SE3d Solve(const opengv::bearingVectors_t &bvs1,
                     const opengv::bearingVectors_t &bvs2,
                     const std::vector<Eigen::Matrix3d> &projected_covs,
                     const Sophus::SE3d &initial_pose,
                     std::vector<int> &inliers);

  Sophus::SE3d Solve(const opengv::bearingVectors_t &bvs1,
                     const opengv::bearingVectors_t &bvs2,
                     const std::vector<Eigen::Matrix3d> &projected_covs,
                     const Sophus::SE3d &initial_pose,
                     pnec::common::FrameTiming &timing);

  Sophus::SE3d Solve(const opengv::bearingVectors_t &bvs1,
                     const opengv::bearingVectors_t &bvs2,
                     const std::vector<Eigen::Matrix3d> &projected_covs,
                     const Sophus::SE3d &initial_pose,
                     std::vector<int> &inliers,
                     pnec::common::FrameTiming &timing);

  Sophus::SE3d Eigensolver(const opengv::bearingVectors_t &bvs1,
                           const opengv::bearingVectors_t &bvs2,
                           const Sophus::SE3d &initial_pose,
                           std::vector<int> &inliers);

  Sophus::SE3d
  WeightedEigensolver(const opengv::bearingVectors_t &bvs1,
                      const opengv::bearingVectors_t &bvs2,
                      const std::vector<Eigen::Matrix3d> &projected_covariances,
                      const Sophus::SE3d &initial_pose);

  Sophus::SE3d
  CeresSolver(const opengv::bearingVectors_t &bvs1,
              const opengv::bearingVectors_t &bvs2,
              const std::vector<Eigen::Matrix3d> &projected_covariances,
              const Sophus::SE3d &initial_pose);
  Sophus::SE3d
  CeresSolverFull(const opengv::bearingVectors_t &bvs1,
                  const opengv::bearingVectors_t &bvs2,
                  const std::vector<Eigen::Matrix3d> &projected_covariances,
                  double regularization, const Sophus::SE3d &initial_pose);

  Sophus::SE3d NECCeresSolver(const opengv::bearingVectors_t &bvs1,
                              const opengv::bearingVectors_t &bvs2,
                              const Sophus::SE3d &initial_pose);

private:
  void InlierExtraction(const opengv::bearingVectors_t &bvs1,
                        const opengv::bearingVectors_t &bvs2,
                        const std::vector<Eigen::Matrix3d> &covs,
                        opengv::bearingVectors_t &in_bvs1,
                        opengv::bearingVectors_t &in_bvs2,
                        std::vector<Eigen::Matrix3d> &in_proj_covs,
                        const std::vector<int> &inliers);

protected:
  pnec::rel_pose_estimation::Options options_;
};
} // namespace rel_pose_estimation
} // namespace pnec

#endif // REL_POSE_ESTIMATION_PNEC_H_