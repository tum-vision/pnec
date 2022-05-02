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

#ifndef ALIGN_FRAME2FRAME_H_
#define ALIGN_FRAME2FRAME_H_

#include "opengv/types.hpp"
#include "sophus/se3.hpp"

#include "base_frame.h"
#include "pnec_config.h"
#include "timing.h"

namespace pnec {
namespace rel_pose_estimation {

class Frame2Frame {
public:
  using Ptr = std::shared_ptr<Frame2Frame>;
  Frame2Frame(const pnec::rel_pose_estimation::Options &options);
  ~Frame2Frame();

  Sophus::SE3d Align(pnec::frames::BaseFrame::Ptr frame1,
                     pnec::frames::BaseFrame::Ptr frame2,
                     pnec::FeatureMatches &matches, Sophus::SE3d prev_rel_pose,
                     std::vector<int> &inliers,
                     pnec::common::FrameTiming &frame_timing,
                     bool ablation = false, std::string ablation_folder = "");

  Sophus::SE3d AlignFurther(pnec::frames::BaseFrame::Ptr frame1,
                            pnec::frames::BaseFrame::Ptr frame2,
                            pnec::FeatureMatches &matches,
                            Sophus::SE3d prev_rel_pose,
                            std::vector<int> &inliers, bool &success);

private:
  Sophus::SE3d PNECAlign(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         Sophus::SE3d prev_rel_pose, std::vector<int> &inliers,
                         pnec::common::FrameTiming &frame_timing,
                         std::string ablation_folder = "");

  void AblationAlign(const opengv::bearingVectors_t &bvs1,
                     const opengv::bearingVectors_t &bvs2,
                     const std::vector<Eigen::Matrix3d> &projected_covs,
                     std::string ablation_folder);

  void GetFeatures(pnec::frames::BaseFrame::Ptr frame1,
                   pnec::frames::BaseFrame::Ptr frame2,
                   pnec::FeatureMatches &matches,
                   opengv::bearingVectors_t &bvs1,
                   opengv::bearingVectors_t &bvs2,
                   std::vector<Eigen::Matrix3d> &proj_covs);

protected:
  // Paramters
  pnec::rel_pose_estimation::Options options_;

  double curr_timestamp_;

  std::map<std::string, Sophus::SE3d> prev_rel_poses_;

  // Output values
  int ransac_iterations_;
};
} // namespace rel_pose_estimation
} // namespace pnec

#endif // ALIGN_FRAME2FRAME_H_