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

#include "frame2frame.h"

#include "math.h"
#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>
#include <opengv/types.hpp>

#include "common.h"
#include "essential_matrix_methods.h"
#include "odometry_output.h"
#include "pnec.h"

namespace pnec {
namespace rel_pose_estimation {

Frame2Frame::Frame2Frame(const pnec::rel_pose_estimation::Options &options)
    : options_{options} {}
Frame2Frame::~Frame2Frame() {}

Sophus::SE3d Frame2Frame::Align(pnec::frames::BaseFrame::Ptr frame1,
                                pnec::frames::BaseFrame::Ptr frame2,
                                pnec::FeatureMatches &matches,
                                Sophus::SE3d prev_rel_pose,
                                std::vector<int> &inliers,
                                pnec::common::FrameTiming &frame_timing,
                                bool ablation, std::string ablation_folder) {
  if (ablation) {
    if (!boost::filesystem::exists(ablation_folder)) {
      boost::filesystem::create_directory(ablation_folder);
    }
  }

  curr_timestamp_ = frame2->Timestamp();

  opengv::bearingVectors_t bvs1;
  opengv::bearingVectors_t bvs2;
  std::vector<Eigen::Matrix3d> proj_covs;
  GetFeatures(frame1, frame2, matches, bvs1, bvs2, proj_covs);

  if (ablation) {
    AblationAlign(bvs1, bvs2, proj_covs, ablation_folder);
  }

  return PNECAlign(bvs1, bvs2, proj_covs, prev_rel_pose, inliers, frame_timing,
                   ablation_folder);
}

Sophus::SE3d Frame2Frame::AlignFurther(pnec::frames::BaseFrame::Ptr frame1,
                                       pnec::frames::BaseFrame::Ptr frame2,
                                       pnec::FeatureMatches &matches,
                                       Sophus::SE3d prev_rel_pose,
                                       std::vector<int> &inliers,
                                       bool &success) {
  opengv::bearingVectors_t bvs1;
  opengv::bearingVectors_t bvs2;
  std::vector<Eigen::Matrix3d> proj_covs;
  GetFeatures(frame1, frame2, matches, bvs1, bvs2, proj_covs);

  pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  Sophus::SE3d rel_pose =
      PNECAlign(bvs1, bvs2, proj_covs, prev_rel_pose, inliers, dummy_timing);

  if (ransac_iterations_ >= options_.max_ransac_iterations_) {
    success = false;
  } else if (inliers.size() < options_.min_inliers_ && inliers.size() != 0) {
    success = false;
  } else {
    success = true;
  }
  return rel_pose;
}

const std::map<std::string, std::vector<std::pair<double, Sophus::SE3d>>> &
Frame2Frame::GetAblationResults() const {
  return ablation_rel_poses_;
}

Sophus::SE3d Frame2Frame::PNECAlign(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covs,
    Sophus::SE3d prev_rel_pose, std::vector<int> &inliers,
    pnec::common::FrameTiming &frame_timing, std::string ablation_folder) {
  pnec::rel_pose_estimation::PNEC pnec(options_);
  Sophus::SE3d rel_pose = pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose,
                                     inliers, frame_timing);

  if (ablation_rel_poses_.count("PNEC") == 0) {
    ablation_rel_poses_["PNEC"] = {
        std::make_pair<double, Sophus::SE3d>(0.0, Sophus::SE3d())};
  }
  ablation_rel_poses_["PNEC"].push_back(
      std::make_pair(curr_timestamp_, rel_pose));
  return rel_pose;
}

void Frame2Frame::AblationAlign(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covs,
    std::string ablation_folder) {

  auto GetPrevRelPose = [this](std::string method_name) -> Sophus::SE3d {
    Sophus::SE3d prev_rel_pose;
    if (ablation_rel_poses_.count(method_name) == 0) {
      ablation_rel_poses_[method_name] = {
          std::make_pair<double, Sophus::SE3d>(0.0, Sophus::SE3d())};
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = ablation_rel_poses_[method_name].back().second;
    }
    return prev_rel_pose;
  };

  {
    std::string name = "NEC";

    Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

    pnec::rel_pose_estimation::Options nec_options = options_;
    nec_options.use_nec_ = true;
    nec_options.use_ceres_ = false;

    pnec::rel_pose_estimation::PNEC pnec(nec_options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    ablation_rel_poses_[name].push_back(
        std::make_pair(curr_timestamp_, rel_pose));
  }

  {
    std::string name = "NEC-LS";

    Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

    pnec::rel_pose_estimation::Options nec_options = options_;
    nec_options.use_nec_ = true;

    pnec::rel_pose_estimation::PNEC pnec(nec_options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    ablation_rel_poses_[name].push_back(
        std::make_pair(curr_timestamp_, rel_pose));
  }

  // {
  //   std::string name = "NEC+PNEC-LS";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.weighted_iterations_ = 1;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "PNECwoLS";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.use_ceres_ = false;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "PNEConlyLS";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.weighted_iterations_ = 0;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "PNEConlyLSfull";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   pnec::rel_pose_estimation::Options options = options_;
  //   options.weighted_iterations_ = 0;
  //   options.ceres_options_.max_num_iterations = 10000;

  //   pnec::rel_pose_estimation::PNEC pnec(options);
  //   pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
  //   Sophus::SE3d rel_pose =
  //       pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::vector<size_t> iterations = {5, 15};

  //   for (const auto &max_it : iterations) {
  //     {
  //       std::string name = "PNEC-" + std::to_string(max_it);

  //       Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //       pnec::rel_pose_estimation::Options options = options_;
  //       options.weighted_iterations_ = max_it;

  //       pnec::rel_pose_estimation::PNEC pnec(options);
  //       pnec::common::FrameTiming dummy_timing =
  //       pnec::common::FrameTiming(0); Sophus::SE3d rel_pose =
  //           pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose,
  //           dummy_timing);

  //       ablation_rel_poses_[name].push_back(
  //           std::make_pair(curr_timestamp_, rel_pose));
  //       pnec::out::SavePose(ablation_folder, name, curr_timestamp_,
  //       rel_pose);
  //     }
  //   }
  // }

  // {
  //   std::string name = "8pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           EIGHTPT);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "Nister5pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           NISTER);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "Stewenius5pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           STEWENIUS);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }

  // {
  //   std::string name = "7pt";

  //   Sophus::SE3d prev_rel_pose = GetPrevRelPose(name);

  //   Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
  //       bvs1, bvs2, prev_rel_pose,
  //       opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
  //           SEVENPT);

  //   ablation_rel_poses_[name].push_back(
  //       std::make_pair(curr_timestamp_, rel_pose));
  //   pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  // }
}

void Frame2Frame::GetFeatures(pnec::frames::BaseFrame::Ptr host_frame,
                              pnec::frames::BaseFrame::Ptr target_frame,
                              pnec::FeatureMatches &matches,
                              opengv::bearingVectors_t &host_bvs,
                              opengv::bearingVectors_t &target_bvs,
                              std::vector<Eigen::Matrix3d> &proj_covs) {
  std::vector<size_t> host_matches;
  std::vector<size_t> target_matches;
  for (const auto &match : matches) {
    host_matches.push_back(match.queryIdx);
    target_matches.push_back(match.trainIdx);
  }
  pnec::features::KeyPoints host_keypoints =
      host_frame->keypoints(host_matches);
  pnec::features::KeyPoints target_keypoints =
      target_frame->keypoints(target_matches);

  std::vector<Eigen::Matrix3d> host_covs;
  std::vector<Eigen::Matrix3d> target_covs;
  for (auto const &[id, keypoint] : host_keypoints) {
    host_bvs.push_back(keypoint.bearing_vector_);
    host_covs.push_back(keypoint.bv_covariance_);
  }
  for (auto const &[id, keypoint] : target_keypoints) {
    target_bvs.push_back(keypoint.bearing_vector_);
    target_covs.push_back(keypoint.bv_covariance_);
  }

  if (options_.noise_frame_ == pnec::common::Host) {
    proj_covs = host_covs;
  } else {
    proj_covs = target_covs;
  }
}

} // namespace rel_pose_estimation
} // namespace pnec