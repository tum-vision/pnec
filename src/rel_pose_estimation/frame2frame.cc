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

// TO Odometry
// void InlierExtraction(
//     const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t
//     &bvs2, const std::vector<Eigen::Vector3d> &fvs1, const
//     std::vector<Eigen::Vector3d> &fvs2, std::vector<Eigen::Vector2d> &pvs1,
//     std::vector<Eigen::Vector2d> &pvs2, const std::vector<Eigen::Matrix3d>
//     &covs, opengv::bearingVectors_t &in_bvs1, opengv::bearingVectors_t
//     &in_bvs2, std::vector<Eigen::Matrix3d> &in_proj_covs, const
//     std::vector<int> &inliers, bool host_frame = false) {
//   bool unscented = true;
//   in_bvs1.clear();
//   in_bvs1.reserve(inliers.size());
//   in_bvs2.clear();
//   in_bvs2.reserve(inliers.size());
//   in_proj_covs.clear();
//   in_proj_covs.reserve(inliers.size());
//   Eigen::Matrix3d K;
//   cv::cv2eigen(pnec::Camera::instance().cameraParameters().intrinsic(), K);
//   Eigen::Matrix3d K_inv = K.inverse();
//   for (const auto inlier : inliers) {
//     // for (int inlier = 0; inlier < n_matches; inlier++) {
//     in_bvs1.push_back(bvs1[inlier]);
//     in_bvs2.push_back(bvs2[inlier]);
//     if (unscented) {
//       if (host_frame) {
//         Eigen::Vector3d mu(pvs1[inlier](0), pvs1[inlier](1), 1.0);
//         in_proj_covs.push_back(
//             mro::common::UnscentedTransform(mu, covs[inlier], 1.0, K_inv));
//       } else {
//         Eigen::Vector3d mu(pvs2[inlier](0), pvs2[inlier](1), 1.0);
//         in_proj_covs.push_back(
//             mro::common::UnscentedTransform(mu, covs[inlier], 1.0, K_inv));
//       }
//     } else {
//       if (host_frame) {
//         Eigen::Matrix3d jacobian =
//             mro::common::ProjectionJacobian(fvs1[inlier]);
//         in_proj_covs.push_back(jacobian * covs[inlier] *
//         jacobian.transpose());
//       } else {
//         Eigen::Matrix3d jacobian =
//             mro::common::ProjectionJacobian(fvs2[inlier]);
//         in_proj_covs.push_back(jacobian * covs[inlier] *
//         jacobian.transpose());
//       }
//     }
//   }
// }

// void GetFeatures(
//     pnec::frames::BaseFrame::Ptr frame1, pnec::frames::BaseFrame::Ptr frame2,
//     const pnec::FeatureMatches &matches, std::vector<cv::KeyPoint> &img_p1,
//     std::vector<cv::KeyPoint> &img_p2, opengv::bearingVectors_t &bvs1,
//     opengv::bearingVectors_t &bvs2, std::vector<Eigen::Vector3d> &fvs1,
//     std::vector<Eigen::Vector3d> &fvs2, std::vector<Eigen::Vector2d> &pvs1,
//     std::vector<Eigen::Vector2d> &pvs2, std::vector<Eigen::Matrix3d> &covs,
//     bool weighted, bool host_frame = false) {
//   const auto &kps1 = frame1->undistortedKeypoints();
//   const auto &kps2 = frame2->undistortedKeypoints();
//   std::vector<Eigen::Matrix2d> covariances;
//   if (host_frame) {
//     covariances = frame1->covariances();
//   } else {
//     covariances = frame2->covariances();
//   }

//   const size_t n_matches = (size_t)matches.size();
//   img_p1.clear();
//   img_p1.reserve(n_matches);
//   img_p2.clear();
//   img_p2.reserve(n_matches);
//   bvs1.clear();
//   bvs1.reserve(n_matches);
//   bvs2.clear();
//   bvs2.reserve(n_matches);
//   fvs1.clear();
//   fvs1.reserve(n_matches);
//   fvs2.clear();
//   fvs2.reserve(n_matches);
//   pvs1.clear();
//   pvs1.reserve(n_matches);
//   pvs2.clear();
//   pvs2.reserve(n_matches);
//   covs.clear();
//   covs.reserve(n_matches);

//   Eigen::Matrix3d K;
//   cv::cv2eigen(pnec::Camera::instance().cameraParameters().intrinsic(), K);
//   Eigen::Matrix3d K_inv = K.inverse();

//   for (const auto &match : matches) {
//     img_p1.push_back(kps1[match.queryIdx]);
//     Eigen::Vector3d point_1(kps1[match.queryIdx].pt.x * 1.0,
//                             kps1[match.queryIdx].pt.y * 1.0, 1.0);
//     fvs1.push_back(K_inv * point_1);
//     bvs1.push_back((K_inv * point_1).normalized());
//     pvs1.push_back(
//         Eigen::Vector2d(kps1[match.queryIdx].pt.x,
//         kps1[match.queryIdx].pt.y));
//     Eigen::Vector3d point_2(kps2[match.trainIdx].pt.x * 1.0,
//                             kps2[match.trainIdx].pt.y * 1.0, 1.0);
//     img_p1.push_back(kps1[match.trainIdx]);
//     fvs2.push_back(K_inv * point_2);
//     bvs2.push_back((K_inv * point_2).normalized());
//     pvs2.push_back(
//         Eigen::Vector2d(kps1[match.trainIdx].pt.x,
//         kps1[match.trainIdx].pt.y));
//     if (weighted) {
//       if (host_frame) {
//         Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
//         covariance.topLeftCorner(2, 2) = covariances[match.queryIdx];
//         covs.push_back(covariance);
//       } else {
//         Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
//         covariance.topLeftCorner(2, 2) = covariances[match.trainIdx];
//         covs.push_back(covariance);
//       }
//     } else {
//       covs.push_back(Eigen::Matrix3d::Zero());
//     }
//   }
// }

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
    // std::cout << "Could align Frames: RANSAC took too long!" << std::endl;
    success = false;
  } else if (inliers.size() < options_.min_inliers_ && inliers.size() != 0) {
    // std::cout << "Could align Frames: Didn't find enough inliers!" <<
    // std::endl;
    success = false;
  } else {
    success = true;
  }
  return rel_pose;
}

Sophus::SE3d Frame2Frame::PNECAlign(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covs,
    Sophus::SE3d prev_rel_pose, std::vector<int> &inliers,
    pnec::common::FrameTiming &frame_timing, std::string ablation_folder) {
  pnec::rel_pose_estimation::PNEC pnec(options_);
  Sophus::SE3d rel_pose =
      pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, frame_timing);

  if (ablation_folder != "") {
    pnec::out::SavePose(ablation_folder, "PNEC", curr_timestamp_, rel_pose);
  }

  // To rel_pose_estimation
  // std::vector<Eigen::Vector3d> bvs_1;
  // std::vector<Eigen::Vector3d> bvs_2;
  // for (const auto &bv : in_bvs1) {
  //   bvs_1.push_back(bv);
  // }
  // for (const auto &bv : in_bvs2) {
  //   bvs_2.push_back(bv);
  // }
  // double nec_cost = mro::align::CostFunction(bvs_1, bvs_2, covs, ES_pose);
  // double pnec_cost = mro::align::CostFunction(bvs_1, bvs_2, covs,
  // prev_PNEC_); SaveCost("cost", nec_cost, pnec_cost, curr_timestamp_,
  // weighted_,
  //          ablation_folder);
  return rel_pose;
}

void Frame2Frame::AblationAlign(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covs,
    std::string ablation_folder) {
  {
    std::string name = "NEC";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    pnec::rel_pose_estimation::Options nec_options = options_;
    nec_options.use_nec_ = true;
    nec_options.use_ceres_ = false;

    pnec::rel_pose_estimation::PNEC pnec(nec_options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }

  {
    std::string name = "NEC-LS";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    pnec::rel_pose_estimation::Options nec_options = options_;
    nec_options.use_nec_ = true;

    pnec::rel_pose_estimation::PNEC pnec(nec_options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }

  {
    std::string name = "NEC+PNEC-LS";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    pnec::rel_pose_estimation::Options options = options_;
    options.weighted_iterations_ = 1;

    pnec::rel_pose_estimation::PNEC pnec(options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }

  {
    std::string name = "PNECwoLS";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    pnec::rel_pose_estimation::Options options = options_;
    options.use_ceres_ = false;

    pnec::rel_pose_estimation::PNEC pnec(options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }

  {
    std::string name = "PNEConlyLS";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    pnec::rel_pose_estimation::Options options = options_;
    options.weighted_iterations_ = 0;

    pnec::rel_pose_estimation::PNEC pnec(options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }

  {
    std::string name = "PNEConlyLSfull";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    pnec::rel_pose_estimation::Options options = options_;
    options.weighted_iterations_ = 0;
    options.ceres_options_.max_num_iterations = 10000;

    pnec::rel_pose_estimation::PNEC pnec(options);
    pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
    Sophus::SE3d rel_pose =
        pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }

  {
    std::vector<size_t> iterations = {5, 15};

    for (const auto &max_it : iterations) {
      {
        std::string name = "PNEC-" + std::to_string(max_it);

        Sophus::SE3d prev_rel_pose;
        if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
          prev_rel_poses_[name] = Sophus::SE3d();
          prev_rel_pose = Sophus::SE3d();
        } else {
          prev_rel_pose = prev_rel_poses_[name];
        }

        pnec::rel_pose_estimation::Options options = options_;
        options.weighted_iterations_ = max_it;

        pnec::rel_pose_estimation::PNEC pnec(options);
        pnec::common::FrameTiming dummy_timing = pnec::common::FrameTiming(0);
        Sophus::SE3d rel_pose =
            pnec.Solve(bvs1, bvs2, projected_covs, prev_rel_pose, dummy_timing);

        prev_rel_poses_[name] = rel_pose;
        pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
      }
    }
  }

  {
    std::string name = "8pt";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
        bvs1, bvs2, prev_rel_pose,
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
            EIGHTPT);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }

  {
    std::string name = "Nister5pt";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
        bvs1, bvs2, prev_rel_pose,
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
            NISTER);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }

  {
    std::string name = "Stewenius5pt";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
        bvs1, bvs2, prev_rel_pose,
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
            STEWENIUS);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }

  {
    std::string name = "7pt";

    Sophus::SE3d prev_rel_pose;
    if (prev_rel_poses_.find(name) != prev_rel_poses_.end()) {
      prev_rel_poses_[name] = Sophus::SE3d();
      prev_rel_pose = Sophus::SE3d();
    } else {
      prev_rel_pose = prev_rel_poses_[name];
    }

    Sophus::SE3d rel_pose = pnec::rel_pose_estimation::EMPoseEstimation(
        bvs1, bvs2, prev_rel_pose,
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
            SEVENPT);

    prev_rel_poses_[name] = rel_pose;
    pnec::out::SavePose(ablation_folder, name, curr_timestamp_, rel_pose);
  }
}

void Frame2Frame::GetFeatures(pnec::frames::BaseFrame::Ptr frame1,
                              pnec::frames::BaseFrame::Ptr frame2,
                              pnec::FeatureMatches &matches,
                              opengv::bearingVectors_t &bvs1,
                              opengv::bearingVectors_t &bvs2,
                              std::vector<Eigen::Matrix3d> &proj_covs) {
  const std::vector<Eigen::Vector3d> pts1 = frame1->ProjectedPoints();
  const std::vector<Eigen::Vector3d> pts2 = frame2->ProjectedPoints();
  std::vector<Eigen::Matrix3d> covs;
  if (options_.noise_frame_ == pnec::common::Host) {
    covs = frame1->ProjectedCovariances();
  } else {
    covs = frame2->ProjectedCovariances();
  }

  for (const auto &match : matches) {
    bvs1.push_back(pts1[match.queryIdx]);
    bvs2.push_back(pts2[match.trainIdx]);
    if (options_.noise_frame_ == pnec::common::Host) {
      proj_covs.push_back(covs[match.queryIdx]);
    } else {
      proj_covs.push_back(covs[match.trainIdx]);
    }
  }
}

} // namespace rel_pose_estimation
} // namespace pnec