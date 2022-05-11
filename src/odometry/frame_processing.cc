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

#include "frame_processing.h"

#include <cmath>
#include <ctime>
#include <opencv2/core/eigen.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>

#include "base_matcher.h"
#include "common.h"

#define PLOT true

namespace pnec {
namespace odometry {

bool FrameProcessing::ProcessFrame(pnec::frames::BaseFrame::Ptr frame,
                                   pnec::common::FrameTiming &frame_timing,
                                   std::string results_folder, int win_size) {
  const int skip = 0;

  // Create View
  pnec::odometry::View::Ptr curr_view =
      std::make_shared<pnec::odometry::View>(frame);

  const int m = 1 + view_graph_->GraphSize();

  // no pose estimation for the first frame
  if (m <= skip + 1) {
    view_graph_->AddView(curr_view);
    return true;
  }

  pnec::frames::BaseFrame::Ptr curr_frame = curr_view->Frame();

  const int curr_view_idx = m - 1;
  int prev_view_idx = curr_view_idx - skip - 1;

  int count_connections = 0;

  pnec::odometry::View::Ptr prev_view;
  view_graph_->GetViewByPos(prev_view_idx, prev_view);
  pnec::frames::BaseFrame::Ptr prev_frame = prev_view->Frame();

  bool skipping_frame;
  pnec::FeatureMatches matches =
      matcher_->FindMatches(prev_frame, curr_frame, skipping_frame);

  if (skipping_frame && !no_skip_) {
    return false;
  }

  std::vector<int> inliers;

  Sophus::SO3d prev_rel_rotation = Sophus::SO3d();
  if (curr_frame->id() > 1) {
    prev_rel_rotation = PrevRelRotation(prev_view, prev_view_idx);
  }

  Sophus::SE3d prev_pose(prev_rel_rotation, Eigen::Vector3d(0.0, 0.0, 0.0));

  Sophus::SE3d rel_pose = f2f_pose_estimation_->Align(
      prev_frame, curr_frame, matches, prev_pose, inliers, frame_timing, true,
      results_folder + "ablation/");
  int n_epi_inlr = inliers.size();

  //   matches_vec.push_back(matches.size());
  //   inliers_vec.push_back(n_epi_inlr);
  view_graph_->AddView(curr_view);
  bool success = view_graph_->Connect(prev_view, curr_view, matches, rel_pose);
  if (!success) {
    return false;
  }
  view_graph_->ShortenViewGraph();

  // Visualization
  // TODO: Make Visualization
  if (skipping_counter_ >= skip_showing_n_) {
    switch (visualization_level_) {
    case Features:
      break;
    case InitMatches:
    case AllMatches:
      break;
    case InitMatchesRANSAC:
    case AllMatchesRANSAC:
      break;
    case NoViz:
      break;
    }
    skipping_counter_ = 0;
  } else {
    skipping_counter_++;
  }

  count_connections++;
  prev_view_idx--;

  return true;
}

bool FrameProcessing::ProcessUncertaintyExtraction(pnec::frames::BaseFrame::Ptr host_frame, pnec::frames::BaseFrame::Ptr target_frame, Sophus::SE3d init_pose,
                                   std::string results_folder, bool extract_host) {
  const int skip = 0;

  const int m = 1 + view_graph_->GraphSize();

  bool skipping_frame;
  pnec::FeatureMatches matches =
      matcher_->FindMatches(host_frame, target_frame, skipping_frame);
  std::cout << "Found " << matches.size() << " matches" << std::endl;

  std::vector<int> inliers;

  std::cout << "start aligning" << std::endl;
  pnec::common::FrameTiming dummy_timing(0);
  Sophus::SE3d rel_pose = f2f_pose_estimation_->Align(
      host_frame, target_frame, matches, init_pose, inliers, dummy_timing, false,
      results_folder + "ablation/");
  std::cout << "Finished aligning" << std::endl;
  std::cout << "Found " << inliers.size() << " inliers." << std::endl;

  // Pass the inliers to the prev_frame, to extract patches and covariances
  std::vector<int> inlier_kp_idx;
  for (const auto& inlier: inliers) {
    cv::DMatch match = matches[inlier];
    if (extract_host) {
      inlier_kp_idx.push_back(match.queryIdx);
    } else {
      inlier_kp_idx.push_back(match.trainIdx);
    }
  }
  std::cout << "Saving " << inlier_kp_idx.size() << " inlier patches" << std::endl;
  if (extract_host) {
    host_frame->SaveInlierPatches(inlier_kp_idx, extraction_counter_, results_folder);
  } else {
    target_frame->SaveInlierPatches(inlier_kp_idx, extraction_counter_, results_folder);
  }
  std::cout << "Saved Patches." << std::endl;

  return true;
}

Sophus::SO3d
FrameProcessing::PrevRelRotation(pnec::odometry::View::Ptr prev_view,
                                 int prev_view_idx) {
  for (const auto &connection : prev_view->connections()) {
    pnec::odometry::View::Ptr view = connection.first;
    pnec::odometry::View::Connection *view_connection = connection.second;

    int i = view->Frame()->id();

    if (i == prev_view_idx - 1) {
      return view_connection->Pose().so3();
    }
  }
  return Sophus::SO3d();
}

} // namespace odometry
} // namespace pnec
