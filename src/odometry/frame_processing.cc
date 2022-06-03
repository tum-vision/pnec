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
#include "odometry_output.h"
#include "visualization.h"

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

  BOOST_LOG_TRIVIAL(debug) << "Finding matches between frames "
                           << prev_frame->id() << " and " << curr_frame->id();
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

  BOOST_LOG_TRIVIAL(debug) << "Finding the relative pose between "
                           << prev_frame->id() << " and " << curr_frame->id();
  Sophus::SE3d rel_pose = f2f_pose_estimation_->Align(
      prev_frame, curr_frame, matches, prev_pose, inliers, frame_timing, false,
      results_folder + "ablation/");

  view_graph_->AddView(curr_view);
  bool success = view_graph_->Connect(prev_view, curr_view, matches, rel_pose);
  if (!success) {
    return false;
  }
  view_graph_->ShortenViewGraph();

  pnec::visualization::plotMatches(prev_frame, curr_frame, matches, inliers,
                                   "_presentation");

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

bool FrameProcessing::ProcessUncertaintyExtraction(
    pnec::frames::BaseFrame::Ptr host_frame,
    pnec::frames::BaseFrame::Ptr target_frame, Sophus::SE3d init_pose,
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
      host_frame, target_frame, matches, init_pose, inliers, dummy_timing,
      false, results_folder + "ablation/");
  std::cout << "Finished aligning" << std::endl;
  std::cout << "Found " << inliers.size() << " inliers." << std::endl;

  // Pass the inliers to the prev_frame, to extract patches and covariances
  std::vector<size_t> inlier_ids;
  for (const auto &inlier : inliers) {
    cv::DMatch match = matches[inlier];
    if (extract_host) {
      inlier_ids.push_back(match.queryIdx);
    } else {
      inlier_ids.push_back(match.trainIdx);
    }
  }
  std::cout << "Saving " << inlier_ids.size() << " inlier patches" << std::endl;

  if (extract_host) {
    host_frame->SaveInlierPatches(host_frame->keypoints(inlier_ids),
                                  extraction_counter_, results_folder);
  } else {
    target_frame->SaveInlierPatches(target_frame->keypoints(inlier_ids),
                                    extraction_counter_, results_folder);
  }
  std::cout << "Saved Patches." << std::endl;

  return true;
}

bool FrameProcessing::ProcessUncertaintyExtractionVO(
    pnec::frames::BaseFrame::Ptr frame, Sophus::SE3d init_pose,
    std::string results_folder, bool save_uncertainty) {
  // Create View
  pnec::odometry::View::Ptr curr_view =
      std::make_shared<pnec::odometry::View>(frame);

  const int m = view_graph_->GraphSize();

  // no pose estimation for the first frame
  BOOST_LOG_TRIVIAL(info) << "Graph size " << m;
  if (m == 1) {
    std::cout << "No pose estimation for the first frame" << std::endl;
    view_graph_->AddView(curr_view);
    return true;
  }

  pnec::frames::BaseFrame::Ptr curr_frame = curr_view->Frame();

  const int curr_view_idx = m;
  int prev_view_idx = m - 2;

  int count_connections = 0;

  pnec::odometry::View::Ptr prev_view;
  view_graph_->GetViewByPos(prev_view_idx, prev_view);
  pnec::frames::BaseFrame::Ptr prev_frame = prev_view->Frame();
  BOOST_LOG_TRIVIAL(info) << "Host Frame ID: " << prev_frame->id()
                          << " Target Frame ID: " << curr_frame->id();

  bool skipping_frame;
  pnec::FeatureMatches matches =
      matcher_->FindMatches(prev_frame, curr_frame, skipping_frame);

  BOOST_LOG_TRIVIAL(info) << "Found: " << matches.size() << " matches.";

  std::vector<int> inliers;

  pnec::common::FrameTiming dummy_timing(0);
  Sophus::SE3d rel_pose = f2f_pose_estimation_->Align(
      prev_frame, curr_frame, matches, init_pose, inliers, dummy_timing, false,
      results_folder + "ablation/");

  BOOST_LOG_TRIVIAL(info) << "Found: " << inliers.size() << " inliers.";

  if (save_uncertainty) {
    // Pass the inliers to the prev_frame, to extract patches and covariances
    std::vector<size_t> host_inlier_ids;
    std::vector<size_t> target_inlier_ids;

    for (const auto &inlier : inliers) {
      cv::DMatch match = matches[inlier];
      host_inlier_ids.push_back(match.queryIdx);
      target_inlier_ids.push_back(match.trainIdx);
    }

    std::string image_results_folder =
        results_folder + std::to_string(curr_frame->id()) + "/";
    if (!boost::filesystem::exists(image_results_folder)) {
      boost::filesystem::create_directory(image_results_folder);
    }

    // save ground truth
    pnec::out::SavePose(image_results_folder, "pose", curr_frame->Timestamp(),
                        init_pose);

    BOOST_LOG_TRIVIAL(info)
        << "Saving " << host_inlier_ids.size() << " inlier patches";
    curr_frame->SaveInlierPatchesStructured(
        prev_frame->keypoints(host_inlier_ids),
        curr_frame->keypoints(target_inlier_ids), extraction_counter_,
        image_results_folder);
    BOOST_LOG_TRIVIAL(info) << "Saved Patches.";
  }

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
