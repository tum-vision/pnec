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

  pnec::frames::BaseFrame::Ptr target_frame = curr_view->Frame();

  const int curr_view_idx = m - 1;
  int prev_view_idx = curr_view_idx - skip - 1;

  int count_connections = 0;

  pnec::odometry::View::Ptr prev_view;
  view_graph_->GetViewByPos(prev_view_idx, prev_view);
  pnec::frames::BaseFrame::Ptr host_frame = prev_view->Frame();

  BOOST_LOG_TRIVIAL(debug) << "Finding matches between frames "
                           << host_frame->id() << " and " << target_frame->id();
  bool skipping_frame;
  pnec::FeatureMatches matches =
      matcher_->FindMatches(host_frame, target_frame, skipping_frame);

  if (skipping_frame && !no_skip_) {
    return false;
  }

  std::vector<int> inliers;

  Sophus::SO3d prev_rel_rotation = Sophus::SO3d();
  if (target_frame->id() > 1) {
    prev_rel_rotation = PrevRelRotation(prev_view, prev_view_idx);
  }

  Sophus::SE3d prev_pose(prev_rel_rotation, Eigen::Vector3d(0.0, 0.0, 0.0));

  BOOST_LOG_TRIVIAL(debug) << "Finding the relative pose between "
                           << host_frame->id() << " and " << target_frame->id();
  Sophus::SE3d rel_pose = f2f_pose_estimation_->Align(
      host_frame, target_frame, matches, prev_pose, inliers, frame_timing, true,
      results_folder + "ablation/");

  view_graph_->AddView(curr_view);
  bool success = view_graph_->Connect(prev_view, curr_view, matches, rel_pose);
  if (!success) {
    return false;
  }
  view_graph_->ShortenViewGraph();

  if (visualization_options_.keypoints != pnec::visualization::Options::NO) {
    pnec::visualization::plotMatches(host_frame, target_frame, matches, inliers,
                                     visualization_options_, "");
  }

  count_connections++;
  prev_view_idx--;

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
