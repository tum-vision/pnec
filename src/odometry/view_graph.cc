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

#include "view_graph.h"

#include <cmath>
#include <ctime>

#include "common.h"
#include "odometry_output.h"

namespace pnec {
namespace odometry {

int ViewGraph::GraphSize() const { return views_.size(); }

void ViewGraph::AddView(pnec::odometry::View::Ptr view) {
  views_.push_back(view);
  fixed_mask_.push_back(false);
}

pnec::odometry::View::Ptr ViewGraph::CurrentView() { return views_.back(); }

bool ViewGraph::GetViewByPos(const int position,
                             pnec::odometry::View::Ptr &view) {
  if (position >= views_.size()) {
    return false;
  }
  view = views_[position];
  return true;
}

bool ViewGraph::GetViewById(const int id, pnec::odometry::View::Ptr &view) {
  for (auto view_ptr : views_) {
    if (view_ptr->Frame()->id() == id) {
      view = view_ptr;
      return true;
    } else if (view_ptr->Frame()->id() > id) {
      return false;
    }
  }
  return false;
}

bool ViewGraph::Connect(pnec::odometry::View::Ptr v1,
                        pnec::odometry::View::Ptr v2, FeatureMatches matches,
                        const Sophus::SE3d &rel_pose) {
  assert(matches.size() > 4); //
  if (v1->CountConnections(v2) > 0) {
    assert(v2->CountConnections(v1) > 0); // we must have an undirected graph
    return false;
  }

  // Create a Connection object
  pnec::odometry::View::Connection *connection =
      new pnec::odometry::View::Connection(v1, v2, std::move(matches),
                                           rel_pose);

  v1->AddConnection(v2, connection);
  v2->AddConnection(v1, connection);

  return true;
}

bool ViewGraph::RemoveConnection(pnec::odometry::View::Ptr v1,
                                 pnec::odometry::View::Ptr v2) {
  if (!AreConnected(v1, v2)) {
    return false;
  }
  pnec::odometry::View::Connection *connection = v1->GetConnection(v2);
  delete connection;
  v1->RemoveConnection(v2);
  v2->RemoveConnection(v1);
  return true;
}

bool ViewGraph::AreConnected(const pnec::odometry::View::Ptr v1,
                             const pnec::odometry::View::Ptr v2) {
  if (v1->CountConnections(v2) == 0) {
    assert(v2->CountConnections(v1) == 0);
    return false;
  }

  return true;
}

void ViewGraph::fixPose(int idx, Sophus::SE3d &new_pose) {
  assert(fixed_mask_.size() == views_.size());

  fixed_mask_[idx] = true;

  pnec::odometry::View::Ptr view = views_[idx];
  Sophus::SE3d &pose = view->Pose();
  pose = new_pose;

  assert(fixed_mask_.size() == views_.size());
}

bool ViewGraph::isPoseFixed(int idx) const { return fixed_mask_[idx]; }

int ViewGraph::countFixedPoses() const {
  int resp = 0;
  for (const auto x : fixed_mask_) {
    if (x)
      resp++;
  }
  return resp;
}

void ViewGraph::ShortenViewGraph() {
  while (views_.size() > max_graph_size_) {
    pnec::odometry::View::Ptr earliest_view = views_.front();

    while (earliest_view->connections().size() > 0) {
      RemoveConnection(earliest_view,
                       earliest_view->connections().begin()->first);
    }
    // for (const auto &connection : earliest_view->connections()) {
    //   RemoveConnection(earliest_view, connection.first);
    // }
    // Save Pose before deleting it
    pnec::out::SavePose(results_path_ + poses_directory_, poses_filename_,
                        earliest_view->Timestamp(), earliest_view->Pose(),
                        output_mode_);

    if (output_mode_ == std::ios_base::out) {
      output_mode_ = std::ios_base::app;
    }
    views_.erase(views_.begin());
  }
}

Sophus::SE3d ViewGraph::RelPose(pnec::odometry::View::Ptr v1,
                                pnec::odometry::View::Ptr v2) {
  if (AreConnected(v1, v2)) {
    return v1->GetConnection(v2)->Pose();
  } else {
    return Sophus::SE3d();
  }
}

void ViewGraph::savePoses() {
  for (const auto &view : views_) {
    pnec::out::SavePose(results_path_ + poses_directory_, poses_filename_,
                        view->Timestamp(), view->Pose(), output_mode_);
    if (output_mode_ == std::ios_base::out) {
      output_mode_ = std::ios_base::app;
    }
  }
}

} // namespace odometry
} // namespace pnec
