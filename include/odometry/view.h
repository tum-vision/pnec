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

#ifndef ODOMETRY_VIEW_H_
#define ODOMETRY_VIEW_H_

// #include <algorithm>  // std::sort
// #include <functional> // std::greater
#include <map>
#include <stdio.h>
#include <vector>

#include <sophus/se3.hpp>

#include "base_frame.h"
#include "common.h"

namespace pnec {
namespace odometry {
class View {
public:
  using Ptr = std::shared_ptr<View>;
  using ConstPtr = std::shared_ptr<const View>;
  class Connection {
  public:
    Connection(View::Ptr v1, View::Ptr v2, pnec::FeatureMatches matches,
               const Sophus::SE3d &rel_pose)
        : v1_(v1), v2_(v2), matches_(std::move(matches)), rel_pose_{rel_pose} {}

    FeatureMatches &Matches() { return matches_; }
    size_t size() const { return matches_.size(); }
    const Sophus::SE3d &Pose() const { return rel_pose_; }

  private:
    // TODO: temporatily removed
    View::Ptr v1_;
    View::Ptr v2_;
    FeatureMatches matches_;
    Sophus::SE3d rel_pose_;
  };

  typedef std::map<pnec::odometry::View::Ptr, Connection *> Connections;

  View(pnec::frames::BaseFrame::Ptr frame) : frame_{frame} {}

  pnec::frames::BaseFrame::Ptr Frame() { return frame_; }

  Sophus::SE3d &Pose() { return pose_; }

  double Timestamp() { return frame_->Timestamp(); }

  const Connections &connections() const { return connections_; }

  void AddConnection(const View::Ptr view, Connection *connection) {
    connections_[view] = connection;
  }

  Connection *GetConnection(const View::Ptr view) { return connections_[view]; }

  void RemoveConnection(const View::Ptr view) { connections_.erase(view); }

  int CountConnections(const View::Ptr view) const {
    return connections_.count(view);
  }

private:
  pnec::frames::BaseFrame::Ptr frame_;
  Sophus::SE3d pose_;
  std::map<pnec::odometry::View::Ptr, Connection *> connections_;
};

} // namespace odometry
} // namespace pnec

#endif // ODOMETRY_VIEW_H_
