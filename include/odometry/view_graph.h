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

#ifndef ODOMETRY_VIEW_GRAPH_H_
#define ODOMETRY_VIEW_GRAPH_H_

#include <boost/filesystem.hpp>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "base_frame.h"
#include "view.h"

namespace pnec {
namespace odometry {
class ViewGraph {
public:
  using Ptr = std::shared_ptr<ViewGraph>;
  ViewGraph(int max_graph_size, const std::string &results_path)
      : max_graph_size_{max_graph_size}, results_path_{results_path} {
    if (!boost::filesystem::exists(results_path_ + poses_directory_)) {
      boost::filesystem::create_directory(results_path_ + poses_directory_);
    }
  }

  int GraphSize() const;

  void AddView(pnec::odometry::View::Ptr view);
  pnec::odometry::View::Ptr CurrentView();
  bool GetViewByPos(const int position, pnec::odometry::View::Ptr &view);
  bool GetViewById(const int id, pnec::odometry::View::Ptr &view);

  bool Connect(pnec::odometry::View::Ptr v1, pnec::odometry::View::Ptr v2,
               FeatureMatches matches, const Sophus::SE3d &rel_pose);
  bool RemoveConnection(pnec::odometry::View::Ptr v1,
                        pnec::odometry::View::Ptr v2);
  bool AreConnected(const pnec::odometry::View::Ptr v1,
                    const pnec::odometry::View::Ptr v2);

  void fixPose(int idx, Sophus::SE3d &pose);
  bool isPoseFixed(int idx) const;
  int countFixedPoses() const;

  void ShortenViewGraph();

  Sophus::SE3d RelPose(pnec::odometry::View::Ptr v1,
                       pnec::odometry::View::Ptr v2);

  void savePoses();

private:
  std::vector<pnec::odometry::View::Ptr> views_;

  std::vector<bool> fixed_mask_; // mask for fixed views

  int max_graph_size_;

  std::string results_path_;
  std::string poses_directory_ = "rot_avg/";
  std::string poses_filename_ = "poses";

  std::ios_base::openmode output_mode_ = std::ios_base::out;
};
} // namespace odometry
} // namespace pnec

#endif // ODOMETRY_VIEW_GRAPH_H_
