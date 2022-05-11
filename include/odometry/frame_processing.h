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

#ifndef ODOMETRY_FRAME_PROCESSING_H_
#define ODOMETRY_FRAME_PROCESSING_H_

#include <stdio.h>
#include <vector>

#include "base_frame.h"
#include "base_matcher.h"
#include "frame2frame.h"
#include "view.h"
#include "view_graph.h"

namespace pnec {
namespace odometry {
enum VisualizationLevel {
  NoViz,
  Features,
  InitMatches,
  InitMatchesRANSAC,
  AllMatches,
  AllMatchesRANSAC
};

class FrameProcessing {
public:
  FrameProcessing(
      pnec::odometry::ViewGraph::Ptr view_graph,
      pnec::rel_pose_estimation::Frame2Frame::Ptr f2f_pose_estimation,
      pnec::features::BaseMatcher::Ptr matcher, bool no_skip = false)
      : view_graph_{view_graph}, matcher_{matcher},
        f2f_pose_estimation_{f2f_pose_estimation}, no_skip_{no_skip} {};

  // Process a frame
  // return true if frame is added, otherwie false
  bool ProcessFrame(pnec::frames::BaseFrame::Ptr frame,
                    pnec::common::FrameTiming &frame_timing,
                    std::string results_folder, int win_size = 5);

  bool ProcessUncertaintyExtraction(pnec::frames::BaseFrame::Ptr host_frame, pnec::frames::BaseFrame::Ptr target_frame, Sophus::SE3d init_pose,
    std::string results_folder, bool extract_host = true);

private:
  Sophus::SO3d PrevRelRotation(pnec::odometry::View::Ptr prev_view,
                               int prev_view_idx);

  pnec::odometry::ViewGraph::Ptr view_graph_;
  pnec::features::BaseMatcher::Ptr matcher_;
  pnec::rel_pose_estimation::Frame2Frame::Ptr f2f_pose_estimation_;

  // opencv imshow
  VisualizationLevel visualization_level_;

  int skip_showing_n_;
  int skipping_counter_;

  bool no_skip_;
  size_t extraction_counter_ = 0;
};
} // namespace odometry
} // namespace pnec

#endif // ODOMETRY_FRAME_PROCESSING_H_
