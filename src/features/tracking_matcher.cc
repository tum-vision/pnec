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

#include "tracking_matcher.h"

#include "common.h"

namespace pnec {
namespace features {

TrackingMatcher::TrackingMatcher(int sufficient_matches, double min_local_rad)
    : sufficient_matches_{sufficient_matches}, min_local_rad_{min_local_rad} {}

TrackingMatcher::~TrackingMatcher() {}

pnec::FeatureMatches
TrackingMatcher::FindMatches(pnec::frames::BaseFrame::Ptr prev_frame,
                             pnec::frames::BaseFrame::Ptr curr_frame,
                             bool &skipping) const {
  pnec::FeatureMatches matches;
  // find matching keypoint ids, store them in curr2prev_map
  for (int prev_idx = 0; prev_idx < prev_frame->keypoint_ids().size();
       prev_idx++) {
    for (int curr_idx = 0; curr_idx < curr_frame->keypoint_ids().size();
         curr_idx++) {
      uint32_t prev_keypoint_id = prev_frame->keypoint_ids()[prev_idx];
      uint32_t curr_keypoint_id = curr_frame->keypoint_ids()[curr_idx];

      if (prev_keypoint_id > curr_keypoint_id) {
        continue;
      }
      if (prev_keypoint_id == curr_keypoint_id) {
        matches.push_back(cv::DMatch(prev_idx, curr_idx, 0));
        break;
      }
      if (prev_keypoint_id < curr_keypoint_id) {
        break;
      }
    }
  }

  // find and update mean rad
  std::vector<double> dists;
  dists.reserve(matches.size());
  for (auto match : matches) {
    const auto &prev_p = prev_frame->undistortedKeypoints()[match.queryIdx];
    const auto &curr_p = curr_frame->undistortedKeypoints()[match.trainIdx];
    double d = cv::norm(prev_p.pt - curr_p.pt);
    dists.push_back(d);
  }

  double m_local_rad =
      std::accumulate(dists.begin(), dists.end(), 0.0) / dists.size();

  if (m_local_rad < min_local_rad_) {
    skipping = true;
  } else {
    skipping = false;
  }
  return matches;
}

} // namespace features
} // namespace pnec