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

#ifndef COMMON_VISUALIZATION_H_
#define COMMON_VISUALIZATION_H_

#include <string.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "base_frame.h"
#include "common.h"

namespace pnec {
namespace visualization {
struct Options {
  enum VisualizationLevel { NO, INLIER, TRACKED, ALL };

  Options(std::string folder,
          Options::VisualizationLevel kp_level =
              Options::VisualizationLevel::INLIER,
          Options::VisualizationLevel cov_level =
              Options::VisualizationLevel::INLIER)
      : base_folder(folder), keypoints(kp_level), covariances(cov_level) {
    // only visualize covariances for which the keypoints are also visualized
    if (covariances > keypoints) {
      covariances = keypoints;
    }
  }
  std::string base_folder;

  VisualizationLevel keypoints;
  VisualizationLevel covariances;
  cv::Scalar inlier_color = cv::Scalar(0, 255, 0);
  cv::Scalar tracked_color = cv::Scalar(0, 0, 255);
  cv::Scalar covariance_color = cv::Scalar(255, 0, 0);
  double cov_scaling = 2.4477 * 10.0;
  int cov_thickness = 2;
};

cv::RotatedRect GetErrorEllipse(double chisquare_val, cv::Point2f mean,
                                cv::Mat covmat);

char plotMatches(pnec::frames::BaseFrame::Ptr host_frame,
                 pnec::frames::BaseFrame::Ptr target_frame,
                 pnec::FeatureMatches &matches, std::vector<int> &inliers,
                 pnec::visualization::Options visualization_options,
                 std::string suffix);

char plotMatches(pnec::frames::BaseFrame::Ptr host_frame, cv::Mat &host_image,
                 pnec::frames::BaseFrame::Ptr target_frame,
                 cv::Mat &target_image, pnec::FeatureMatches &matches,
                 std::vector<int> &inliers,
                 pnec::visualization::Options visualization_options,
                 std::string suffix);
} // namespace visualization
} // namespace pnec

#endif // COMMON_VISUALIZATION_H_
