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

#include "converter.h"

#include "common.h"
namespace pnec {
namespace converter {
basalt::OpticalFlowInput::Ptr OpticalFlowFromOpenCV(const cv::Mat &image,
                                                    const int64_t timestamp) {
  std::vector<basalt::ImageData> res(1);
  res[0].img.reset(new basalt::ManagedImage<uint16_t>(image.cols, image.rows));

  const uint8_t *data_in = image.ptr();
  uint16_t *data_out = res[0].img->ptr;

  size_t full_size = image.cols * image.rows;
  for (size_t i = 0; i < full_size; i++) {
    int val = data_in[i];
    val = val << 8;
    data_out[i] = val;
  }

  basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);

  data->t_ns = timestamp;
  data->img_data = res;

  return data;
}

// void KeypointsFromOpticalFlow(basalt::OpticalFlowResult::Ptr result,
//                               std::vector<cv::KeyPoint> &keypoints,
//                               std::vector<uint32_t> &keypoint_ids) {
//   // single camera
//   for (auto observation : result->observations[0]) {
//     keypoint_ids.push_back(observation.first);
//     keypoints.push_back(
//         cv::KeyPoint(observation.second(0, 2), observation.second(1, 2), 1));
//   }
// }

pnec::features::KeyPoints KeypointsFromOpticalFlow(
    basalt::KLTPatchOpticalFlow<float, basalt::Pattern52> &optical_flow,
    bool undistort) {
  pnec::features::KeyPoints keypoints;
  for (auto observation : optical_flow.Transforms()->observations[0]) {
    pnec::features::KeyPointID id = observation.first;
    Eigen::Vector2d point(observation.second(0, 2), observation.second(1, 2));
    if (undistort) {
      point = pnec::common::Undistort(point);
    }
    Eigen::Matrix2d cov =
        optical_flow.Covariances()[0][observation.first].cast<double>();
    // dummy
    Eigen::Matrix3d hessian = Eigen::Matrix3d::Zero();
    pnec::features::KeyPoint keypoint(point, cov, hessian);

    keypoints[id] = keypoint;
  }
  return keypoints;
}

} // namespace converter
} // namespace pnec
