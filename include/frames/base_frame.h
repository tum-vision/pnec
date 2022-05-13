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

#ifndef FRAMES_BASE_FRAME_H_
#define FRAMES_BASE_FRAME_H_

#include <boost/log/trivial.hpp>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdint.h>
#include <stdio.h>

#include "opencv2/opencv.hpp"
#include <Eigen/Core>

#include "camera.h"
#include "keypoints.h"

namespace pnec {
namespace frames {
class BaseFrame {

public:
  using Ptr = std::shared_ptr<BaseFrame>;
  BaseFrame(int id, double timestamp, const std::string path)
      : id_(id), timestamp_{timestamp}, path_(path) {}

  ~BaseFrame() { BOOST_LOG_TRIVIAL(debug) << "Getting rid of frame " << id_; }

  pnec::features::KeyPoints &keypoints() { return keypoints_; }

  pnec::features::KeyPoints keypoints(std::vector<size_t> &ids);

  double Timestamp() { return timestamp_; }

  // std::vector<Eigen::Vector3d> &ProjectedPoints() { return projected_points_;
  // } std::vector<Eigen::Matrix3d> &ProjectedCovariances() {
  //   return projected_covariances_;
  // }

  int id() const { return id_; }

  cv::Mat getImage();

  std::string getPath() const { return path_; }

  void PlotFeatures();

protected:
  const int id_;
  const double timestamp_;
  const std::string path_;

  pnec::features::KeyPoints keypoints_;

  // void undistortKeypoints();

  // void UnscentedTransform();

  // // Before the unscented Transform
  // std::vector<cv::KeyPoint> keypoints_;
  // std::vector<uint32_t> keypoint_ids_;

  // std::vector<cv::KeyPoint> undistorted_keypoints_;
  // std::vector<Eigen::Matrix2d> covariances_;

  // // After the Unscented Transform
  // std::vector<Eigen::Matrix3d> projected_covariances_;
  // std::vector<Eigen::Vector3d> projected_points_;
};
} // namespace frames
} // namespace pnec

#endif // FRAMES_BASE_FRAME_H_
