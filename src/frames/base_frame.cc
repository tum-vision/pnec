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

#include "base_frame.h"

#include <ctime>

#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "camera.h"
#include "common.h"
#include "keypoints.h"
#include "visualization.h"

namespace pnec {
namespace frames {

pnec::features::KeyPoints BaseFrame::keypoints(std::vector<size_t> &ids) {

  pnec::features::KeyPoints filtered_keypoints;
  for (size_t id : ids) {
    // pnec::features::KeyPoint keypoint =;
    filtered_keypoints[id] = keypoints_[id];
  }
  return filtered_keypoints;
}

cv::Mat BaseFrame::getImage() {
  cv::Mat image;

  image = cv::imread(this->path_, cv::IMREAD_GRAYSCALE); // Read the file

  if (!image.data) // Check for invalid input
  {
    class FrameException : public std::exception {
      virtual const char *what() const throw() {
        return "Could not read frame";
      }
    } frameException;
    throw frameException;
  }
  return image;
}

void BaseFrame::PlotFeatures() {

  const cv::Mat image = getImage();
  for (auto const &[id, keypoint] : keypoints_) {
    cv::Mat cv_cov;
    cv::eigen2cv(keypoint.img_covariance_, cv_cov);
    cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
        2.4477, pnec::features::KeyPointToCV(keypoint).pt, cv_cov);
    cv::ellipse(image, ellipse, cv::Scalar::all(255), 2);
  }
  cv::imshow("EllipseDemo", image);
  cv::waitKey(0);
}
} // namespace frames
} // namespace pnec
