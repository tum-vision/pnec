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

#include "keypoints.h"

#include "common.h"
#include <opencv2/core/eigen.hpp>

namespace pnec {
namespace features {
KeyPoint::KeyPoint(Eigen::Vector2d point, Eigen::Matrix2d covariance,
                   Eigen::Matrix3d hessian)
    : point_(point), img_covariance_(covariance), hessian_(hessian) {
  Unproject();
}

void KeyPoint::Unproject() {
  Eigen::Matrix3d K;
  cv::cv2eigen(pnec::Camera::instance().cameraParameters().intrinsic(), K);
  Eigen::Matrix3d K_inv = K.inverse();

  bearing_vector_ = pnec::common::Unproject(point_, K_inv);

  Eigen::Vector3d mu(point_(0), point_(1), 1.0);

  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  cov.topLeftCorner(2, 2) = img_covariance_;
  bv_covariance_ = pnec::common::UnscentedTransform(mu, cov, K_inv, 1.0,
                                                    pnec::common::Pinhole);
}

cv::KeyPoint KeyPointToCV(KeyPoint keypoint) {
  return cv::KeyPoint(keypoint.point_(0), keypoint.point_(1), 1.0);
}

} // namespace features
} // namespace pnec