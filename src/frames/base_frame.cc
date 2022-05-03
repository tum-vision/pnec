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
#include "visualization.h"

namespace pnec {
namespace frames {
void BaseFrame::undistortKeypoints() {
  const pnec::CameraParameters &cam_pars =
      Camera::instance().cameraParameters();
  const cv::Vec4d &dist_coef = cam_pars.dist_coef();

  if (dist_coef(0) == 0.0) {
    undistorted_keypoints_ = keypoints_;
    return;
  }

  // Fill matrix with points
  const int n = (int)keypoints_.size();
  cv::Mat mat(n, 2, CV_64F);
  for (int i = 0; i < n; i++) {
    mat.at<double>(i, 0) = keypoints_[i].pt.x;
    mat.at<double>(i, 1) = keypoints_[i].pt.y;
  }

  // undistort points
  mat = mat.reshape(2);
  cv::undistortPoints(mat, mat, cam_pars.intrinsic(), dist_coef, cv::Mat(),
                      cam_pars.intrinsic());
  mat = mat.reshape(1);

  // Fill undistorted keypoint vector
  // undistorted_keypoints_.resize(n);
  undistorted_keypoints_.reserve(n);
  for (int i = 0; i < n; i++) {
    cv::KeyPoint kp = keypoints_[i];
    kp.pt.x = mat.at<double>(i, 0);
    kp.pt.y = mat.at<double>(i, 1);
    undistorted_keypoints_.push_back(kp);
  }
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

void BaseFrame::UnscentedTransform() {
  Eigen::Matrix3d K;
  cv::cv2eigen(pnec::Camera::instance().cameraParameters().intrinsic(), K);
  projected_covariances_.clear();
  projected_points_.clear();
  Eigen::Matrix3d K_inv = K.inverse();

  for (int i = 0; i < undistorted_keypoints_.size(); i++) {
    cv::KeyPoint kp = undistorted_keypoints_[i];
    Eigen::Vector3d mu(kp.pt.x, kp.pt.y, 1.0);

    projected_points_.push_back((K_inv * mu).normalized());
    if (i < covariances_.size()) {
      Eigen::Matrix2d cov_2d = covariances_[i];
      Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
      cov.topLeftCorner(2, 2) = cov_2d;
      projected_covariances_.push_back(pnec::common::UnscentedTransform(
          mu, cov, K_inv, 1.0, pnec::common::Pinhole));
    }
  }
}

void BaseFrame::PlotFeatures() {
  const cv::Mat image = getImage();
  for (size_t i = 0; i < covariances_.size(); i++) {
    cv::Mat cv_cov;
    cv::eigen2cv(covariances_[i], cv_cov);
    cv::RotatedRect ellipse =
        pnec::visualization::GetErrorEllipse(2.4477, keypoints_[i].pt, cv_cov);
    cv::ellipse(image, ellipse, cv::Scalar::all(255), 2);
  }
  cv::imshow("EllipseDemo", image);
  cv::waitKey(0);
}
} // namespace frames
} // namespace pnec
