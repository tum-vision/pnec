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

bool valid_covariance(Eigen::Matrix2d covariance) {
  if (covariance.trace() > 1.0e2) {
    return false;
  }
  return true;
}

void BaseFrame::SaveInlierPatches(const std::vector<int> &inlier_kp_idx,
                                  size_t &counter, std::string results_dir) {
  const cv::Mat image = getImage();

  std::ofstream outfile_cov;
  std::ofstream outfile_hessian;

  outfile_cov.open(results_dir + "covariances.txt", std::ios_base::app);
  outfile_hessian.open(results_dir + "hessians.txt", std::ios_base::app);

  const static Eigen::IOFormat CSVFormat(
      Eigen::FullPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "\n");

  for (const auto &idx : inlier_kp_idx) {
    cv::KeyPoint keypoint = keypoints_[idx];

    int width = 6;
    if ((keypoint.pt.x - width) > 0 &&
        (keypoint.pt.x + width) < (image.size().width - 1) &&
        (keypoint.pt.y - width) > 0 &&
        (keypoint.pt.y + width) < (image.size().height - 1)) {
      Eigen::Matrix2d covariance = covariances_[idx];
      Eigen::Matrix3d hessian = hessians_[idx];

      if (!valid_covariance(covariance)) {
        std::cout << "found invalid covariance" << std::endl;
        continue;
      }

      cv::Mat patch =
          image(cv::Range(keypoint.pt.y - width, keypoint.pt.y + width),
                cv::Range(keypoint.pt.x - width, keypoint.pt.x + width));

      cv::imwrite(results_dir + std::to_string(counter++) + ".png", patch);
      outfile_cov << covariance.format(CSVFormat);
      outfile_hessian << hessian.format(CSVFormat);
    } else {
      std::cout << idx << " out of image range" << std::endl;
    }
  }
  outfile_cov.close();
  outfile_hessian.close();
  // PlotFeatures();
}

void BaseFrame::SaveInlierPatchesStructured(
    const std::vector<int> &inlier_kp_idx, size_t &counter,
    std::string results_dir, const std::vector<cv::KeyPoint> &host_keypoints,
    const std::vector<cv::KeyPoint> &target_keypoints) {
  const cv::Mat image = getImage();

  std::ofstream outfile_cov;
  std::ofstream outfile_hessian;
  std::ofstream outfile_kp;

  outfile_kp.open(results_dir + "keypoints.txt", std::ios_base::app);
  outfile_cov.open(results_dir + "covariances.txt", std::ios_base::app);
  outfile_hessian.open(results_dir + "hessians.txt", std::ios_base::app);

  const static Eigen::IOFormat CSVFormat(
      Eigen::FullPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "\n");

  size_t curr_frame_count = 0;
  for (const auto &idx : inlier_kp_idx) {
    cv::KeyPoint keypoint = keypoints_[idx];
    cv::KeyPoint host_keypoint = host_keypoints[curr_frame_count];
    cv::KeyPoint target_keypoint = target_keypoints[curr_frame_count];

    int width = 6;
    if ((keypoint.pt.x - width) > 0 &&
        (keypoint.pt.x + width) < (image.size().width - 1) &&
        (keypoint.pt.y - width) > 0 &&
        (keypoint.pt.y + width) < (image.size().height - 1)) {
      Eigen::Matrix2d covariance = covariances_[idx];
      Eigen::Matrix3d hessian = hessians_[idx];

      if (!valid_covariance(covariance)) {
        std::cout << "found invalid covariance" << std::endl;
        continue;
      }

      cv::Mat patch =
          image(cv::Range(keypoint.pt.y - width, keypoint.pt.y + width),
                cv::Range(keypoint.pt.x - width, keypoint.pt.x + width));

      std::string image_counter = std::to_string(counter++);
      cv::imwrite(results_dir + image_counter + ".png", patch);
      outfile_kp << id_ << ", " << curr_frame_count << ", " << image_counter
                 << ", " << host_keypoint.pt.x << ", " << host_keypoint.pt.y
                 << ", " << target_keypoint.pt.x << ", " << target_keypoint.pt.y
                 << "\n";
      outfile_cov << id_ << ", " << curr_frame_count << ", " << image_counter
                  << ", " << covariance.format(CSVFormat);
      outfile_hessian << id_ << ", " << curr_frame_count << ", "
                      << image_counter << ", " << hessian.format(CSVFormat);
      curr_frame_count++;
    } else {
      std::cout << idx << " out of image range" << std::endl;
    }
  }
  outfile_cov.close();
  outfile_hessian.close();
  // PlotFeatures();
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
