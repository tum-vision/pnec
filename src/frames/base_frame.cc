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

// https://answers.opencv.org/question/13876/read-multiple-images-from-folder-and-concat-display-images-in-single-window-opencv-c-visual-studio-2010/
cv::Mat createOne(std::vector<cv::Mat> &images, int cols, int min_gap_size) {
  // let's first find out the maximum dimensions
  int max_width = 0;
  int max_height = 0;
  for (int i = 0; i < images.size(); i++) {
    // check if type is correct
    // you could actually remove that check and convert the image
    // in question to a specific type
    if (i > 0 && images[i].type() != images[i - 1].type()) {
      std::cerr << "WARNING:createOne failed, different types of images";
      return cv::Mat();
    }
    max_height = std::max(max_height, images[i].rows);
    max_width = std::max(max_width, images[i].cols);
  }
  // number of images in y direction
  int rows = std::ceil(images.size() / cols);

  // create our result-matrix
  cv::Mat result = cv::Mat::zeros(rows * max_height + (rows - 1) * min_gap_size,
                                  cols * max_width + (cols - 1) * min_gap_size,
                                  images[0].type());
  size_t i = 0;
  int current_height = 0;
  int current_width = 0;
  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      if (i >= images.size()) // shouldn't happen, but let's be safe
        return result;
      // get the ROI in our result-image
      cv::Mat to(result,
                 cv::Range(current_height, current_height + images[i].rows),
                 cv::Range(current_width, current_width + images[i].cols));
      // copy the current image to the ROI
      images[i++].copyTo(to);
      current_width += max_width + min_gap_size;
    }
    // next line - reset width and update height
    current_width = 0;
    current_height += max_height + min_gap_size;
  }
  return result;
}

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

void BaseFrame::SaveInlierPatches(const pnec::features::KeyPoints keypoints,
                                  size_t &counter, std::string results_dir) {
  const cv::Mat image = getImage();

  std::ofstream outfile_cov;
  std::ofstream outfile_hessian;

  outfile_cov.open(results_dir + "covariances.txt", std::ios_base::app);
  outfile_hessian.open(results_dir + "hessians.txt", std::ios_base::app);

  const static Eigen::IOFormat CSVFormat(
      Eigen::FullPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "\n");

  int width = 6;
  for (auto const &[id, keypoint] : keypoints) {
    cv::KeyPoint point = pnec::features::KeyPointToCV(keypoint);
    bool in_bounds = (point.pt.x - width) > 0 &&
                     (point.pt.x + width) < (image.size().width - 1) &&
                     (point.pt.y - width) > 0 &&
                     (point.pt.y + width) < (image.size().height - 1);
    if (in_bounds) {
      if (!valid_covariance(keypoint.img_covariance_)) {
        std::cout << "found invalid covariance" << std::endl;
        continue;
      }
      cv::Mat patch = image(cv::Range(point.pt.y - width, point.pt.y + width),
                            cv::Range(point.pt.x - width, point.pt.x + width));

      cv::imwrite(results_dir + std::to_string(counter++) + ".png", patch);
      outfile_cov << keypoint.img_covariance_.format(CSVFormat);
      outfile_hessian << keypoint.hessian_.format(CSVFormat);
    } else {
      std::cout << id << " out of image range" << std::endl;
    }
  }

  // for (const auto &idx : inlier_kp_idx) {
  //   cv::KeyPoint keypoint = keypoints_[idx];

  //   int width = 6;
  //   if ((keypoint.pt.x - width) > 0 &&
  //       (keypoint.pt.x + width) < (image.size().width - 1) &&
  //       (keypoint.pt.y - width) > 0 &&
  //       (keypoint.pt.y + width) < (image.size().height - 1)) {
  //     Eigen::Matrix2d covariance = covariances_[idx];
  //     Eigen::Matrix3d hessian = hessians_[idx];

  //     if (!valid_covariance(covariance)) {
  //       std::cout << "found invalid covariance" << std::endl;
  //       continue;
  //     }

  //     cv::Mat patch =
  //         image(cv::Range(keypoint.pt.y - width, keypoint.pt.y + width),
  //               cv::Range(keypoint.pt.x - width, keypoint.pt.x + width));

  //     cv::imwrite(results_dir + std::to_string(counter++) + ".png", patch);
  //     outfile_cov << covariance.format(CSVFormat);
  //     outfile_hessian << hessian.format(CSVFormat);
  //   } else {
  //     std::cout << idx << " out of image range" << std::endl;
  //   }
  // }
  outfile_cov.close();
  outfile_hessian.close();
  // PlotFeatures();
}

void BaseFrame::SaveInlierPatchesStructured(
    const pnec::features::KeyPoints host_keypoints,
    const pnec::features::KeyPoints target_keypoints, size_t &counter,
    std::string results_dir) {
  const cv::Mat image = getImage();

  std::ofstream outfile_cov;
  std::ofstream outfile_hessian;
  std::ofstream outfile_kp;

  outfile_kp.open(results_dir + "keypoints.txt", std::ios_base::app);
  outfile_cov.open(results_dir + "covariances.txt", std::ios_base::app);
  outfile_hessian.open(results_dir + "hessians.txt", std::ios_base::app);

  const static Eigen::IOFormat CSVFormat(
      Eigen::FullPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "", "\n");

  int width = 6;
  size_t curr_frame_count = 0;
  std::vector<cv::Mat> cv_patches;
  for (auto const &[id, host_keypoint] : host_keypoints) {
    const auto target_keypoint = target_keypoints.at(id);

    cv::KeyPoint host_point = pnec::features::KeyPointToCV(host_keypoint);
    cv::KeyPoint target_point = pnec::features::KeyPointToCV(target_keypoint);
    bool in_bounds = (target_point.pt.x - width) > 0 &&
                     (target_point.pt.x + width) < (image.size().width - 1) &&
                     (target_point.pt.y - width) > 0 &&
                     (target_point.pt.y + width) < (image.size().height - 1);

    if (in_bounds) {
      if (!valid_covariance(target_keypoint.img_covariance_)) {
        std::cout << "found invalid covariance" << std::endl;
        continue;
      }
      cv::Mat patch = image(
          cv::Range(target_point.pt.y - width, target_point.pt.y + width),
          cv::Range(target_point.pt.x - width, target_point.pt.x + width));
      cv_patches.push_back(patch);

      std::string image_counter = std::to_string(counter++);
      // cv::imwrite(results_dir + image_counter + ".png", patch);
      outfile_kp << id_ << ", " << curr_frame_count << ", " << image_counter
                 << ", " << host_point.pt.x << ", " << host_point.pt.y << ", "
                 << target_point.pt.x << ", " << target_point.pt.y << "\n";
      outfile_cov << id_ << ", " << curr_frame_count << ", " << image_counter
                  << ", " << target_keypoint.img_covariance_.format(CSVFormat);
      outfile_hessian << id_ << ", " << curr_frame_count << ", "
                      << image_counter << ", "
                      << target_keypoint.hessian_.format(CSVFormat);
      curr_frame_count++;
    } else {
      std::cout << id << " out of image range" << std::endl;
    }
  }
  cv::Mat patches_img = createOne(cv_patches, 1, 0);
  cv::imwrite(results_dir + "patches.png", patches_img);
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

void GetInlierKeyPoints(
    pnec::frames::BaseFrame::Ptr host_frame,
    pnec::frames::BaseFrame::Ptr target_frame, pnec::FeatureMatches matches,
    std::vector<int> inliers,
    std::vector<pnec::features::KeyPoint> &inlier_host_frame,
    std::vector<pnec::features::KeyPoint> &inlier_target_frame) {
  inlier_host_frame.clear();
  inlier_host_frame.reserve(inliers.size());
  inlier_target_frame.clear();
  inlier_target_frame.reserve(inliers.size());
  pnec::features::KeyPoints host_keypoints = host_frame->keypoints();
  pnec::features::KeyPoints target_keypoints = target_frame->keypoints();

  for (const auto &inlier : inliers) {
    cv::DMatch match = matches[inlier];
    inlier_host_frame.push_back(host_keypoints[match.queryIdx]);
    inlier_target_frame.push_back(target_keypoints[match.trainIdx]);
  }
}

} // namespace frames
} // namespace pnec
