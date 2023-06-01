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

#include "visualization.h"

#include <math.h>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "common.h"
#include "keypoints.h"
namespace pnec {
namespace visualization {

// https://gist.github.com/pkmital/8a15555d3b29eabaa606
cv::RotatedRect GetErrorEllipse(double chisquare_val, cv::Point2f mean,
                                cv::Mat covmat) {

  // Get the eigenvalues and eigenvectors
  cv::Mat eigenvalues, eigenvectors;
  cv::eigen(covmat, eigenvalues, eigenvectors);

  // Calculate the angle between the largest eigenvector and the x-axis
  double angle =
      std::atan2(eigenvectors.at<double>(0, 1), eigenvectors.at<double>(0, 0));

  // Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
  if (angle < 0)
    angle += 6.28318530718;

  // Conver to degrees instead of radians
  angle = 180 * angle / 3.14159265359;

  // Calculate the size of the minor and major axes
  double halfmajoraxissize =
      chisquare_val * std::sqrt(eigenvalues.at<double>(0));
  double halfminoraxissize =
      chisquare_val * std::sqrt(eigenvalues.at<double>(1));

  // Return the oriented ellipse
  // The -angle is used because OpenCV defines the angle clockwise instead of
  // anti-clockwise
  return cv::RotatedRect(mean, cv::Size2f(halfmajoraxissize, halfminoraxissize),
                         angle);
}

char plotMatches(pnec::frames::BaseFrame::Ptr host_frame,
                 pnec::frames::BaseFrame::Ptr target_frame,
                 pnec::FeatureMatches &matches, std::vector<int> &inliers,
                 pnec::visualization::Options visualization_options,
                 std::string suffix) {

  // get image name
  std::stringstream ss_host;
  ss_host << std::setw(6) << std::setfill('0') << host_frame->id();
  std::stringstream ss_target;
  ss_target << std::setw(6) << std::setfill('0') << target_frame->id();
  std::string image_name = ss_host.str() + "_" + ss_target.str() + suffix;

  // get inlier
  pnec::FeatureMatches inlier_matches;
  for (const auto &inlier : inliers) {
    inlier_matches.push_back(matches[inlier]);
  }
  BOOST_LOG_TRIVIAL(info) << "Number of matches " << matches.size();
  BOOST_LOG_TRIVIAL(info) << "Number of inlier " << inlier_matches.size();

  cv::Mat host_image = host_frame->getImage();
  cv::Mat target_image = target_frame->getImage();

  if (visualization_options.keypoints == pnec::visualization::Options::ALL) {
    std::vector<cv::KeyPoint> cv_host_keypoints;
    std::vector<cv::KeyPoint> cv_target_keypoints;

    for (auto const &[id, kp] : host_frame->keypoints()) {
      cv_host_keypoints.push_back(pnec::features::KeyPointToCV(kp));
    }
    for (auto const &[id, kp] : target_frame->keypoints()) {
      cv_target_keypoints.push_back(pnec::features::KeyPointToCV(kp));
    }

    // draw all keypoints
    cv::drawKeypoints(host_image, cv_host_keypoints, host_image);
    cv::drawKeypoints(target_image, cv_target_keypoints, target_image);

    if (visualization_options.covariances ==
        pnec::visualization::Options::ALL) {
      // draw all covariances
      for (auto const &[id, kp] : target_frame->keypoints()) {
        cv::Mat cv_cov;
        cv::eigen2cv(kp.img_covariance_, cv_cov);

        cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
            visualization_options.cov_scaling,
            pnec::features::KeyPointToCV(target_frame->keypoints()[id]).pt,
            cv_cov);
        cv::ellipse(target_image, ellipse,
                    visualization_options.covariance_color,
                    visualization_options.cov_thickness);
      }
    }
  }

  if (visualization_options.keypoints >=
      pnec::visualization::Options::TRACKED) {
    cv::Mat full_image;
    // draw inlier keypoints
    std::vector<cv::KeyPoint> cv_host_keypoints;
    std::vector<cv::KeyPoint> cv_target_keypoints;

    std::vector<cv::DMatch> cv_matches;
    size_t idx = 0;
    for (const auto &match : matches) {
      cv_host_keypoints.push_back(pnec::features::KeyPointToCV(
          host_frame->keypoints()[match.queryIdx]));
      cv_target_keypoints.push_back(pnec::features::KeyPointToCV(
          target_frame->keypoints()[match.trainIdx]));
      cv_matches.push_back(cv::DMatch(idx, idx, 0));
      idx++;
    }

    cv::drawMatches(host_image, cv_host_keypoints, target_image,
                    cv_target_keypoints, cv_matches, full_image,
                    visualization_options.tracked_color);

    // split up images again
    host_image =
        full_image(cv::Rect(0, 0, full_image.cols / 2, full_image.rows));
    target_image = full_image(
        cv::Rect(full_image.cols / 2, 0, full_image.cols / 2, full_image.rows));

    if (visualization_options.covariances >=
        pnec::visualization::Options::TRACKED) {
      // draw inlier covariances
      for (const auto &match : matches) {
        cv::Mat cv_cov;
        cv::eigen2cv(target_frame->keypoints()[match.trainIdx].img_covariance_,
                     cv_cov);
        cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
            visualization_options.cov_scaling,
            pnec::features::KeyPointToCV(
                target_frame->keypoints()[match.trainIdx])
                .pt,
            cv_cov);
        cv::ellipse(target_image, ellipse,
                    visualization_options.covariance_color,
                    visualization_options.cov_thickness);
      }
    }
  }

  if (visualization_options.keypoints >= pnec::visualization::Options::INLIER) {
    cv::Mat full_image;
    // draw inlier keypoints
    std::vector<cv::KeyPoint> cv_host_keypoints;
    std::vector<cv::KeyPoint> cv_target_keypoints;

    std::vector<cv::DMatch> cv_matches;
    size_t idx = 0;
    for (const auto &match : inlier_matches) {
      cv_host_keypoints.push_back(pnec::features::KeyPointToCV(
          host_frame->keypoints()[match.queryIdx]));
      cv_target_keypoints.push_back(pnec::features::KeyPointToCV(
          target_frame->keypoints()[match.trainIdx]));
      cv_matches.push_back(cv::DMatch(idx, idx, 0));
      idx++;
    }

    cv::drawMatches(host_image, cv_host_keypoints, target_image,
                    cv_target_keypoints, cv_matches, full_image,
                    visualization_options.inlier_color);

    // split up images again
    host_image =
        full_image(cv::Rect(0, 0, full_image.cols / 2, full_image.rows));
    target_image = full_image(
        cv::Rect(full_image.cols / 2, 0, full_image.cols / 2, full_image.rows));

    if (visualization_options.covariances >=
        pnec::visualization::Options::INLIER) {
      // draw inlier covariances
      for (const auto &match : inlier_matches) {
        cv::Mat cv_cov;
        cv::eigen2cv(target_frame->keypoints()[match.trainIdx].img_covariance_,
                     cv_cov);
        cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
            visualization_options.cov_scaling,
            pnec::features::KeyPointToCV(
                target_frame->keypoints()[match.trainIdx])
                .pt,
            cv_cov);
        cv::ellipse(target_image, ellipse,
                    visualization_options.covariance_color,
                    visualization_options.cov_thickness);
      }
    }
  }

  // concatenate images
  cv::Mat image;
  cv::hconcat(host_image, target_image, image);

  double s = 1.0f;
  cv::Size size(s * image.cols, s * image.rows);
  resize(image, image, size);

  cv::imwrite(visualization_options.base_folder + image_name + ".png", image);
  return 's';
  // cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
  // cv::imshow(image_name, image);
  // char key = cv::waitKey(0);
  // if (key == 's') {
  //   cv::imwrite(visualization_options.base_folder + image_name + ".png",
  //   image);
  // }
  // cv::destroyWindow(image_name);
  // return key;
}

char plotMatches(pnec::frames::BaseFrame::Ptr host_frame, cv::Mat &host_image,
                 pnec::frames::BaseFrame::Ptr target_frame,
                 cv::Mat &target_image, pnec::FeatureMatches &matches,
                 std::vector<int> &inliers,
                 pnec::visualization::Options visualization_options,
                 std::string suffix) {

  cv::imwrite(visualization_options.base_folder + "host_img.png", host_image);
  cv::imwrite(visualization_options.base_folder + "target_img.png",
              target_image);

  // get image name
  std::stringstream ss_host;
  ss_host << std::setw(6) << std::setfill('0') << host_frame->id();
  std::stringstream ss_target;
  ss_target << std::setw(6) << std::setfill('0') << target_frame->id();
  std::string image_name = ss_host.str() + "_" + ss_target.str() + suffix;

  // get inlier
  pnec::FeatureMatches inlier_matches;
  for (const auto &inlier : inliers) {
    inlier_matches.push_back(matches[inlier]);
  }
  BOOST_LOG_TRIVIAL(info) << "Number of matches " << matches.size();
  BOOST_LOG_TRIVIAL(info) << "Number of inlier " << inlier_matches.size();

  if (visualization_options.keypoints == pnec::visualization::Options::ALL) {
    std::vector<cv::KeyPoint> cv_host_keypoints;
    std::vector<cv::KeyPoint> cv_target_keypoints;

    for (auto const &[id, kp] : host_frame->keypoints()) {
      cv_host_keypoints.push_back(pnec::features::KeyPointToCV(kp));
    }
    for (auto const &[id, kp] : target_frame->keypoints()) {
      cv_target_keypoints.push_back(pnec::features::KeyPointToCV(kp));
    }

    // draw all keypoints
    cv::drawKeypoints(host_image, cv_host_keypoints, host_image);
    cv::drawKeypoints(target_image, cv_target_keypoints, target_image);

    if (visualization_options.covariances ==
        pnec::visualization::Options::ALL) {
      // draw all covariances
      for (auto const &[id, kp] : target_frame->keypoints()) {
        cv::Mat cv_cov;
        cv::eigen2cv(kp.img_covariance_, cv_cov);

        cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
            visualization_options.cov_scaling,
            pnec::features::KeyPointToCV(target_frame->keypoints()[id]).pt,
            cv_cov);
        cv::ellipse(target_image, ellipse,
                    visualization_options.covariance_color,
                    visualization_options.cov_thickness);
      }
    }
  }

  if (visualization_options.keypoints >=
      pnec::visualization::Options::TRACKED) {
    cv::Mat full_image;
    // draw inlier keypoints
    std::vector<cv::KeyPoint> cv_host_keypoints;
    std::vector<cv::KeyPoint> cv_target_keypoints;

    std::vector<cv::DMatch> cv_matches;
    size_t idx = 0;
    for (const auto &match : matches) {
      cv_host_keypoints.push_back(pnec::features::KeyPointToCV(
          host_frame->keypoints()[match.queryIdx]));
      cv_target_keypoints.push_back(pnec::features::KeyPointToCV(
          target_frame->keypoints()[match.trainIdx]));
      cv_matches.push_back(cv::DMatch(idx, idx, 0));
      idx++;
    }

    cv::drawMatches(host_image, cv_host_keypoints, target_image,
                    cv_target_keypoints, cv_matches, full_image,
                    visualization_options.tracked_color);

    // split up images again
    host_image =
        full_image(cv::Rect(0, 0, full_image.cols / 2, full_image.rows));
    target_image = full_image(
        cv::Rect(full_image.cols / 2, 0, full_image.cols / 2, full_image.rows));

    if (visualization_options.covariances >=
        pnec::visualization::Options::TRACKED) {
      // draw inlier covariances
      for (const auto &match : matches) {
        cv::Mat cv_cov;
        cv::eigen2cv(target_frame->keypoints()[match.trainIdx].img_covariance_,
                     cv_cov);
        cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
            visualization_options.cov_scaling,
            pnec::features::KeyPointToCV(
                target_frame->keypoints()[match.trainIdx])
                .pt,
            cv_cov);
        cv::ellipse(target_image, ellipse,
                    visualization_options.covariance_color,
                    visualization_options.cov_thickness);
      }
    }
  }

  if (visualization_options.keypoints >= pnec::visualization::Options::INLIER) {
    cv::Mat full_image;
    // draw inlier keypoints
    std::vector<cv::KeyPoint> cv_host_keypoints;
    std::vector<cv::KeyPoint> cv_target_keypoints;

    std::vector<cv::DMatch> cv_matches;
    size_t idx = 0;
    for (const auto &match : inlier_matches) {
      cv_host_keypoints.push_back(pnec::features::KeyPointToCV(
          host_frame->keypoints()[match.queryIdx]));
      cv_target_keypoints.push_back(pnec::features::KeyPointToCV(
          target_frame->keypoints()[match.trainIdx]));
      cv_matches.push_back(cv::DMatch(idx, idx, 0));
      idx++;
    }

    cv::drawMatches(host_image, cv_host_keypoints, target_image,
                    cv_target_keypoints, cv_matches, full_image,
                    visualization_options.inlier_color);

    // split up images again
    host_image =
        full_image(cv::Rect(0, 0, full_image.cols / 2, full_image.rows));
    target_image = full_image(
        cv::Rect(full_image.cols / 2, 0, full_image.cols / 2, full_image.rows));

    if (visualization_options.covariances >=
        pnec::visualization::Options::INLIER) {
      // draw inlier covariances
      for (const auto &match : inlier_matches) {
        cv::Mat cv_cov;
        cv::eigen2cv(target_frame->keypoints()[match.trainIdx].img_covariance_,
                     cv_cov);
        cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
            visualization_options.cov_scaling,
            pnec::features::KeyPointToCV(
                target_frame->keypoints()[match.trainIdx])
                .pt,
            cv_cov);
        cv::ellipse(target_image, ellipse,
                    visualization_options.covariance_color,
                    visualization_options.cov_thickness);
      }
    }
  }

  // concatenate images
  cv::Mat image;
  cv::hconcat(host_image, target_image, image);

  double s = 1.0f;
  cv::Size size(s * image.cols, s * image.rows);
  resize(image, image, size);

  cv::imwrite(visualization_options.base_folder + image_name + ".png", image);
  return 's';
  // cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
  // cv::imshow(image_name, image);
  // char key = cv::waitKey(0);
  // if (key == 's') {
  //   cv::imwrite(visualization_options.base_folder + image_name + ".png",
  //   image);
  // }
  // cv::destroyWindow(image_name);
  // return key;
}
} // namespace visualization
} // namespace pnec