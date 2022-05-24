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
                 std::string suffix) {
  std::stringstream ss_host;
  ss_host << std::setw(6) << std::setfill('0') << host_frame->id();
  std::stringstream ss_target;
  ss_target << std::setw(6) << std::setfill('0') << target_frame->id();
  std::string image_name = ss_host.str() + "_" + ss_target.str() + suffix;

  std::vector<pnec::features::KeyPoint> host_keypoints;
  std::vector<pnec::features::KeyPoint> target_keypoints;

  pnec::frames::GetInlierKeyPoints(host_frame, target_frame, matches, inliers,
                                   host_keypoints, target_keypoints);

  std::vector<cv::KeyPoint> cv_host_keypoints;
  std::vector<cv::KeyPoint> cv_target_keypoints;
  std::vector<cv::Mat> covariances;
  std::vector<cv::DMatch> cv_matches;

  int skip = 3;
  int counter = 0;
  for (size_t idx = 0; idx < inliers.size(); idx++) {
    if (matches[inliers[idx]].queryIdx % skip != 0) {
      continue;
    }
    // if (target_keypoints[idx].img_covariance_(0, 0) <= 0) {
    //   continue;
    // }
    cv_host_keypoints.push_back(
        pnec::features::KeyPointToCV(host_keypoints[idx]));
    cv_target_keypoints.push_back(
        pnec::features::KeyPointToCV(target_keypoints[idx]));

    cv::Mat cov;
    cv::eigen2cv(target_keypoints[idx].img_covariance_, cov);
    covariances.push_back(cov);
    cv_matches.push_back(cv::DMatch(counter, counter, 0.0));
    counter++;
  }
  BOOST_LOG_TRIVIAL(info) << "number of matches: " << cv_matches.size();

  cv::Mat image;
  cv::drawMatches(host_frame->getImage(), cv_host_keypoints,
                  target_frame->getImage(), cv_target_keypoints, cv_matches,
                  image);

  // draw covariances
  BOOST_LOG_TRIVIAL(info) << "number of covariances: " << covariances.size();
  for (size_t idx = 0; idx < covariances.size(); idx++) {
    cv::Mat covariance = covariances[idx];
    cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
        2.4477 * 10,
        cv_target_keypoints[idx].pt + cv::Point2f(image.cols / 2.0, 0.0),
        covariance);
    cv::ellipse(image, ellipse, cv::Scalar(255, 0, 0), 2);
  }

  double s = 1.0f;
  cv::Size size(s * image.cols, s * image.rows);
  resize(image, image, size);
  cv::namedWindow(image_name, cv::WINDOW_AUTOSIZE);
  cv::imshow(image_name, image);
  char key = cv::waitKey(10);
  // if (key == 's') {
  cv::imwrite("/storage/user/muhled/outputs/pnec/keypoint_visualization/" +
                  image_name + ".png",
              image);
  // }
  cv::destroyWindow(image_name);
  return key;
}

// char plotMatches(pnec::frames::BaseFrame::Ptr host_frame,
//                  pnec::frames::BaseFrame::Ptr target_frame,
//                  pnec::FeatureMatches &id_matches, std::vector<int> &inliers,
//                  std::string suffix) {
//   std::string window_name = std::to_string(target_frame->id()) + "_" +
//                             std::to_string(host_frame->id()) + suffix;

//   auto im1 = host_frame->getImage();
//   auto im2 = target_frame->getImage();
//   std::vector<cv::KeyPoint> host_kps;
//   std::vector<cv::KeyPoint> target_kps;
//   std::vector<cv::DMatch> idx_matches;
//   std::vector<cv::DMatch> inlier_idx_matches;

//   for (auto const &[id, keypoint] : host_frame->keypoints()) {
//     host_kps.push_back(pnec::features::KeyPointToCV(keypoint));
//   }
//   for (auto const &[id, keypoint] : target_frame->keypoints()) {
//     target_kps.push_back(pnec::features::KeyPointToCV(keypoint));
//   }

//   for (const auto match : id_matches) {
//     size_t host_idx =
//         std::distance(host_frame->keypoints().begin(),
//                       host_frame->keypoints().find(match.queryIdx));
//     size_t target_idx =
//         std::distance(target_frame->keypoints().begin(),
//                       target_frame->keypoints().find(match.trainIdx));
//     idx_matches.push_back(cv::DMatch(host_idx, target_idx, 0.0));
//   }

//   std::vector<Eigen::Matrix2d> covariances;
//   for (auto const &[id, keypoint] : target_frame->keypoints()) {
//     covariances.push_back(keypoint.img_covariance_);
//   }

//   pnec::FeatureMatches inlier_matches;
//   for (const auto &inlier : inliers) {
//     inlier_matches.push_back(id_matches[inlier]);
//   }

//   cv::Mat matches_img;
//   cv::drawMatches(im1, host_kps, im2, target_kps, id_matches, matches_img);

//   for (const auto &match : id_matches) {
//     if (covariances[match.trainIdx](0, 0) >= 0.0) {
//       cv::Mat cv_cov;
//       cv::eigen2cv(covariances[match.trainIdx], cv_cov);
//       cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
//           2.4477,
//           target_kps[match.trainIdx].pt +
//               cv::Point2f(matches_img.cols / 2.0, 0.0),
//           cv_cov);
//       cv::ellipse(matches_img, ellipse, cv::Scalar(255, 0, 0), 1);
//     }
//   }

//   cv::Mat inlier_img;
//   cv::drawMatches(im1, host_kps, im2, target_kps, inlier_matches,
//   inlier_img);

//   for (const auto &inlier : inlier_matches) {
//     if (covariances[inlier.trainIdx](0, 0) >= 0.0) {
//       cv::Mat cv_cov;
//       cv::eigen2cv(covariances[inlier.trainIdx], cv_cov);
//       cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
//           2.4477,
//           target_kps[inlier.trainIdx].pt +
//               cv::Point2f(inlier_img.cols / 2.0, 0.0),
//           cv_cov);
//       cv::ellipse(inlier_img, ellipse, cv::Scalar(255, 0, 0), 1);
//     }
//   }

//   cv::Mat img;
//   cv::vconcat(matches_img, inlier_img, img);

//   double s = 1.0f;
//   cv::Size size(s * img.cols, s * img.rows);
//   resize(img, img, size);
//   cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
//   cv::imshow(window_name, img);
//   char key = cv::waitKey(0);
//   if (key == 's') {
//     cv::imwrite("default location" + window_name + ".png", img);
//   }
//   cv::destroyAllWindows();
//   cv::destroyWindow(window_name);
//   return key;
// }

// char plotCovariancess(pnec::frames::BaseFrame::Ptr curr_frame,
//                       pnec::FeatureMatches &matches, std::vector<int>
//                       &inliers, std::string suffix) {
//   std::string window_name = std::to_string(curr_frame->id()) + suffix;

//   std::string path = curr_frame->getPath();
//   cv::Mat im2;

//   im2 = cv::imread(path, cv::IMREAD_COLOR); // Read the file
//   // auto im2 = curr_frame->getImage();
//   auto kps2 = curr_frame->keypoints();
//   std::vector<Eigen::Matrix2d> covariances = curr_frame->covariances();

//   pnec::FeatureMatches inlier_matches;
//   std::vector<cv::KeyPoint> inlier_kp;

//   // for (const auto &inlier : inliers) {
//   for (size_t i = 0; i < inliers.size(); i += 3) {
//     const auto inlier = inliers[i];
//     if ((kps2[matches[inlier].trainIdx].pt.x <
//          (int)im2.size().width / 4 + 20) ||
//         (kps2[matches[inlier].trainIdx].pt.x >
//          (int)3 * im2.size().width / 4 - 20) ||
//         (kps2[matches[inlier].trainIdx].pt.y <
//          (int)im2.size().height / 4 + 10) ||
//         (kps2[matches[inlier].trainIdx].pt.y >
//          (int)3 * im2.size().height / 4 - 10)) {
//       continue;
//     }
//     inlier_matches.push_back(matches[inlier]);
//     inlier_kp.push_back(kps2[matches[inlier].trainIdx]);
//   }

//   cv::Mat img;
//   cv::drawKeypoints(im2, inlier_kp, img, cv::Scalar(51, 0, 255));
//   for (const auto &inlier : inlier_matches) {
//     if (covariances[inlier.trainIdx](0, 0) >= 0.0) {
//       cv::Mat cv_cov;
//       cv::eigen2cv(covariances[inlier.trainIdx], cv_cov);
//       cv::RotatedRect ellipse = pnec::visualization::GetErrorEllipse(
//           80.0, kps2[inlier.trainIdx].pt, cv_cov);
//       cv::ellipse(img, ellipse, cv::Scalar(51, 0, 255), 2);
//     }
//   }

//   double s = 1.0f;
//   cv::Size size(s * img.cols, s * img.rows);
//   resize(img, img, size);
//   cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
//   cv::imshow(window_name, img);
//   char key = cv::waitKey(0);
//   if (key == 's') {
//     cv::imwrite("default location" + window_name + ".png", img);
//   }
//   cv::destroyAllWindows();
//   cv::destroyWindow(window_name);
//   return key;
// }
} // namespace visualization
} // namespace pnec