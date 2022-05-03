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

#include "experiments.h"

#include <boost/filesystem.hpp>
#include <cmath>
namespace simulation {
namespace experiments {

void BaseExperiments::SamplePoses(double max_euler_angle, bool translation,
                                  Sophus::SE3d &pose_1, Sophus::SE3d &pose_2) {
  pose_1 = Sophus::SE3d();

  double roll = (uniform01_(generator_) * 2.0 - 1.0) * max_euler_angle,
         pitch = (uniform01_(generator_) * 2.0 - 1.0) * max_euler_angle,
         yaw = (uniform01_(generator_) * 2.0 - 1.0) * max_euler_angle;
  Eigen::Quaterniond q;
  q = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());

  Eigen::Vector3d rel_translation;
  if (!translation_) {
    rel_translation = Eigen::Vector3d(0.0, 0.0, 0.0);
  } else {
    double theta = 2 * M_PI * uniform01_(generator_);
    double phi = std::acos(1.0 - 2.0 * uniform01_(generator_));
    double magnitude = 2 * uniform01_(generator_);

    rel_translation =
        magnitude * Eigen::Vector3d(std::sin(phi) * std::cos(theta),
                                    std::sin(phi) * std::sin(theta),
                                    std::cos(phi));
  }

  pose_2 = pose_1 * Sophus::SE3d(q.toRotationMatrix(), rel_translation);
}

void BaseExperiments::SamplePoints(size_t num_points,
                                   pnec::common::CameraModel camera_model,
                                   Sophus::SE3d &pose_1, Sophus::SE3d &pose_2,
                                   std::vector<Eigen::Vector3d> &points_1,
                                   std::vector<Eigen::Vector3d> &points_2) {
  Eigen::Vector3d point_mean(0.0, 0.0, 0.0);
  double focal_length = 800;

  points_1.clear();
  points_1.reserve(num_points);
  points_2.clear();
  points_2.reserve(num_points);
  switch (camera_model_) {
  case pnec::common::CameraModel::Pinhole:
    for (size_t i = 0; i < num_points; i++) {
      double a = 0.4;
      double width = 1.0;
      double height = 1.5;
      double max_depth = 5.0;
      double depth = ((1 - a) * uniform01_(generator_) + a) * max_depth;

      Eigen::Vector3d point =
          pose_1 *
          Eigen::Vector3d((uniform01_(generator_) - 0.5) * width,
                          (uniform01_(generator_) - 0.5) * height, 1.0) *
          depth;

      Eigen::Vector3d point_1 = (pose_1.inverse() * point).normalized();
      point_1 = point_1 / point_1(2) * focal_length;
      Eigen::Vector3d point_2 = (pose_2.inverse() * point).normalized();
      point_2 = point_2 / point_2(2) * focal_length;

      points_1.push_back(point_1);
      points_2.push_back(point_2);
    }
    break;
  case pnec::common::CameraModel::Omnidirectional:
    for (size_t i = 0; i < num_points; i++) {
      double theta = 2 * M_PI * uniform01_(generator_);
      double phi = std::acos(1.0 - 2.0 * uniform01_(generator_));
      double magnitude = 4.0 * uniform01_(generator_) + 4.0;

      Eigen::Vector3d point =
          magnitude * Eigen::Vector3d(sin(phi) * cos(theta),
                                      sin(phi) * sin(theta), cos(phi)) +
          point_mean;

      Eigen::Vector3d point_1 =
          (pose_1.inverse() * point).normalized() * focal_length;
      Eigen::Vector3d point_2 =
          (pose_2.inverse() * point).normalized() * focal_length;

      points_1.push_back(point_1);
      points_2.push_back(point_2);
    }
    break;
  }
}

void BaseExperiments::SaveExperiments(std::string base_folder) {
  if (!boost::filesystem::is_directory(base_folder)) {
    boost::filesystem::create_directory(base_folder);
  }

  const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                         Eigen::DontAlignCols, ",", ",");

  std::ofstream outFilePoses1(base_folder + "/poses_1.csv");
  std::ofstream outFilePoses2(base_folder + "/poses_2.csv");
  std::ofstream outFilePoints1(base_folder + "/points_1.csv");
  std::ofstream outFilePoints2(base_folder + "/points_2.csv");
  std::ofstream outFileCovs1(base_folder + "/covs_1.csv");
  std::ofstream outFileCovs2(base_folder + "/covs_2.csv");

  for (const auto &experiment : experiments_) {
    outFilePoses1 << experiment.pose_1.unit_quaternion().coeffs().format(
                         CSVFormat)
                  << "," << experiment.pose_1.translation().format(CSVFormat)
                  << "\n";
    outFilePoses2 << experiment.pose_2.unit_quaternion().coeffs().format(
                         CSVFormat)
                  << "," << experiment.pose_2.translation().format(CSVFormat)
                  << "\n";
    for (const auto &point : experiment.points_1) {
      outFilePoints1 << point.format(CSVFormat) << ",";
    }
    outFilePoints1 << "\n";
    for (const auto &point : experiment.points_2) {
      outFilePoints2 << point.format(CSVFormat) << ",";
    }
    outFilePoints2 << "\n";
    for (const auto &cov : experiment.covs_1) {
      outFileCovs1 << cov.format(CSVFormat) << ",";
    }
    outFileCovs1 << "\n";
    for (const auto &cov : experiment.covs_2) {
      outFileCovs2 << cov.format(CSVFormat) << ",";
    }
    outFileCovs2 << "\n";
  }
}

Eigen::Matrix3d BaseExperiments::CovFromParameters(double noise_scale,
                                                   double scale, double alpha,
                                                   double beta) {
  Eigen::Matrix2d rot;
  rot << std::cos(alpha), -std::sin(alpha), std::sin(alpha), std::cos(alpha);
  Eigen::Matrix2d cov2d_unrot;
  cov2d_unrot << beta, 0, 0, 1.0 - beta;
  Eigen::Matrix2d cov2d = scale * rot * cov2d_unrot * rot.transpose();
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  cov.topLeftCorner(2, 2) = noise_scale * cov2d;
  return cov;
}

} // namespace experiments
} // namespace simulation