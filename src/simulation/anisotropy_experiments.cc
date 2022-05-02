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

#include "anisotropy_experiments.h"

#include <boost/filesystem.hpp>

#include "sim_common.h"

namespace simulation {
namespace experiments {
void AnisotropyExperiments::GenerateExperiments(std::string base_folder,
                                                size_t num_experiments,
                                                size_t num_points,
                                                std::vector<double> levels) {
  for (const auto &level : levels) {
    std::string level_folder = base_folder + "/" + std::to_string(level);

    experiments_.clear();
    experiments_.reserve(num_experiments);

    for (int i = 0; i < num_experiments; i++) {
      Sophus::SE3d pose_1, pose_2;
      SamplePoses(0.5, translation_, pose_1, pose_2);

      std::vector<Eigen::Vector3d> points_1, points_2;
      SamplePoints(num_points, camera_model_, pose_1, pose_2, points_1,
                   points_2);

      std::vector<Eigen::Matrix3d> covariances;
      SampleCovariances(level, num_points, covariances);

      std::vector<Eigen::Vector3d> points_with_noise;
      std::vector<Eigen::Matrix3d> image_plane_covariances_1,
          image_plane_covariances_2;
      AddNoise(points_2, covariances, points_with_noise,
               image_plane_covariances_2);
      for (int j = 0; j < num_points; j++) {
        image_plane_covariances_1.push_back(-1.0 * Eigen::Matrix3d::Identity());
      }

      experiments_.push_back(SingleExperiment(
          pose_1, pose_2, points_1, points_with_noise,
          image_plane_covariances_1, image_plane_covariances_2));
    }
    SaveExperiments(level_folder);
  }
}

void AnisotropyExperiments::SampleCovariances(
    double anisotropy_level, size_t num_covs,
    std::vector<Eigen::Matrix3d> &covariances) {
  double noise_level = 1.0;
  double inhomgeniety_offset = 0.5;

  covariances.clear();
  covariances.reserve(num_covs);

  for (size_t i = 0; i < num_covs; i++) {
    double alpha = uniform01_(generator_) * M_PI;
    double scale = (uniform01_(generator_) + inhomgeniety_offset);

    covariances.push_back(
        CovFromParameters(noise_level, scale, alpha, anisotropy_level));
  }
}

void AnisotropyExperiments::AddNoise(
    const std::vector<Eigen::Vector3d> &points,
    const std::vector<Eigen::Matrix3d> &covariances,
    std::vector<Eigen::Vector3d> &points_with_noise,
    std::vector<Eigen::Matrix3d> &image_plane_covariances) {
  for (size_t i = 0; i < points.size(); i++) {
    Eigen ::Vector3d point = points[i];
    Eigen::Matrix3d cov = covariances[i];
    Eigen::Matrix3d rotation;
    switch (camera_model_) {
    case pnec::common::CameraModel::Pinhole:
      rotation = Eigen::Matrix3d::Identity();
      break;
    case pnec::common::CameraModel::Omnidirectional:
      rotation = pnec::common::RotationBetweenPoints(
          Eigen::Vector3d(0.0, 0.0, 1.0), point.normalized());
      break;
    }

    Eigen::Matrix3d rot_cov = rotation * cov * rotation.transpose();

    Eigen::Vector3d uniform_noise =
        Eigen::Vector3d(distribution_(normal_generator_),
                        distribution_(normal_generator_), 0.0);
    Eigen::Matrix2d C = cov.topLeftCorner(2, 2).llt().matrixL();
    Eigen::Matrix3d C3D = Eigen::Matrix3d::Zero();
    C3D.topLeftCorner(2, 2) = C;
    Eigen::Vector3d noise = rotation * C3D * uniform_noise;

    points_with_noise.push_back(point + noise);
    image_plane_covariances.push_back(rot_cov);
  }
}

} // namespace experiments
} // namespace simulation