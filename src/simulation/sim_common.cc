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

#include "sim_common.h"

#include "common.h"

namespace simulation {
namespace common {

double clip(double n, double lower, double upper) {
  return std::max(lower, std::min(n, upper));
}

Results::Results() {}
Results::Results(size_t num_experiments) {
  rotational_error_.reserve(num_experiments);
  translational_error_.reserve(num_experiments);
  costs_.reserve(num_experiments);
}
Results::~Results() {}

void Results::AddResults(const Sophus::SE3d &solution,
                         const Sophus::SE3d &ground_truth,
                         const opengv::bearingVectors_t &bvs_1,
                         const opengv::bearingVectors_t &bvs_2,
                         const std::vector<Eigen::Matrix3d> &covs) {
  rotational_error_.push_back(
      pnec::common::RotationalDifference(ground_truth.so3(), solution.so3()));
  translational_error_.push_back(pnec::common::TranslationalDifference(
      ground_truth.translation(), solution.translation(), true));
  costs_.push_back(pnec::common::CostFunction(bvs_1, bvs_2, covs, solution));
}

std::istream &operator>>(std::istream &str, CSVRow &data) {
  data.readNextRow(str);
  return str;
}

void GetFeatures(const opengv::bearingVectors_t &points_1,
                 const opengv::bearingVectors_t &points_2,
                 const std::vector<Eigen::Matrix3d> &covs,
                 opengv::bearingVectors_t &bvs1, opengv::bearingVectors_t &bvs2,
                 std::vector<Eigen::Matrix3d> &transformed_covs,
                 const pnec::common::CameraModel camera_model,
                 const pnec::common::NoiseFrame noise_frame) {
  const size_t n_matches = covs.size();
  bvs1.clear();
  bvs1.reserve(n_matches);
  bvs2.clear();
  bvs2.reserve(n_matches);
  transformed_covs.clear();
  transformed_covs.reserve(n_matches);

  for (size_t i = 0; i < n_matches; i++) {
    Eigen::Vector3d point_1 = points_1[i];
    bvs1.push_back(point_1.normalized());

    Eigen::Vector3d point_2 = points_2[i];
    bvs2.push_back(point_2.normalized());

    Eigen::Matrix3d K_inv;
    K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
    switch (noise_frame) {
    case pnec::common::Host:
      transformed_covs.push_back(pnec::common::UnscentedTransform(
          point_1, covs[i], K_inv, 1.0, camera_model));
      break;
    case pnec::common::Target:
      transformed_covs.push_back(pnec::common::UnscentedTransform(
          point_2, covs[i], K_inv, 1.0, camera_model));
      break;
    }
  }
}

void ReadExperiments(std::string folder,
                     std::vector<LoadedExperiment> &experiments,
                     std::mt19937 &generator,
                     std::uniform_real_distribution<double> &uniform01,
                     pnec::common::CameraModel camera_model,
                     pnec::common::NoiseFrame noise_frame,
                     double init_scaling) {

  // rel_poses
  std::vector<Sophus::SE3d> poses_1;
  std::vector<Sophus::SE3d> poses_2;
  std::vector<Sophus::SE3d> rel_poses;
  std::ifstream poses_1_file(folder + "/poses_1.csv");
  for (auto &row : CSVRange(poses_1_file)) {
    Eigen::Quaterniond orientation(row[3], row[0], row[1], row[2]);
    Eigen::Vector3d translation(row[4], row[5], row[6]);
    poses_1.push_back(Sophus::SE3d(orientation, translation));
  }
  std::ifstream poses_2_file(folder + "/poses_2.csv");
  for (auto &row : CSVRange(poses_2_file)) {
    Eigen::Quaterniond orientation(row[3], row[0], row[1], row[2]);
    Eigen::Vector3d translation(row[4], row[5], row[6]);
    poses_2.push_back(Sophus::SE3d(orientation, translation));
  }
  rel_poses.clear();
  for (size_t i = 0; i < poses_1.size(); i++) {
    rel_poses.push_back(poses_1[i].inverse() * poses_2[i]);
  }

  // points_1
  std::vector<opengv::bearingVectors_t> points_1_vec;
  std::ifstream points_1_file(folder + "/points_1.csv");
  for (auto &row : CSVRange(points_1_file)) {
    opengv::bearingVectors_t points;
    for (size_t i = 0; i < row.size() / 3; i++) {
      points.push_back(
          Eigen::Vector3d(row[3 * i], row[3 * i + 1], row[3 * i + 2]));
    }
    points_1_vec.push_back(points);
  }

  // cov_1
  std::vector<std::vector<Eigen::Matrix3d>> covs_1_vec;
  std::ifstream cov_1_file(folder + "/covs_1.csv");
  for (auto &row : CSVRange(cov_1_file)) {
    std::vector<Eigen::Matrix3d> covs;
    for (size_t i = 0; i < row.size() / 9; i++) {
      Eigen::Matrix3d cov;
      cov << row[3 * i], row[3 * i + 1], row[3 * i + 2], row[3 * i + 3],
          row[3 * i + 4], row[3 * i + 5], row[3 * i + 6], row[3 * i + 7],
          row[3 * i + 8];
      covs.push_back(cov);
    }
    covs_1_vec.push_back(covs);
  }

  // points_2
  std::vector<opengv::bearingVectors_t> points_2_vec;
  std::ifstream points_2_file(folder + "/points_2.csv");
  for (auto &row : CSVRange(points_2_file)) {
    opengv::bearingVectors_t points;
    for (size_t i = 0; i < row.size() / 3; i++) {
      points.push_back(
          Eigen::Vector3d(row[3 * i], row[3 * i + 1], row[3 * i + 2]));
    }
    points_2_vec.push_back(points);
  }

  // cov_2
  std::vector<std::vector<Eigen::Matrix3d>> covs_2_vec;
  std::ifstream cov_2_file(folder + "/covs_2.csv");
  for (auto &row : CSVRange(cov_2_file)) {
    std::vector<Eigen::Matrix3d> covs;
    for (size_t i = 0; i < row.size() / 9; i++) {
      Eigen::Matrix3d cov;
      cov << row[9 * i], row[9 * i + 1], row[9 * i + 2], row[9 * i + 3],
          row[9 * i + 4], row[9 * i + 5], row[9 * i + 6], row[9 * i + 7],
          row[9 * i + 8];
      covs.push_back(cov);
    }
    covs_2_vec.push_back(covs);
  }

  experiments.clear();
  for (int i = 0; i < rel_poses.size(); i++) {
    opengv::bearingVectors_t points_1 = points_1_vec[i];
    std::vector<Eigen::Matrix3d> covs_1 = covs_1_vec[i];
    opengv::bearingVectors_t points_2 = points_2_vec[i];
    std::vector<Eigen::Matrix3d> covs_2 = covs_2_vec[i];

    opengv::bearingVectors_t bvs_1;
    opengv::bearingVectors_t bvs_2;
    std::vector<Eigen::Matrix3d> transformed_covs_2;
    GetFeatures(points_1, points_2, covs_2, bvs_1, bvs_2, transformed_covs_2,
                camera_model, noise_frame);

    // Random starting position near to the gt
    double theta = 2 * M_PI * uniform01(generator);
    double phi = acos(1 - 2 * uniform01(generator));
    Eigen::Vector3d rotation_vec(std::sin(phi) * std::cos(theta),
                                 std::sin(phi) * std::sin(theta),
                                 std::cos(phi));
    bool uniform = true;
    double angle;
    double max_angle = 0.01 * init_scaling; // in radian
    if (uniform) {
      angle = std::sqrt(uniform01(generator)) * max_angle;
    } else {
      angle = uniform01(generator) * max_angle;
    }
    Eigen::Quaterniond orientation_vec(Eigen::AngleAxisd(angle, rotation_vec));
    double max_distance = 0.01 * init_scaling;
    theta = 2 * M_PI * uniform01(generator);
    phi = acos(1 - 2 * uniform01(generator));
    double magnitude = std::sqrt(uniform01(generator)) * max_distance;
    Eigen::Vector3d translation_vec =
        magnitude * Eigen::Vector3d(std::sin(phi) * std::cos(theta),
                                    std::sin(phi) * std::sin(theta),
                                    std::cos(phi));
    Sophus::SE3d pose_offset = Sophus::SE3d(orientation_vec, translation_vec);

    Sophus::SE3d init = pose_offset * rel_poses[i];
    init.translation().normalize();

    experiments.push_back(
        LoadedExperiment(bvs_1, bvs_2, transformed_covs_2, rel_poses[i], init));
  }
}

} // namespace common
} // namespace simulation
