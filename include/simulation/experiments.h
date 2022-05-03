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

#ifndef SIMULATION_EXPERIMENTS_H_
#define SIMULATION_EXPERIMENTS_H_

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "sophus/se3.hpp"

#include "common.h"
#include "sim_common.h"

namespace simulation {
namespace experiments {

struct SingleExperiment {
  SingleExperiment() {}
  SingleExperiment(size_t num_points) {
    points_1.reserve(num_points);
    points_2.reserve(num_points);
    covs_1.reserve(num_points);
    covs_2.reserve(num_points);
  }
  SingleExperiment(Sophus::SE3d p_1, Sophus::SE3d p_2,
                   std::vector<Eigen::Vector3d> pts_1,
                   std::vector<Eigen::Vector3d> pts_2,
                   std::vector<Eigen::Matrix3d> c_1,
                   std::vector<Eigen::Matrix3d> c_2)
      : pose_1(p_1), pose_2(p_2), points_1(pts_1), points_2(pts_2), covs_1(c_1),
        covs_2(c_2) {}
  ~SingleExperiment() {}

  Sophus::SE3d pose_1;
  Sophus::SE3d pose_2;

  std::vector<Eigen::Vector3d> points_1;
  std::vector<Eigen::Vector3d> points_2;
  std::vector<Eigen::Matrix3d> covs_1;
  std::vector<Eigen::Matrix3d> covs_2;
};

class BaseExperiments {
public:
  BaseExperiments(int seed = 1, bool translation = true,
                  pnec::common::CameraModel camera_model =
                      pnec::common::CameraModel::Omnidirectional)
      : seed_{seed}, translation_{translation}, camera_model_{camera_model} {
    SetRandom();
  }
  ~BaseExperiments() {}

protected:
  void SetRandom() {
    generator_ = std::mt19937(seed_);
    uniform01_ = std::uniform_real_distribution<double>(0.0, 1.0);
    normal_generator_ = std::default_random_engine();
    distribution_ = std::normal_distribution<double>(0.0, 1.0);
  }
  void SamplePoses(double max_euler_angle, bool translation,
                   Sophus::SE3d &pose_1, Sophus::SE3d &pose_2);
  void SamplePoints(size_t num_points, pnec::common::CameraModel camera_model,
                    Sophus::SE3d &pose_1, Sophus::SE3d &pose_2,
                    std::vector<Eigen::Vector3d> &points_1,
                    std::vector<Eigen::Vector3d> &points_2);

  void SaveExperiments(std::string base_folder);

  Eigen::Matrix3d CovFromParameters(double noise_scale, double scale,
                                    double alpha, double beta);

  int seed_ = 1;
  std::mt19937 generator_;
  std::uniform_real_distribution<double> uniform01_;
  std::default_random_engine normal_generator_;
  std::normal_distribution<double> distribution_;

  bool translation_;
  pnec::common::CameraModel camera_model_;

  std::vector<SingleExperiment> experiments_;
};

} // namespace experiments
} // namespace simulation

#endif // SIMULATION_EXPERIMENTS_H_
