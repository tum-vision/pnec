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

#ifndef SIMULATION_OFFSET_EXPERIMENTS_H_
#define SIMULATION_OFFSET_EXPERIMENTS_H_

#include "common.h"
#include "experiments.h"

namespace simulation {
namespace experiments {

class OffsetExperiments : BaseExperiments {
public:
  OffsetExperiments(
      pnec::common::CameraModel camera_model, simulation::NoiseType noise_type,
      bool translation = true,
      pnec::common::NoiseFrame noise_frame = pnec::common::NoiseFrame::Target,
      int seed = 1)
      : BaseExperiments(seed, translation, camera_model),
        noise_type_(noise_type), noise_frame_(noise_frame) {
    if (noise_type == simulation::NoiseType::isotropic_homogenous) {
      std::cout << "There is no implementation for the offset experiment with "
                   "isotropic and homogeneous noise."
                << std::endl;
    }
  }

  void GenerateExperiments(std::string base_folder, size_t num_experiments,
                           size_t num_points, std::vector<double> levels);

private:
  void SampleCovariances(size_t num_covs, double offset_level,
                         std::vector<Eigen::Matrix3d> &covariances,
                         std::vector<Eigen::Matrix3d> &est_covariances);
  void AddNoise(const std::vector<Eigen::Vector3d> &points,
                const std::vector<Eigen::Matrix3d> &covariances,
                const std::vector<Eigen::Matrix3d> &est_covariances,
                std::vector<Eigen::Vector3d> &points_with_noise,
                std::vector<Eigen::Matrix3d> &est_img_plane_covariances);

  simulation::NoiseType noise_type_;
  pnec::common::NoiseFrame noise_frame_;
};

} // namespace experiments
} // namespace simulation

#endif // SIMULATION_OFFSET_EXPERIMENTS_H_
