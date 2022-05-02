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

#ifndef COMMON_COMMON_H_
#define COMMON_COMMON_H_

#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/types.hpp>
#include <sophus/se3.hpp>
#include <sophus/types.hpp>

namespace pnec {
typedef std::vector<cv::DMatch> FeatureMatches;

namespace common {
enum NoiseFrame { Host, Target };

enum CameraModel { Omnidirectional, Pinhole };

// Geometry
Eigen::Matrix3d SkewFromVector(const Eigen::Vector3d &vector);

void AnglesFromVec(const Eigen::Vector3d &vector, double &theta, double &phi);

Eigen::Matrix3d RotationBetweenPoints(const Eigen::Vector3d point1,
                                      const Eigen::Vector3d point2);

// PNEC Cost function
Eigen::Matrix3d ComposeM(const opengv::bearingVectors_t &bvs_1,
                         const opengv::bearingVectors_t &bvs_2,
                         const Eigen::Matrix3d &rotation);

Eigen::Matrix3d ComposeMPNEC(const opengv::bearingVectors_t &bvs_1,
                             const opengv::bearingVectors_t &bvs_2,
                             const std::vector<Eigen::Matrix3d> &covs,
                             const Sophus::SE3d &pose, const double &reg);

Eigen::Vector3d TranslationFromM(const Eigen::Matrix3d &M);

double Weight(const opengv::bearingVector_t &bearing_vector_1,
              const opengv::bearingVector_t &bearing_vector_2,
              const opengv::translation_t &translation,
              const opengv::rotation_t &rotation,
              const Eigen::Matrix3d &covariance, double regularization,
              bool host_frame);

double RotationalDifference(const Sophus::SO3d &rotation_1,
                            const Sophus::SO3d &rotation_2);

double TranslationalDifference(const Eigen::Vector3d &translation_1,
                               const Eigen::Vector3d &translation_2,
                               bool both_directions = true);

double CostFunction(const opengv::bearingVectors_t &bvs_1,
                    const opengv::bearingVectors_t &bvs_2,
                    const std::vector<Eigen::Matrix3d> &covs,
                    const Sophus::SE3d &camera_pose);

// essential matrix based solutions
bool PoseFromEssentialMatrix(
    const opengv::essentials_t &essentialMatrices,
    opengv::relative_pose::CentralRelativeAdapter adapter,
    opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
        algorithm_t algorithm,
    opengv::transformation_t &outModel);

// unscented Transform
Eigen::Matrix3d UnscentedTransform(const Eigen::Vector3d &mu,
                                   const Eigen::Matrix3d &cov,
                                   const Eigen::Matrix3d &K_inv,
                                   double kappa = 1.0,
                                   CameraModel camera_model = Pinhole);

std::vector<Eigen::Matrix3d>
UnscentedTransform(const std::vector<Eigen::Vector3d> &mu,
                   const std::vector<Eigen::Matrix3d> &cov,
                   const Eigen::Matrix3d &K_inv, double kappa = 1.0,
                   CameraModel camera_model = Pinhole);

} // namespace common
} // namespace pnec

#endif // COMMON_COMMON_H_