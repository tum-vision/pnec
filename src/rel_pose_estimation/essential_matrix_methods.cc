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

#include "essential_matrix_methods.h"

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/types.hpp>

namespace pnec {
namespace rel_pose_estimation {
Sophus::SE3d
EMPoseEstimation(const opengv::bearingVectors_t &bvs1,
                 const opengv::bearingVectors_t &bvs2,
                 const Sophus::SE3d &initial_pose,
                 opengv::sac_problems::relative_pose::
                     CentralRelativePoseSacProblem::algorithm_t algorithm,
                 bool ransac) {
  opengv::relative_pose::CentralRelativeAdapter adapter(
      bvs1, bvs2, initial_pose.translation(), initial_pose.rotationMatrix());
  Sophus::SE3d solution;

  if (ransac) {
    opengv::sac::Ransac<
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
        ransac;

    std::shared_ptr<
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
        relposeproblem_ptr(
            new opengv::sac_problems::relative_pose::
                CentralRelativePoseSacProblem(adapter, algorithm));
    ransac.sac_model_ = relposeproblem_ptr;

    ransac.threshold_ = 1.0e-6;
    // ransac.max_iterations_ = options_.max_ransac_iterations_;
    ransac.max_iterations_ = 5000;

    ransac.computeModel();
    opengv::transformation_t best_transformation = ransac.model_coefficients_;
    return Sophus::SE3d(best_transformation.block<3, 3>(0, 0),
                        best_transformation.block<3, 1>(0, 3));
  } else {
    opengv::essentials_t essential_matrices;

    switch (algorithm) {
    case opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
        EIGHTPT: {
      opengv::essential_t essential_matrix =
          opengv::relative_pose::eightpt(adapter);
      essential_matrices.push_back(essential_matrix);
      break;
    }
    case opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
        NISTER: {
      opengv::essentials_t fivept_em =
          opengv::relative_pose::fivept_nister(adapter);
      essential_matrices.insert(essential_matrices.begin(), fivept_em.begin(),
                                fivept_em.end());
      break;
    }
    case opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
        STEWENIUS: {
      opengv::complexEssentials_t complexEssentialMatrices =
          opengv::relative_pose::fivept_stewenius(adapter);
      for (size_t i = 0; i < complexEssentialMatrices.size(); i++) {
        opengv::essential_t essentialMatrix;
        for (size_t r = 0; r < 3; r++) {
          for (size_t c = 0; c < 3; c++)
            essentialMatrix(r, c) = complexEssentialMatrices.at(i)(r, c).real();
        }
        essential_matrices.push_back(essentialMatrix);
      }
      break;
    }
    case opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
        SEVENPT: {
      opengv::essentials_t sevenpt_em = opengv::relative_pose::sevenpt(adapter);
      essential_matrices.insert(essential_matrices.begin(), sevenpt_em.begin(),
                                sevenpt_em.end());
      break;
    }
    }
    opengv::transformation_t best_transformation;
    if (pnec::common::PoseFromEssentialMatrix(essential_matrices, adapter,
                                              algorithm, best_transformation)) {
      return Sophus::SE3d(best_transformation.block<3, 3>(0, 0),
                          best_transformation.block<3, 1>(0, 3));
    } else {
      return initial_pose;
    }
  }
}
} // namespace rel_pose_estimation
} // namespace pnec