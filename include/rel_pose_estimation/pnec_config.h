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

#ifndef REL_POSE_ESTIMATION_PNEC_CONFIG_H_
#define REL_POSE_ESTIMATION_PNEC_CONFIG_H_

#include <ceres/ceres.h>

#include "common.h"

namespace pnec {
namespace rel_pose_estimation {

struct Options {
  bool use_nec_ = false;

  pnec::common::NoiseFrame noise_frame_ = pnec::common::Target;
  double regularization_ = 1.0e-13;

  int weighted_iterations_ = 10;
  bool use_scf_ = true;

  bool use_ceres_ = true;
  ceres::Solver::Options ceres_options_ = ceres::Solver::Options();

  bool use_ransac_ = true;
  int max_ransac_iterations_ = 5000;
  int ransac_sample_size_ = 10;

  int min_matches_ = 30;
  int min_inliers_ = 10;
  int min_matches_further_ = 20;
};
} // namespace rel_pose_estimation
} // namespace pnec

#endif // REL_POSE_ESTIMATION_PNEC_CONFIG_H_