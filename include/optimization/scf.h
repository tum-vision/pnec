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

#ifndef OPTIMIZATION_SCF_H_
#define OPTIMIZATION_SCF_H_

#include "sophus/se3.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace pnec {
namespace optimization {

double obj_fun(const Eigen::Vector3d &t, std::vector<Eigen::Matrix3d> &Ai,
               std::vector<Eigen::Matrix3d> &Bi);

std::vector<Eigen::Vector3d> fibonacci_sphere(size_t samples = 1);

std::vector<double> phi_G(std::vector<Eigen::Matrix3d> &G, Eigen::Vector3d &t);

Eigen::Matrix3d construct_E(std::vector<Eigen::Matrix3d> &Ai,
                            std::vector<Eigen::Matrix3d> &Bi,
                            Eigen::Vector3d &t);

Eigen::Matrix3d alt_construct_E(std::vector<Eigen::Matrix3d> &Ai,
                                std::vector<Eigen::Matrix3d> &Bi,
                                Eigen::Vector3d &t);

Eigen::Vector3d scf(std::vector<Eigen::Matrix3d> &Ai,
                    std::vector<Eigen::Matrix3d> &Bi, Eigen::Vector3d &init_t,
                    size_t steps = 10);

} // namespace optimization
} // namespace pnec

#endif // OPTIMIZATION_SCF_H_