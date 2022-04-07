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

#include "scf.h"

#include <Eigen/Dense>

namespace pnec {
namespace optimization {

double obj_fun(const Eigen::Vector3d &t, std::vector<Eigen::Matrix3d> &Ai,
               std::vector<Eigen::Matrix3d> &Bi) {
  double cost = 0;
  for (size_t i = 0; i < Ai.size(); i++) {
    cost +=
        (t.transpose() * Ai[i] * t)(0, 0) / (t.transpose() * Bi[i] * t)(0, 0);
  }
  return cost;
}

std::vector<Eigen::Vector3d> fibonacci_sphere(size_t samples) {
  std::vector<Eigen::Vector3d> points;
  points.reserve(samples);

  double phi = M_PI * (3.0 - std::sqrt(5.0));
  for (size_t i = 0; i < samples; i++) {
    double y = 1.0 - ((float)i / (float)(samples - 1)) * 2.0;

    double radius = std::sqrt(1 - y * y);

    double theta = phi * (float)i;

    double x = std::cos(theta) * radius;
    double z = std::sin(theta) * radius;

    points.push_back(Eigen::Vector3d(x, y, z));
  }

  return points;
}

std::vector<double> phi_G(std::vector<Eigen::Matrix3d> &G, Eigen::Vector3d &t) {
  std::vector<double> phi_Gs;
  phi_Gs.reserve(G.size());
  for (const auto &g : G) {
    phi_Gs.push_back((t.transpose() * g * t)(0, 0));
  }
  return phi_Gs;
}

Eigen::Matrix3d construct_E(std::vector<Eigen::Matrix3d> &Ai,
                            std::vector<Eigen::Matrix3d> &Bi,
                            Eigen::Vector3d &t) {
  std::vector<double> phi_A = phi_G(Ai, t);
  std::vector<double> phi_B = phi_G(Bi, t);
  double prod_phi_B = 1.0;
  std::vector<double> frac;
  frac.resize(Ai.size());
  for (size_t i = 0; i < Ai.size(); i++) {
    frac.push_back(phi_A[i] / phi_B[i]);
    prod_phi_B *= phi_B[i];
  }
  std::vector<double> coeffs;
  coeffs.reserve(Ai.size());
  for (const auto &phi_b : phi_B) {
    coeffs.push_back(prod_phi_B / phi_b);
  }

  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < Ai.size(); i++) {
    E += (prod_phi_B / phi_B[i]) * (Ai[i] - frac[i] * Bi[i]);
  }

  return E;
}

Eigen::Matrix3d alt_construct_E(std::vector<Eigen::Matrix3d> &Ai,
                                std::vector<Eigen::Matrix3d> &Bi,
                                Eigen::Vector3d &t) {
  std::vector<double> phi_A = phi_G(Ai, t);
  std::vector<double> phi_B = phi_G(Bi, t);
  std::vector<double> frac;
  frac.resize(Ai.size());
  for (size_t i = 0; i < Ai.size(); i++) {
    frac.push_back(phi_A[i] / phi_B[i]);
  }

  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
  for (size_t i = 0; i < Ai.size(); i++) {
    E += (1.0 / phi_B[i]) * (Ai[i] - frac[i] * Bi[i]);
  }

  return E;
}

Eigen::Vector3d scf(std::vector<Eigen::Matrix3d> &Ai,
                    std::vector<Eigen::Matrix3d> &Bi, Eigen::Vector3d &init_t,
                    size_t steps) {
  Eigen::Vector3d t = init_t;
  for (size_t i = 0; i < steps; i++) {
    Eigen::Matrix3d E = alt_construct_E(Ai, Bi, t);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(E);
    Eigen::Vector3d e_values = eigensolver.eigenvalues();
    Eigen::Matrix3d e_vectors = eigensolver.eigenvectors();

    double min_e_value = e_values(0);
    t = e_vectors.col(0);
    for (size_t j = 1; j < 3; j++) {
      if (e_values(j) < min_e_value) {
        min_e_value = e_values(j);
        t = e_vectors.col(j);
      }
    }
  }
  return t;
}

} // namespace optimization
} // namespace pnec