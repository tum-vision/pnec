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

#include "pnec.h"

#include "math.h"
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Lmeds.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/EigensolverSacProblem.hpp>
#include <opengv/types.hpp>

#include "common.h"
#include "nec_ceres.h"
#include "pnec_ceres.h"
#include "scf.h"

namespace pnec {
namespace rel_pose_estimation {
PNEC::PNEC(const pnec::rel_pose_estimation::Options &options)
    : options_{options} {}

PNEC::~PNEC() {}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const Sophus::SE3d &initial_pose) {
  std::vector<int> inliers;
  return Solve(bvs1, bvs2, projected_covs, initial_pose, inliers);
}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const Sophus::SE3d &initial_pose,
                         std::vector<int> &inliers) {
  opengv::bearingVectors_t in_bvs1;
  opengv::bearingVectors_t in_bvs2;
  std::vector<Eigen::Matrix3d> in_proj_covs;

  Sophus::SE3d ES_solution = Eigensolver(bvs1, bvs2, initial_pose, inliers);
  if (options_.use_ransac_) {
    InlierExtraction(bvs1, bvs2, projected_covs, in_bvs1, in_bvs2, in_proj_covs,
                     inliers);
  } else {
    in_bvs1 = bvs1;
    in_bvs2 = bvs2;
    in_proj_covs = projected_covs;
  }

  if (options_.use_nec_) {
    if (options_.use_ceres_) {
      return NECCeresSolver(in_bvs1, in_bvs2, ES_solution);
    } else {
      return ES_solution;
    }
  } else {
    Sophus::SE3d ceres_init;
    if (options_.weighted_iterations_ > 1) {
      ceres_init =
          WeightedEigensolver(in_bvs1, in_bvs2, in_proj_covs, ES_solution);
    } else if (options_.weighted_iterations_ == 1) {
      ceres_init = ES_solution;
    } else {
      ceres_init = initial_pose;
    }

    Sophus::SE3d solution;
    if (options_.use_ceres_) {
      solution = CeresSolver(in_bvs1, in_bvs2, in_proj_covs, ceres_init);
    } else {
      solution = ceres_init;
    }
    return solution;
  }
}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const Sophus::SE3d &initial_pose,
                         pnec::common::FrameTiming &timing) {
  std::vector<int> inliers;
  return Solve(bvs1, bvs2, projected_covs, initial_pose, inliers, timing);
}

Sophus::SE3d PNEC::Solve(const opengv::bearingVectors_t &bvs1,
                         const opengv::bearingVectors_t &bvs2,
                         const std::vector<Eigen::Matrix3d> &projected_covs,
                         const Sophus::SE3d &initial_pose,
                         std::vector<int> &inliers,
                         pnec::common::FrameTiming &timing) {
  opengv::bearingVectors_t in_bvs1;
  opengv::bearingVectors_t in_bvs2;
  std::vector<Eigen::Matrix3d> in_proj_covs;

  auto tic = std::chrono::high_resolution_clock::now(),
       toc = std::chrono::high_resolution_clock::now();
  Sophus::SE3d ES_solution = Eigensolver(bvs1, bvs2, initial_pose, inliers);
  if (options_.use_ransac_) {
    InlierExtraction(bvs1, bvs2, projected_covs, in_bvs1, in_bvs2, in_proj_covs,
                     inliers);
  } else {
    in_bvs1 = bvs1;
    in_bvs2 = bvs2;
    in_proj_covs = projected_covs;
  }
  toc = std::chrono::high_resolution_clock::now();
  timing.nec_es_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

  if (options_.use_nec_) {
    Sophus::SE3d solution;
    if (options_.use_ceres_) {
      tic = std::chrono::high_resolution_clock::now();
      solution = NECCeresSolver(in_bvs1, in_bvs2, ES_solution);
      toc = std::chrono::high_resolution_clock::now();
      timing.ceres_ =
          std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    } else {
      solution = ES_solution;
      timing.ceres_ = std::chrono::milliseconds(0);
    }
    return solution;
  } else {
    Sophus::SE3d ceres_init;
    if (options_.weighted_iterations_ > 1) {
      tic = std::chrono::high_resolution_clock::now();
      ceres_init =
          WeightedEigensolver(in_bvs1, in_bvs2, in_proj_covs, ES_solution);
      toc = std::chrono::high_resolution_clock::now();
      timing.it_es_ =
          std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
      timing.avg_it_es_ = std::chrono::duration_cast<std::chrono::milliseconds>(
          (toc - tic) / options_.weighted_iterations_);
    } else if (options_.weighted_iterations_ == 1) {
      ceres_init = ES_solution;
      timing.it_es_ = std::chrono::milliseconds(0);
    } else {
      ceres_init = initial_pose;
      timing.it_es_ = std::chrono::milliseconds(0);
    }
    Sophus::SE3d solution;
    if (options_.use_ceres_) {
      tic = std::chrono::high_resolution_clock::now();
      solution = CeresSolver(in_bvs1, in_bvs2, in_proj_covs, ceres_init);
      toc = std::chrono::high_resolution_clock::now();
      timing.ceres_ =
          std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    } else {
      solution = ceres_init;
      timing.ceres_ = std::chrono::milliseconds(0);
    }
    return solution;
  }
}

void PNEC::InlierExtraction(const opengv::bearingVectors_t &bvs1,
                            const opengv::bearingVectors_t &bvs2,
                            const std::vector<Eigen::Matrix3d> &proj_covs,
                            opengv::bearingVectors_t &in_bvs1,
                            opengv::bearingVectors_t &in_bvs2,
                            std::vector<Eigen::Matrix3d> &in_proj_covs,
                            const std::vector<int> &inliers) {
  in_bvs1.clear();
  in_bvs1.reserve(inliers.size());
  in_bvs2.clear();
  in_bvs2.reserve(inliers.size());
  in_proj_covs.clear();
  in_proj_covs.reserve(inliers.size());
  for (const auto inlier : inliers) {
    // for (int inlier = 0; inlier < n_matches; inlier++) {
    in_bvs1.push_back(bvs1[inlier]);
    in_bvs2.push_back(bvs2[inlier]);
    in_proj_covs.push_back(proj_covs[inlier]);
  }
}

Sophus::SE3d PNEC::Eigensolver(const opengv::bearingVectors_t &bvs1,
                               const opengv::bearingVectors_t &bvs2,
                               const Sophus::SE3d &initial_pose,
                               std::vector<int> &inliers) {
  opengv::rotation_t init_rotation = initial_pose.rotationMatrix();
  opengv::relative_pose::CentralRelativeAdapter adapter(bvs1, bvs2,
                                                        init_rotation);
  Sophus::SE3d solution;
  if (options_.use_ransac_) {
    opengv::sac::Ransac<
        opengv::sac_problems::relative_pose::EigensolverSacProblem>
        ransac;
    std::shared_ptr<opengv::sac_problems::relative_pose::EigensolverSacProblem>
        eigenproblem_ptr(
            new opengv::sac_problems::relative_pose::EigensolverSacProblem(
                adapter, options_.ransac_sample_size_));
    ransac.sac_model_ = eigenproblem_ptr;
    ransac.threshold_ = 1.0e-6;
    ransac.max_iterations_ = options_.max_ransac_iterations_;

    ransac.computeModel();

    opengv::sac_problems::relative_pose::EigensolverSacProblem::model_t
        optimizedModel;
    eigenproblem_ptr->optimizeModelCoefficients(
        ransac.inliers_, ransac.model_coefficients_, optimizedModel);

    inliers = ransac.inliers_;
    // TODO: use translation from M
    opengv::bearingVectors_t in_bvs1;
    opengv::bearingVectors_t in_bvs2;
    in_bvs1.reserve(inliers.size());
    in_bvs2.reserve(inliers.size());
    for (const auto inlier : inliers) {
      // for (int inlier = 0; inlier < n_matches; inlier++) {
      in_bvs1.push_back(bvs1[inlier]);
      in_bvs2.push_back(bvs2[inlier]);
    }

    opengv::translation_t translation = pnec::common::TranslationFromM(
        pnec::common::ComposeM(in_bvs1, in_bvs2, optimizedModel.rotation));
    solution = Sophus::SE3d(optimizedModel.rotation, translation);
  } else {
    opengv::rotation_t rotation = opengv::relative_pose::eigensolver(adapter);
    opengv::translation_t translation = pnec::common::TranslationFromM(
        pnec::common::ComposeM(bvs1, bvs2, rotation));
    inliers.clear();
    solution = Sophus::SE3d(rotation, translation);
  }
  return solution;
}

Sophus::SE3d PNEC::WeightedEigensolver(
    const opengv::bearingVectors_t &bvs1, const opengv::bearingVectors_t &bvs2,
    const std::vector<Eigen::Matrix3d> &projected_covariances,
    const Sophus::SE3d &initial_pose) {
  Sophus::SE3d rel_pose = initial_pose;
  opengv::rotation_t old_rotation;

  int seed = 1;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);
  for (size_t iteration = 0; iteration < options_.weighted_iterations_ - 1;
       iteration++) {
    std::vector<double> weights;
    for (size_t i = 0; i < projected_covariances.size(); i++) {
      double weight = pnec::common::Weight(
          bvs1[i], bvs2[i], initial_pose.translation(),
          initial_pose.rotationMatrix(), projected_covariances[i],
          options_.regularization_, false);
      weights.push_back(weight * 1.0e-8);
    }

    opengv::bearingVectors_t w_bvs2;
    w_bvs2.reserve(bvs2.size());
    for (size_t i = 0; i < bvs2.size(); i++) {
      w_bvs2.push_back(bvs2[i] * std::sqrt(weights[i]));
    }

    opengv::relative_pose::CentralRelativeAdapter adapter(
        bvs1, w_bvs2, rel_pose.rotationMatrix());

    old_rotation = initial_pose.rotationMatrix();

    opengv::rotation_t rotation = opengv::relative_pose::eigensolver(adapter);

    std::vector<Eigen::Matrix3d> Ai, Bi;
    // TODO: Host and Target frame
    Ai.reserve(projected_covariances.size());
    Bi.reserve(projected_covariances.size());
    for (size_t i = 0; i < projected_covariances.size(); i++) {
      Eigen::Matrix3d bv1_skew = pnec::common::SkewFromVector(bvs1[i]);
      Ai.push_back((bv1_skew * rotation * bvs2[i]) *
                   (bv1_skew * rotation * bvs2[i]).transpose());
      Bi.push_back(bv1_skew * rotation * projected_covariances[i] *
                       rotation.transpose() * bv1_skew.transpose() +
                   Eigen::Matrix3d::Identity() * options_.regularization_);
    }

    std::vector<Eigen::Vector3d> points =
        pnec::optimization::fibonacci_sphere(500);
    Eigen::Vector3d best_point = rel_pose.translation();
    double best_cost = pnec::optimization::obj_fun(best_point, Ai, Bi);
    for (const auto &point : points) {
      double cost = pnec::optimization::obj_fun(point, Ai, Bi);
      if (cost < best_cost) {
        best_cost = cost;
        best_point = point;
      }
    }

    opengv::translation_t translation =
        pnec::optimization::scf(Ai, Bi, best_point, 10);

    rel_pose = Sophus::SE3d(rotation, translation);
  }
  return rel_pose;
}

Sophus::SE3d
PNEC::CeresSolver(const opengv::bearingVectors_t &bvs1,
                  const opengv::bearingVectors_t &bvs2,
                  const std::vector<Eigen::Matrix3d> &projected_covariances,
                  const Sophus::SE3d &initial_pose) {
  pnec::optimization::PNECCeres optimizer;
  optimizer.InitValues(Eigen::Quaterniond(initial_pose.rotationMatrix()),
                       initial_pose.translation());
  std::vector<Eigen::Vector3d> bvs_1;
  std::vector<Eigen::Vector3d> bvs_2;
  for (const auto &bv : bvs1) {
    bvs_1.push_back(bv);
  }
  for (const auto &bv : bvs2) {
    bvs_2.push_back(bv);
  }
  optimizer.Optimize(bvs_1, bvs_2, projected_covariances,
                     options_.regularization_);

  return optimizer.Result();
}

Sophus::SE3d
PNEC::CeresSolverFull(const opengv::bearingVectors_t &bvs1,
                      const opengv::bearingVectors_t &bvs2,
                      const std::vector<Eigen::Matrix3d> &projected_covariances,
                      double regularization, const Sophus::SE3d &initial_pose) {
  pnec::optimization::PNECCeres optimizer;
  optimizer.InitValues(Eigen::Quaterniond(initial_pose.rotationMatrix()),
                       initial_pose.translation());
  std::vector<Eigen::Vector3d> bvs_1;
  std::vector<Eigen::Vector3d> bvs_2;
  for (const auto &bv : bvs1) {
    bvs_1.push_back(bv);
  }
  for (const auto &bv : bvs2) {
    bvs_2.push_back(bv);
  }

  optimizer.Optimize(bvs_1, bvs_2, projected_covariances, regularization);

  return optimizer.Result();
}

Sophus::SE3d PNEC::NECCeresSolver(const opengv::bearingVectors_t &bvs1,
                                  const opengv::bearingVectors_t &bvs2,
                                  const Sophus::SE3d &initial_pose) {
  pnec::optimization::NECCeres optimizer;
  optimizer.InitValues(Eigen::Quaterniond(initial_pose.rotationMatrix()),
                       initial_pose.translation());
  std::vector<Eigen::Vector3d> bvs_1;
  std::vector<Eigen::Vector3d> bvs_2;
  for (const auto &bv : bvs1) {
    bvs_1.push_back(bv);
  }
  for (const auto &bv : bvs2) {
    bvs_2.push_back(bv);
  }
  optimizer.Optimize(bvs_1, bvs_2);

  return optimizer.Result();
}

} // namespace rel_pose_estimation
} // namespace pnec