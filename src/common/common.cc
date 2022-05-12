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

#include "common.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>

#include <opencv2/core/eigen.hpp>
#include <opengv/triangulation/methods.hpp>

#include "camera.h"

namespace pnec {
namespace common {
// Miscillaneous
struct myPair {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d first;
  double second;
};

bool operator<(const myPair &p1, const myPair &p2) {
  if (p1.second > p2.second)
    return true;
  return false;
}

Eigen::Vector2d Undistort(Eigen::Vector2d point) {
  // TODO: write custom undistort function without opencv
  const pnec::CameraParameters &cam_pars =
      Camera::instance().cameraParameters();

  const cv::Vec4d &dist_coef = cam_pars.dist_coef();

  if (dist_coef(0) == 0.0) {
    return point;
  }

  // Fill matrix with points
  const int n = 1;
  cv::Mat mat(n, 2, CV_64F);
  for (int i = 0; i < n; i++) {
    mat.at<double>(i, 0) = point[0];
    mat.at<double>(i, 1) = point[1];
  }

  // undistort points
  mat = mat.reshape(2);
  cv::undistortPoints(mat, mat, cam_pars.intrinsic(), dist_coef, cv::Mat(),
                      cam_pars.intrinsic());
  mat = mat.reshape(1);

  return Eigen::Vector2d(mat.at<double>(0, 0), mat.at<double>(0, 1));
}

// Geometry
Eigen::Matrix3d SkewFromVector(const Eigen::Vector3d &vector) {
  Eigen::Matrix3d skew_sym_matrix;
  skew_sym_matrix << 0, -vector(2), vector(1), vector(2), 0, -vector(0),
      -vector(1), vector(0), 0;
  return skew_sym_matrix;
}

void AnglesFromVec(const Eigen::Vector3d &vector, double &theta, double &phi) {
  if (vector.norm() == 0) {
    theta = 0.0;
    phi = 0.0;
  } else {
    Eigen::Vector3d norm_vec = vector / vector.norm();
    theta = std::acos(norm_vec(2));
    if (abs(theta) < 1e-10) {
      phi = 0.0;
    } else {
      phi = std::atan2(norm_vec(1), norm_vec(0));
    }
  }
}

Eigen::Matrix3d RotationBetweenPoints(const Eigen::Vector3d point1,
                                      const Eigen::Vector3d point2) {
  Eigen::Matrix3d v_hat = pnec::common::SkewFromVector(point1.cross(point2));
  double c = point1.dot(point2);

  return Eigen::Matrix3d::Identity() + v_hat + ((v_hat * v_hat) / (1 + c));
}

// PNEC energy function
Eigen::Matrix3d ComposeM(const opengv::bearingVectors_t &bvs_1,
                         const opengv::bearingVectors_t &bvs_2,
                         const Eigen::Matrix3d &rotation) {
  Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
  for (size_t i = 1; i < bvs_1.size(); i++) {
    Eigen::Vector3d n = bvs_1[i].cross(rotation * bvs_2[i]);
    M += n * n.transpose();
  }
  return M;
}

Eigen::Matrix3d ComposeMPNEC(const opengv::bearingVectors_t &bvs_1,
                             const opengv::bearingVectors_t &bvs_2,
                             const std::vector<Eigen::Matrix3d> &covs,
                             const Sophus::SE3d &pose, const double &reg) {
  Eigen::Matrix3d rotation = pose.rotationMatrix();
  Eigen::Vector3d translation = pose.translation();

  Eigen::Matrix3d M = Eigen::Matrix3d::Zero();
  for (size_t i = 1; i < bvs_1.size(); i++) {
    Eigen::Matrix3d bvs_1_hat = pnec::common::SkewFromVector(bvs_1[i]);
    Eigen::Vector3d n = bvs_1[i].cross(rotation * bvs_2[i]);
    M += n * n.transpose() /
         ((translation.transpose() * bvs_1_hat * rotation * covs[i] *
           rotation.transpose() * bvs_1_hat.transpose() * translation) +
          reg);
  }
  return M;
}

Eigen::Vector3d TranslationFromM(const Eigen::Matrix3d &M) {
  Eigen::EigenSolver<Eigen::Matrix3d> Eig(M, true);
  Eigen::Matrix<std::complex<double>, 3, 1> D_complex = Eig.eigenvalues();
  Eigen::Matrix<std::complex<double>, 3, 3> V_complex = Eig.eigenvectors();
  opengv::eigenvalues_t D;
  opengv::eigenvectors_t V;

  std::vector<myPair> pairs;
  for (size_t i = 0; i < 3; i++) {
    myPair newPair;
    newPair.second = D_complex[i].real();
    for (size_t j = 0; j < 3; j++)
      newPair.first(j, 0) = V_complex(j, i).real();
    pairs.push_back(newPair);
  }
  std::sort(pairs.begin(), pairs.end());
  for (size_t i = 0; i < 3; i++) {
    D[i] = pairs[i].second;
    V.col(i) = pairs[i].first;
  }

  double translationMagnitude = sqrt(pow(D[0], 2) + pow(D[1], 2));
  return translationMagnitude * V.col(2) /
         (translationMagnitude * V.col(2)).norm();
}

double Weight(const opengv::bearingVector_t &bearing_vector_1,
              const opengv::bearingVector_t &bearing_vector_2,
              const opengv::translation_t &translation,
              const opengv::rotation_t &rotation,
              const Eigen::Matrix3d &covariance, double regularization,
              bool host_frame) {
  if (host_frame) {
    Eigen::Matrix3d bv_2_hat =
        pnec::common::SkewFromVector(rotation * bearing_vector_2);
    Eigen::Vector3d transformed_translation =
        (translation.transpose() * bv_2_hat).transpose();

    double weight = 1 / (transformed_translation.transpose() * covariance *
                         transformed_translation)(0, 0);
    return weight;
  } else {
    Eigen::Matrix3d bv_1_hat = pnec::common::SkewFromVector(bearing_vector_1);
    Eigen::Vector3d transformed_translation =
        (translation.transpose() * bv_1_hat * rotation).transpose();

    double weight = 1 / ((transformed_translation.transpose() * covariance *
                          transformed_translation)(0, 0) +
                         regularization);
    return weight;
  }
}

double RotationalDifference(const Sophus::SO3d &rotation_1,
                            const Sophus::SO3d &rotation_2) {
  return std::abs((rotation_1.inverse() * rotation_2).logAndTheta().theta) *
         180 / M_PI;
}

double TranslationalDifference(const Eigen::Vector3d &translation_1,
                               const Eigen::Vector3d &translation_2,
                               bool both_directions) {
  double error;
  if (translation_1.norm() < 1e-10 || translation_1.norm() < 1e-10) {
    error = M_PI / 2;
  } else {
    if (both_directions) {
      double error_1 = std::acos(translation_1.dot(translation_2) /
                                 (translation_1.norm() * translation_2.norm()));
      double error_2 = std::acos(translation_1.dot(-translation_2) /
                                 (translation_1.norm() * translation_2.norm()));
      error = std::min(error_1, error_2);
    } else {
      error = std::acos(translation_1.dot(translation_2) /
                        (translation_1.norm() * translation_2.norm()));
    }
  }
  return error * 180 / M_PI;
}

double CostFunction(const opengv::bearingVectors_t &bvs_1,
                    const opengv::bearingVectors_t &bvs_2,
                    const std::vector<Eigen::Matrix3d> &covs,
                    const Sophus::SE3d &camera_pose) {
  double cost = 0;
  Eigen::Matrix3d rotation = camera_pose.rotationMatrix();
  Eigen::Vector3d translation = camera_pose.translation();

  for (size_t i = 0; i < bvs_1.size(); i++) {
    Eigen::Vector3d bv_1 = bvs_1[i];
    Eigen::Vector3d bv_2 = bvs_2[i];
    Eigen::Matrix3d cov = covs[i];

    Eigen::Matrix3d bv_1_hat = pnec::common::SkewFromVector(bv_1);

    Eigen::Vector3d transformed_t =
        (translation.transpose() * bv_1_hat * rotation).transpose();

    cost += std::pow(translation.transpose() * bv_1.cross(rotation * bv_2), 2) /
            (transformed_t.transpose() * cov * transformed_t);
  }
  return cost / bvs_1.size();
}

// essential matrix based solutions
bool PoseFromEssentialMatrix(
    const opengv::essentials_t &essentialMatrices,
    opengv::relative_pose::CentralRelativeAdapter adapter,
    opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
        algorithm_t algorithm,
    opengv::transformation_t &outModel) {
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  W(0, 1) = -1;
  W(1, 0) = 1;
  W(2, 2) = 1;

  double bestQuality = 1000000.0;
  int bestQualityIndex = -1;
  int bestQualitySubindex = -1;

  int num_points = adapter.getNumberCorrespondences();
  std::vector<int> indices(num_points);
  std::iota(std::begin(indices), std::end(indices), 0);
  for (size_t i = 0; i < essentialMatrices.size(); i++) { // decompose
    Eigen::MatrixXd tempEssential = essentialMatrices[i];
    Eigen::JacobiSVD<Eigen::MatrixXd> SVD(
        tempEssential, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::VectorXd singularValues = SVD.singularValues();

    // check for bad essential matrix
    if (singularValues[2] > 0.001) {
    };
    // continue; //singularity constraints not applied -> removed because too
    // harsh
    if (singularValues[1] < 0.75 * singularValues[0]) {
    };
    // continue; //bad essential matrix -> removed because too harsh

    // maintain scale
    double scale = singularValues[0];

    // get possible rotation and translation vectors
    opengv::rotation_t Ra = SVD.matrixU() * W * SVD.matrixV().transpose();
    opengv::rotation_t Rb =
        SVD.matrixU() * W.transpose() * SVD.matrixV().transpose();
    opengv::translation_t ta = scale * SVD.matrixU().col(2);
    opengv::translation_t tb = -ta;

    // change sign if det = -1
    if (Ra.determinant() < 0)
      Ra = -Ra;
    if (Rb.determinant() < 0)
      Rb = -Rb;

    // derive transformations
    opengv::transformation_t transformation;
    opengv::transformations_t transformations;
    transformation.col(3) = ta;
    transformation.block<3, 3>(0, 0) = Ra;
    transformations.push_back(transformation);
    transformation.col(3) = ta;
    transformation.block<3, 3>(0, 0) = Rb;
    transformations.push_back(transformation);
    transformation.col(3) = tb;
    transformation.block<3, 3>(0, 0) = Ra;
    transformations.push_back(transformation);
    transformation.col(3) = tb;
    transformation.block<3, 3>(0, 0) = Rb;
    transformations.push_back(transformation);

    // derive inverse transformations
    opengv::transformations_t inverseTransformations;
    for (size_t j = 0; j < 4; j++) {
      opengv::transformation_t inverseTransformation;
      inverseTransformation.block<3, 3>(0, 0) =
          transformations[j].block<3, 3>(0, 0).transpose();
      inverseTransformation.col(3) =
          -inverseTransformation.block<3, 3>(0, 0) * transformations[j].col(3);
      inverseTransformations.push_back(inverseTransformation);
    }

    // collect qualities for each of the four solutions solution
    Eigen::Matrix<double, 4, 1> p_hom;
    p_hom[3] = 1.0;

    for (size_t j = 0; j < 4; j++) {
      // prepare variables for triangulation and reprojection
      adapter.sett12(transformations[j].col(3));
      adapter.setR12(transformations[j].block<3, 3>(0, 0));

      // go through all features and compute quality of reprojection
      double quality = 0.0;

      int sampleSize = 0;

      switch (algorithm) {
      case opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
          NISTER:
        // Nister
        // 5 for minimal solver and additional 3 for decomposition and
        // disambiguation
        sampleSize = 5 + 3;
        break;
      case opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
          STEWENIUS:
        // Stewenius
        // 5 for minimal solver and additional 3 for decomposition and
        // disambiguation
        sampleSize = 5 + 3;
        break;
      case opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
          SEVENPT:
        // Sevenpt
        // 7 for minimal solver and additional 2 for decomposition and
        // disambiguation
        sampleSize = 7 + 2;
        break;
      case opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
          EIGHTPT:
        // EightPt
        // 8 for minimal solver and additional 1 for decomposition
        sampleSize = 8 + 1;
        break;
      }

      for (int k = 0; k < sampleSize; k++) {
        p_hom.block<3, 1>(0, 0) =
            opengv::triangulation::triangulate2(adapter, k);
        opengv::bearingVector_t reprojection1 = p_hom.block<3, 1>(0, 0);
        opengv::bearingVector_t reprojection2 =
            inverseTransformations[j] * p_hom;
        reprojection1 = reprojection1 / reprojection1.norm();
        reprojection2 = reprojection2 / reprojection2.norm();
        opengv::bearingVector_t f1 = adapter.getBearingVector1(k);
        opengv::bearingVector_t f2 = adapter.getBearingVector2(k);

        // bearing-vector based outlier criterium (select threshold
        // accordingly): 1-(f1'*f2) = 1-cos(alpha) \in [0:2]
        double reprojError1 = 1.0 - (f1.transpose() * reprojection1);
        double reprojError2 = 1.0 - (f2.transpose() * reprojection2);
        quality += reprojError1 + reprojError2;
      }

      // is quality better? (lower)
      if (quality < bestQuality) {
        bestQuality = quality;
        bestQualityIndex = i;
        bestQualitySubindex = j;
      }
    }
  }

  if (bestQualityIndex == -1)
    return false; // no solution found
  else {
    // rederive the best solution
    // decompose
    Eigen::MatrixXd tempEssential = essentialMatrices[bestQualityIndex];
    Eigen::JacobiSVD<Eigen::MatrixXd> SVD(
        tempEssential, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::VectorXd singularValues = SVD.singularValues();

    // maintain scale
    const double scale = singularValues[0];

    // get possible rotation and translation vectors
    opengv::translation_t translation;
    opengv::rotation_t rotation;

    switch (bestQualitySubindex) {
    case 0:
      translation = scale * SVD.matrixU().col(2);
      rotation = SVD.matrixU() * W * SVD.matrixV().transpose();
      break;
    case 1:
      translation = scale * SVD.matrixU().col(2);
      rotation = SVD.matrixU() * W.transpose() * SVD.matrixV().transpose();
      break;
    case 2:
      translation = -scale * SVD.matrixU().col(2);
      rotation = SVD.matrixU() * W * SVD.matrixV().transpose();
      break;
    case 3:
      translation = -scale * SVD.matrixU().col(2);
      rotation = SVD.matrixU() * W.transpose() * SVD.matrixV().transpose();
      break;
    default:
      // TODO: not correct
      return false;
    }

    // change sign if det = -1
    if (rotation.determinant() < 0)
      rotation = -rotation;

    // output final selection
    outModel.block<3, 3>(0, 0) = rotation;
    outModel.col(3) = translation;
  }
  return true;
}

// Unscented Transform
Eigen::Vector3d Unproject(const Eigen::Vector2d &img_pt,
                          const Eigen::Matrix3d &K_inv) {
  Eigen::Vector3d mu(img_pt(0), img_pt(1), 1.0);

  return (K_inv * mu).normalized();
}

Eigen::Matrix3d UnscentedTransform(const Eigen::Vector3d &mu,
                                   const Eigen::Matrix3d &cov,
                                   const Eigen::Matrix3d &K_inv, double kappa,
                                   CameraModel camera_model) {
  // Pass mu[0], mu[1], 1.0
  int n = 2;
  int m = 2 * n + 1;

  Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
  if (camera_model == Omnidirectional) {
    Eigen::Matrix3d rotation =
        RotationBetweenPoints(Eigen::Vector3d(0.0, 0.0, 1.0), mu.normalized());
    C.topLeftCorner(2, 2) = (rotation.transpose() * cov * rotation)
                                .topLeftCorner(2, 2)
                                .llt()
                                .matrixL();
    C = rotation * C;
  } else {
    C.topLeftCorner(2, 2) = cov.topLeftCorner(2, 2).llt().matrixL();
  }

  std::vector<Eigen::Vector3d> points;
  points.reserve(m);
  std::vector<double> weights;
  weights.reserve(m);
  points.push_back(mu);
  weights.push_back(kappa / ((float)n + kappa));
  for (int i = 0; i < n; i++) {
    Eigen::Vector3d c_col = C.col(i);
    points.push_back(mu + c_col);
    weights.push_back(0.5 / ((float)n + kappa));
  }
  for (int i = 0; i < n; i++) {
    Eigen::Vector3d c_col = C.col(i);
    points.push_back(mu - c_col);
    weights.push_back(0.5 / ((float)n + kappa));
  }

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> transformed_points;
  transformed_points.reserve(m);
  for (int i = 0; i < m; i++) {
    Eigen::Vector3d t_point;
    if (camera_model == Omnidirectional) {
      t_point = points[i].normalized();
    } else {
      t_point = (K_inv * points[i]).normalized();
    }
    transformed_points.push_back(t_point);
    mean = mean + (weights[i] * t_point);
  }
  Eigen::Matrix3d sigma = Eigen::Matrix3d::Zero();
  for (int i = 0; i < m; i++) {
    sigma = sigma + weights[i] * (transformed_points[i] - mean) *
                        (transformed_points[i] - mean).transpose();
  }

  return sigma;
}

std::vector<Eigen::Matrix3d>
UnscentedTransform(const std::vector<Eigen::Vector3d> &mus,
                   const std::vector<Eigen::Matrix3d> &covs,
                   const Eigen::Matrix3d &K_inv, double kappa,
                   CameraModel camera_model) {
  if (mus.size() != covs.size()) {
    std::cout << "Warning! The size of the vectors and covariances provided "
                 "are not the same. Returning the original covariances"
              << std::endl;
    return covs;
  }

  std::vector<Eigen::Matrix3d> proj_covs;
  proj_covs.reserve(mus.size());

  for (int i = 0; i < mus.size(); i++) {
    Eigen::Vector3d mu = mus[i];
    Eigen::Matrix3d cov = covs[i];
    proj_covs.push_back(
        pnec::common::UnscentedTransform(mu, cov, K_inv, kappa, camera_model));
  }

  return proj_covs;
}

} // namespace common
} // namespace pnec