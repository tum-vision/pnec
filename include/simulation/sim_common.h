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

#ifndef SIMULATION_SIM_COMMON_H_
#define SIMULATION_SIM_COMMON_H_

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "sophus/se3.hpp"

#include "common.h"

namespace simulation {
enum NoiseType {
  isotropic_homogenous,
  isotropic_inhomogenous,
  anisotropic_homogenous,
  anisotropic_inhomogenous
};

enum ExperimentType { standard, anisotropy, outlier, offset };

namespace common {
template <typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in) {

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) {
    return linspaced;
  }
  if (num == 1) {
    linspaced.push_back(start);
    return linspaced;
  }

  double delta = (end - start) / (num - 1);

  for (int i = 0; i < num - 1; ++i) {
    linspaced.push_back(start + delta * i);
  }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

struct normal_random_variable {
  normal_random_variable(Eigen::MatrixXd const &covar)
      : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar) {}

  normal_random_variable(Eigen::VectorXd const &mean,
                         Eigen::MatrixXd const &covar)
      : mean(mean) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
    transform = eigenSolver.eigenvectors() *
                eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
  }

  Eigen::VectorXd mean;
  Eigen::MatrixXd transform;

  Eigen::VectorXd operator()() const {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<> dist;

    return mean + transform * Eigen::VectorXd{mean.size()}.unaryExpr(
                                  [&](auto x) { return dist(gen); });
  }
};

double clip(double n, double lower, double upper);

struct Results {
  Results();
  Results(size_t num_experiments);
  ~Results();

  void AddResults(const Sophus::SE3d &solution,
                  const Sophus::SE3d &ground_truth,
                  const opengv::bearingVectors_t &bvs_1,
                  const opengv::bearingVectors_t &bvs_2,
                  const std::vector<Eigen::Matrix3d> &covs);
  std::vector<double> rotational_error_;
  std::vector<double> translational_error_;
  std::vector<double> costs_;
};

// https://stackoverflow.com/a/1120224
class CSVRow {
public:
  double operator[](std::size_t index) const {
    return std::stod(m_line.substr(m_data[index] + 1,
                                   m_data[index + 1] - (m_data[index] + 1)));
  }
  std::size_t size() const { return m_data.size() - 1; }
  void readNextRow(std::istream &str) {
    std::getline(str, m_line);

    m_data.clear();
    m_data.emplace_back(-1);
    std::string::size_type pos = 0;
    while ((pos = m_line.find(',', pos)) != std::string::npos) {
      m_data.emplace_back(pos);
      ++pos;
    }
    // This checks for a trailing comma with no data after it.
    pos = m_line.size();
    m_data.emplace_back(pos);
  }

private:
  std::string m_line;
  std::vector<int> m_data;
};

std::istream &operator>>(std::istream &str, CSVRow &data);

class CSVIterator {
public:
  typedef std::input_iterator_tag iterator_category;
  typedef CSVRow value_type;
  typedef std::size_t difference_type;
  typedef CSVRow *pointer;
  typedef CSVRow &reference;

  CSVIterator(std::istream &str) : m_str(str.good() ? &str : NULL) {
    ++(*this);
  }
  CSVIterator() : m_str(NULL) {}

  // Pre Increment
  CSVIterator &operator++() {
    if (m_str) {
      if (!((*m_str) >> m_row)) {
        m_str = NULL;
      }
    }
    return *this;
  }
  // Post increment
  CSVIterator operator++(int) {
    CSVIterator tmp(*this);
    ++(*this);
    return tmp;
  }
  CSVRow const &operator*() const { return m_row; }
  CSVRow const *operator->() const { return &m_row; }

  bool operator==(CSVIterator const &rhs) {
    return ((this == &rhs) || ((this->m_str == NULL) && (rhs.m_str == NULL)));
  }
  bool operator!=(CSVIterator const &rhs) { return !((*this) == rhs); }

private:
  std::istream *m_str;
  CSVRow m_row;
};

class CSVRange {
  std::istream &stream;

public:
  CSVRange(std::istream &str) : stream(str) {}
  CSVIterator begin() const { return CSVIterator{stream}; }
  CSVIterator end() const { return CSVIterator{}; }
};

void GetFeatures(const opengv::bearingVectors_t &points_1,
                 const opengv::bearingVectors_t &points_2,
                 const std::vector<Eigen::Matrix3d> &covs,
                 opengv::bearingVectors_t &bvs1, opengv::bearingVectors_t &bvs2,
                 std::vector<Eigen::Matrix3d> &transformed_covs,
                 const pnec::common::CameraModel camera_model,
                 const pnec::common::NoiseFrame noise_frame =
                     pnec::common::NoiseFrame::Target);

struct LoadedExperiment {
public:
  LoadedExperiment(opengv::bearingVectors_t &bvs_1,
                   opengv::bearingVectors_t &bvs_2,
                   std::vector<Eigen::Matrix3d> &transformed_covs_2,
                   Sophus::SE3d &rel_pose, Sophus::SE3d &init)
      : bearing_vectors_1(bvs_1), bearing_vectors_2(bvs_2),
        covariances_2(transformed_covs_2), ground_truth(rel_pose),
        initialization(init) {}
  opengv::bearingVectors_t bearing_vectors_1;
  opengv::bearingVectors_t bearing_vectors_2;
  std::vector<Eigen::Matrix3d> covariances_2;
  Sophus::SE3d ground_truth;
  Sophus::SE3d initialization;
};

void ReadExperiments(
    std::string folder, std::vector<LoadedExperiment> &experiments,
    std::mt19937 &generator, std::uniform_real_distribution<double> &uniform01,
    pnec::common::CameraModel camera_model,
    pnec::common::NoiseFrame noise_frame = pnec::common::NoiseFrame::Target,
    double init_scaling = 1.0);
} // namespace common
} // namespace simulation

#endif // SIMULATION_SIM_COMMON_H_
