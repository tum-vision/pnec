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

#include <boost/filesystem.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <fstream>
#include <map>
#include <random>
#include <stdio.h>

#include "sophus/so3.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/CentralRelativeWeightingAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include "common.h"
#include "essential_matrix_methods.h"
#include "math.h"
#include "pnec.h"
#include "pnec_config.h"
#include "sim_common.h"

void log_init() {
  boost::log::add_console_log(std::cout, boost::log::keywords::format =
                                             "[%Severity%] %Message%");
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::debug);
}

void LoadExperiment(std::string folder,
                    simulation::common::LoadedExperiment &experiment) {}

void PNEC(std::unordered_map<std::string, simulation::common::Results> &results,
          const simulation::common::LoadedExperiment &experiment,
          pnec::rel_pose_estimation::Options options =
              pnec::rel_pose_estimation::Options(),
          std::string name = "PNEC") {
  pnec::rel_pose_estimation::PNEC pnec(options);
  Sophus::SE3d result =
      pnec.Solve(experiment.bearing_vectors_1, experiment.bearing_vectors_2,
                 experiment.covariances_2, experiment.initialization);
  results[name].AddResults(
      result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);
}

void EMMethods(
    std::unordered_map<std::string, simulation::common::Results> &results,
    const simulation::common::LoadedExperiment &experiment) {
  Sophus::SE3d result;
  result = pnec::rel_pose_estimation::EMPoseEstimation(
      experiment.bearing_vectors_1, experiment.bearing_vectors_2,
      experiment.initialization,
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
          EIGHTPT,
      false);
  results["8pt"].AddResults(
      result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);

  result = pnec::rel_pose_estimation::EMPoseEstimation(
      experiment.bearing_vectors_1, experiment.bearing_vectors_2,
      experiment.initialization,
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
          SEVENPT,
      false);
  results["7pt"].AddResults(
      result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);

  result = pnec::rel_pose_estimation::EMPoseEstimation(
      experiment.bearing_vectors_1, experiment.bearing_vectors_2,
      experiment.initialization,
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
          NISTER,
      false);
  results["Nister5pt"].AddResults(
      result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);

  result = pnec::rel_pose_estimation::EMPoseEstimation(
      experiment.bearing_vectors_1, experiment.bearing_vectors_2,
      experiment.initialization,
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem::
          STEWENIUS,
      false);
  results["Stewenius5pt"].AddResults(
      result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);
}

void Ablation(
    std::unordered_map<std::string, simulation::common::Results> &results,
    const simulation::common::LoadedExperiment &experiment,
    pnec::rel_pose_estimation::Options options =
        pnec::rel_pose_estimation::Options()) {

  options.use_ransac_ = false;
  std::vector<int> dummy_inliers;
  pnec::rel_pose_estimation::PNEC pnec(options);
  Sophus::SE3d nec_result = pnec.Eigensolver(
      experiment.bearing_vectors_1, experiment.bearing_vectors_2,
      experiment.initialization, dummy_inliers);

  results["NEC"].AddResults(
      nec_result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);

  Sophus::SE3d nec_ls_result = pnec.NECCeresSolver(experiment.bearing_vectors_1,
                                                   experiment.bearing_vectors_2,
                                                   experiment.initialization);
  results["NEC-LS"].AddResults(
      nec_ls_result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);

  Sophus::SE3d nec_pnec_ls_result = pnec.CeresSolver(
      experiment.bearing_vectors_1, experiment.bearing_vectors_2,
      experiment.covariances_2, nec_result);
  results["NEC PNEC-LS"].AddResults(
      nec_pnec_ls_result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);

  Sophus::SE3d pnec_only_ls = pnec.CeresSolver(
      experiment.bearing_vectors_1, experiment.bearing_vectors_2,
      experiment.covariances_2, experiment.initialization);
  results["PNEC only LS"].AddResults(
      pnec_only_ls, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);

  Sophus::SE3d itES_out = pnec.WeightedEigensolver(
      experiment.bearing_vectors_1, experiment.bearing_vectors_2,
      experiment.covariances_2, nec_result);
  results["PNEC w/o LS"].AddResults(
      nec_result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);

  Sophus::SE3d pnec_result = pnec.CeresSolver(
      experiment.bearing_vectors_1, experiment.bearing_vectors_2,
      experiment.covariances_2, itES_out);
  results["PNEC"].AddResults(
      pnec_result, experiment.ground_truth, experiment.bearing_vectors_1,
      experiment.bearing_vectors_2, experiment.covariances_2);
}

int main(int argc, const char *argv[]) {
  log_init();

  const cv::String keys =
      "{help h usage ?    |          | print this message}"
      "{@base_folder      |none      | base folder name}"
      "{@experiment_type  |none      | create anisotropy experiment}"
      "{@camera_model     |none      | pinhole or omnidirectional camera}"
      "{@translation      |true      | translation or not}"
      "{@noise_type       |none      | noise_type}"
      "{@name             |none      | name experiment of directory for "
      "results}"
      "{e EMMethods       |false     | solve with essential matrix methods}"
      "{a AblationMethods |false     | solve with ablation methods}"
      "{sc init_scaling   |1.0       | scaling of intialization offset to the "
      "ground truth}"
      "{ls linspace_start |0.125     | start of linspace for endings}"
      "{le linspace_end   |4.000     | end of linspace for endings}"
      "{ln linspace_number|32        | start of linspace for endings}";

  cv::CommandLineParser parser(argc, argv, keys);

  size_t parse_counter = 0;
  const std::string base_folder(parser.get<cv::String>(parse_counter++));
  const int exp_num(parser.get<int>(parse_counter++));
  const int camera_model_num(parser.get<int>(parse_counter++));
  const bool translation(parser.get<bool>(parse_counter++));
  const int noise_int(parser.get<int>(parse_counter++));
  const std::string name(parser.get<cv::String>(parse_counter++));
  const bool em_methods(parser.get<bool>("e"));
  const bool ablation_methods(parser.get<bool>("a"));
  const double init_scaling(parser.get<double>("sc"));
  const double linspace_start(parser.get<double>("ls"));
  const double linspace_end(parser.get<double>("le"));
  const double linspace_number(parser.get<int>("ln"));

  simulation::ExperimentType experiment_type =
      static_cast<simulation::ExperimentType>(exp_num);
  pnec::common::CameraModel camera_model =
      static_cast<pnec::common::CameraModel>(camera_model_num);
  simulation::NoiseType noise_type =
      static_cast<simulation::NoiseType>(noise_int);

  std::string folder;
  switch (experiment_type) {
  case simulation::ExperimentType::standard: {
    folder = base_folder + "standard/";
    break;
  }
  case simulation::ExperimentType::anisotropy: {
    folder = base_folder + "ansisotropy/";
    break;
  }
  case simulation::ExperimentType::outlier: {
    folder = base_folder + "outlier/";
    break;
  }
  case simulation::ExperimentType::offset: {
    folder = base_folder + "offset/";
    break;
  }
  }

  switch (camera_model) {
  case pnec::common::CameraModel::Omnidirectional: {
    folder = folder + "omnidirectional/";
    break;
  }
  case pnec::common::CameraModel::Pinhole: {
    folder = folder + "pinhole/";
    break;
  }
  }

  if (translation) {
    folder = folder + "translation/";
  } else {
    folder = folder + "no_translation/";
  }

  if (experiment_type != simulation::anisotropy) {
    switch (noise_type) {
    case simulation::isotropic_homogenous:
      folder = folder + "isotropic_homogeneous/";
      break;
    case simulation::isotropic_inhomogenous:
      folder = folder + "isotropic_inhomogeneous/";
      break;
    case simulation::anisotropic_homogenous:
      folder = folder + "anisotropic_homogeneous/";
      break;
    case simulation::anisotropic_inhomogenous:
      folder = folder + "anisotropic_inhomogeneous/";
      break;
    }
  }

  std::vector<double> endings = simulation::common::linspace(
      linspace_start, linspace_end, linspace_number);

  std::vector<std::string> str_endings;
  for (const auto &ending : endings) {
    str_endings.push_back(std::to_string(ending));
  }

  int seed = 1;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);

  for (const auto &ending : str_endings) {
    std::string experiment_folder = folder + ending;
    BOOST_LOG_TRIVIAL(info) << "Working on folder: " << experiment_folder;

    std::vector<simulation::common::LoadedExperiment> experiments;
    simulation::common::ReadExperiments(experiment_folder, experiments,
                                        generator, uniform01, camera_model,
                                        pnec::common::Target, init_scaling);

    int counter = 1;
    size_t num_experiments = experiments.size();

    std::unordered_map<std::string, simulation::common::Results> results;
    if (em_methods) {
      std::vector<std::string> methods = {"8pt", "7pt", "Stewenius5pt",
                                          "Nister5pt"};
      for (const auto &method : methods) {
        results[method] = simulation::common::Results(num_experiments);
      }
    }
    if (ablation_methods) {
      std::vector<std::string> methods = {"NEC",         "NEC-LS",
                                          "NEC PNEC-LS", "PNEC only LS",
                                          "PNEC w/o LS", "PNEC"};
      for (const auto &method : methods) {
        results[method] = simulation::common::Results(num_experiments);
      }
    } else {
      results["PNEC"] = simulation::common::Results(num_experiments);
    }

    for (const auto &experiment : experiments) {
      if (counter % int(num_experiments / 10) == 0) {
        BOOST_LOG_TRIVIAL(info) << "Finished with experiment " << counter;
      }

      if (em_methods) {
        EMMethods(results, experiment);
      }
      if (ablation_methods) {
        Ablation(results, experiment);
      } else {
        PNEC(results, experiment);
      }

      counter++;
    }

    std::string save_folder = experiment_folder + "/" + name;
    if (!boost::filesystem::is_directory(save_folder)) {
      boost::filesystem::create_directory(save_folder);
    }

    std::string r_error_filename = save_folder + "/" + "r_error.csv";
    std::string t_error_filename = save_folder + "/" + "t_error.csv";
    std::string costs_filename = save_folder + "/" + "cost.csv";

    std::ofstream r_error_outfile;
    std::ofstream t_error_outfile;
    std::ofstream costs_outfile;

    r_error_outfile.open(r_error_filename);
    t_error_outfile.open(t_error_filename);
    costs_outfile.open(costs_filename);
    r_error_outfile << "index";
    t_error_outfile << "index";
    costs_outfile << "index";
    for (std::unordered_map<std::string, simulation::common::Results>::iterator
             it = results.begin();
         it != results.end(); ++it) {
      r_error_outfile << "," << it->first;
      t_error_outfile << "," << it->first;
      costs_outfile << "," << it->first;
    }
    r_error_outfile << std::endl;
    t_error_outfile << std::endl;
    costs_outfile << std::endl;
    for (size_t i = 0; i < num_experiments; i++) {
      r_error_outfile << i;
      t_error_outfile << i;
      costs_outfile << i;

      for (std::unordered_map<std::string,
                              simulation::common::Results>::iterator it =
               results.begin();
           it != results.end(); ++it) {
        r_error_outfile << "," << results[it->first].rotational_error_[i];
        t_error_outfile << "," << results[it->first].translational_error_[i];
        costs_outfile << "," << results[it->first].costs_[i];
      }
      r_error_outfile << std::endl;
      t_error_outfile << std::endl;
      costs_outfile << std::endl;
    }
    r_error_outfile.close();
    t_error_outfile.close();
    costs_outfile.close();
  }

  return 0;
}
