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
#include <string.h>

#include "anisotropy_experiments.h"
#include "common.h"
#include "experiments.h"
#include "offset_experiments.h"
#include "outlier_experiments.h"
#include "sim_common.h"
#include "standard_experiments.h"

void log_init() {
  boost::log::add_console_log(std::cout, boost::log::keywords::format =
                                             "[%Severity%] %Message%");
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::debug);
}

int main(int argc, const char *argv[]) {
  log_init();

  const cv::String keys =
      "{help h usage ?    |          | print this message    }"
      "{@base_folder      |none      | base folder name}"
      "{@experiment_type  |none      | create anisotropy experiment}"
      "{@camera_model     |none      | pinhole or omnidirectional camera}"
      "{@translation      |true      | translation or not}"
      "{@noise_type       |none      | noise_type}"
      "{n num_experiments |10000     | number of experiments}"
      "{s seed            |1         | random seed}"
      "{ls linspace_start |0.125     | start of linspace for endings}"
      "{le linspace_end   |4.000     | end of linspace for endings}"
      "{ln linspace_number|32        | start of linspace for endings}";
  ;

  cv::CommandLineParser parser(argc, argv, keys);

  int parse_counter = 0;
  std::string base_folder(parser.get<cv::String>(parse_counter++));
  const int exp_num(parser.get<int>(parse_counter++));
  const int camera_model_num(parser.get<int>(parse_counter++));
  const bool translation(parser.get<bool>(parse_counter++));
  const int noise_int(parser.get<int>(parse_counter++));
  const int num_experiments(parser.get<int>("num_experiments"));
  const int seed(parser.get<int>("seed"));
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
  if (!boost::filesystem::is_directory(folder)) {
    boost::filesystem::create_directory(folder);
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
  if (!boost::filesystem::is_directory(folder)) {
    boost::filesystem::create_directory(folder);
  }

  if (translation) {
    folder = folder + "translation/";
  } else {
    folder = folder + "no_translation/";
  }
  if (!boost::filesystem::is_directory(folder)) {
    boost::filesystem::create_directory(folder);
  }

  if (experiment_type != simulation::anisotropy) {
    switch (noise_type) {
    case simulation::isotropic_homogenous:
      folder = folder + "isotropic_homogeneous";
      break;
    case simulation::isotropic_inhomogenous:
      folder = folder + "isotropic_inhomogeneous";
      break;
    case simulation::anisotropic_homogenous:
      folder = folder + "anisotropic_homogeneous";
      break;
    case simulation::anisotropic_inhomogenous:
      folder = folder + "anisotropic_inhomogeneous";
      break;
    }
    if (!boost::filesystem::is_directory(folder)) {
      boost::filesystem::create_directory(folder);
    }
  }

  std::vector<double> levels = simulation::common::linspace(
      linspace_start, linspace_end, linspace_number);
  switch (experiment_type) {
  case simulation::ExperimentType::standard: {
    simulation::experiments::StandardExperiments standard_experiment(
        camera_model, noise_type, translation, pnec::common::NoiseFrame::Target,
        seed);

    standard_experiment.GenerateExperiments(folder, num_experiments, 10,
                                            levels);
    break;
  }
  case simulation::ExperimentType::anisotropy: {
    simulation::experiments::AnisotropyExperiments anisotropy_experiments(
        camera_model, translation, pnec::common::NoiseFrame::Target, seed);

    anisotropy_experiments.GenerateExperiments(folder, num_experiments, 10,
                                               levels);
    break;
  }
  case simulation::ExperimentType::outlier: {
    simulation::experiments::OutlierExperiments outlier_experiment(
        camera_model, noise_type, translation, pnec::common::NoiseFrame::Target,
        seed);

    outlier_experiment.GenerateExperiments(folder, num_experiments, 50, levels);
    break;
  }
  case simulation::ExperimentType::offset: {
    simulation::experiments::OffsetExperiments offset_experiment(
        camera_model, noise_type, translation, pnec::common::NoiseFrame::Target,
        seed);

    offset_experiment.GenerateExperiments(folder, num_experiments, 10, levels);
    break;
  }
  }
  return 0;
}
