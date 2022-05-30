/**
 BSD 3-Clause License

 This file is part of the PNEC project.
 https://github.com/tum-vision/pnec://github.com/tum-vision/pnec

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
#include <iostream>
#include <string.h>
#include <unordered_map>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <sophus/se3.hpp>

#include "dataset_loader.h"

#include "base_frame.h"
#include "base_matcher.h"
#include "config.h"
#include "frame2frame.h"
#include "frame_processing.h"
#include "klt_patch_optical_flow.h"
#include "nec_ceres.h"
#include "odometry_output.h"
#include "pnec_ceres.h"
#include "timing.h"
#include "tracking_frame.h"
#include "tracking_matcher.h"
#include "view_graph.h"

void log_init() {
  boost::log::add_console_log(std::cout, boost::log::keywords::format =
                                             "[%Severity%] %Message%");
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::info);
}

int main(int argc, const char *argv[]) {
  // TODO: Check Frame2Frame
  // TODO: Update ORBExtractor
  // TODO: Compile Options for ORB/KLT
  // TODO: Logging
  // TODO: Check if all things are timed
  log_init();

  std::string licence_notice =
      "PNEC Copyright (C) 2022 Dominik Muhle\n"
      "PNEC comes with ABSOLUTELY NO WARRANTY.\n"
      "    This is free software, and you are welcome to redistribute it\n"
      "    under certain conditions; visit\n"
      "    https://github.com/tum-vision/pnec for details.\n"
      "\n";

  const cv::String keys =
      "{help h usage ?   |      | print this message                           "
      "         }"
      "{@camera_config   |<none>| tracking                                     "
      "         }"
      "{@pnec_config     |<none>| config file                                  "
      "         }"
      "{@tracking_calib  |<none>| config file                                  "
      "         }"
      "{@tracking_config |<none>| tracking config file                         "
      "         }"
      "{@sequence_path   |<none>| path to images                               "
      "         }"
      "{@timestamp_path  |<none>| path to timestamps                           "
      "         }"
      "{@results         |<none>| path to store results                        "
      "         }"
      "{@visualization   |<none>| path to store visualization frames           "
      "         }"
      "{@no_skip         |<none>| don't skip any frames                        "
      "         }"
      "{image_ext        |.png  | image extension                              "
      "         }"
      "{gt               |      | ground truth                                 "
      "         }"

      ;

  std::cout << licence_notice << std::endl;

  std::cout << "OpenCV version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION
            << "." << CV_SUBMINOR_VERSION << std::endl;

  std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "."
            << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "."
            << CV_SUBMINOR_VERSION << std::endl;

  cv::CommandLineParser parser(argc, argv, keys);

  size_t parser_counter = 0;

  const std::string camera_config_filename(
      parser.get<cv::String>(parser_counter++));
  const std::string pnec_config_filename(
      parser.get<cv::String>(parser_counter++));
  const std::string tracking_config_filename(
      parser.get<cv::String>(parser_counter++));
  const std::string tracking_calib_filename(
      parser.get<cv::String>(parser_counter++));
  const std::string sequence_path(parser.get<cv::String>(parser_counter++));
  const std::string timestamp_path(parser.get<cv::String>(parser_counter++));
  const std::string results_path(parser.get<cv::String>(parser_counter++));
  const std::string visualization_path(
      parser.get<cv::String>(parser_counter++));
  if (!boost::filesystem::exists(results_path)) {
    boost::filesystem::create_directory(results_path);
  }
  const bool no_skip(parser.get<bool>(parser_counter++));
  const std::string image_ext(parser.get<cv::String>("image_ext"));
  bool gt_provided = parser.has("gt");
  std::string gt_path;
  if (gt_provided) {
    gt_path = parser.get<cv::String>("gt");
  }

  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  // load configurations
  pnec::CameraParameters cam_parameters =
      pnec::input::LoadCameraConfig(camera_config_filename);
  pnec::rel_pose_estimation::Options pnec_options =
      pnec::input::LoadPNECConfig(pnec_config_filename);
  basalt::Calibration<double> tracking_calib =
      pnec::input::LoadTrackingCalib(tracking_calib_filename);
  basalt::VioConfig tracking_config =
      pnec::input::LoadTrackingConfig(tracking_config_filename);

  BOOST_LOG_TRIVIAL(info) << "K:\n" << cam_parameters.intrinsic();

  BOOST_LOG_TRIVIAL(info) << "dist coefs:\n"
                          << cam_parameters.dist_coef() << std::endl;

  BOOST_LOG_TRIVIAL(info) << "using image extension: " << image_ext
                          << std::endl;
  pnec::input::DatasetLoader loader(sequence_path, image_ext, timestamp_path,
                                    false, 1.0);

  // get ground truth if provided
  std::vector<Sophus::SE3d> gt_poses;
  std::vector<Sophus::SE3d> rel_gt_poses;
  if (gt_provided) {
    pnec::input::LoadGroundTruth(gt_path, gt_poses, rel_gt_poses);
  }

  /* =============================== TRACKING =============================== */
  basalt::KLTPatchOpticalFlow<float, basalt::Pattern52> tracking(
      tracking_config, tracking_calib, true, false);

  /* =============================== MATCHER ================================ */
  pnec::features::BaseMatcher::Ptr matcher;
  pnec::features::TrackingMatcher::Ptr tracking_matcher =
      std::make_shared<pnec::features::TrackingMatcher>(50, 5.0);
  matcher = tracking_matcher;

  /* ========================= REL POSE ESTIMATION ========================== */
  pnec::rel_pose_estimation::Frame2Frame::Ptr rel_pose_estimation;
  rel_pose_estimation =
      std::make_shared<pnec::rel_pose_estimation::Frame2Frame>(pnec_options);

  pnec::Camera::instance().init(cam_parameters);
  pnec::common::Timing timing;

  /* ============================== VIEW GRAPH ============================== */
  int max_graph_size = 20;
  pnec::odometry::ViewGraph::Ptr view_graph(
      new pnec::odometry::ViewGraph(max_graph_size, results_path));

  /* =========================== FRAME PROCESSING =========================== */
  pnec::visualization::Options vis_options(
      visualization_path, pnec::visualization::Options::NO,
      pnec::visualization::Options::INLIER);
  pnec::odometry::FrameProcessing frame_processor(
      view_graph, rel_pose_estimation, matcher, no_skip, vis_options);

  std::vector<int> selected_frames;

  int count = 0;
  const int sampling_step = 1; // 5

  for (auto image : loader) {
    if (count++ % sampling_step != 0) // sampling
    {
      continue;
    }

    auto tic = std::chrono::high_resolution_clock::now(),
         toc = std::chrono::high_resolution_clock::now();

    double timestamp = image.timestamp_;
    std::string impath = image.path_.string();

    BOOST_LOG_TRIVIAL(debug) << "Processing Image: " << image.id_;
    BOOST_LOG_TRIVIAL(debug)
        << "Loading image from " << impath << " with timestamp " << timestamp;

    pnec::common::FrameTiming frame_timing(image.id_);
    // Create a Frame object
    pnec::frames::BaseFrame::Ptr f;
    f.reset(new pnec::frames::TrackingFrame(image.id_, timestamp, impath,
                                            tracking, frame_timing));
    toc = std::chrono::high_resolution_clock::now();

    frame_timing.feature_creation_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic) -
        frame_timing.frame_loading_;

    bool frame_is_selected =
        frame_processor.ProcessFrame(f, frame_timing, results_path);

    if (!frame_is_selected) {
      BOOST_LOG_TRIVIAL(debug) << "Skipping frame " << image.id_;
      continue;
    }

    selected_frames.push_back(count);

    timing.push_back(frame_timing);
    BOOST_LOG_TRIVIAL(info)
        << "Processed frame " << f->id() << " in " << frame_timing.TotalTime()
        << " miliseconds, not including ablation methods";
  }

  std::map<std::string, std::vector<std::pair<double, Sophus::SE3d>>>
      ablation_pose = rel_pose_estimation->GetAblationResults();
  for (auto const &[name, poses] : ablation_pose) {
    std::ios_base::openmode openmode = std::ios_base::out;
    BOOST_LOG_TRIVIAL(info) << "Saving poses for " << name;
    for (const auto &timestamped_pose : poses) {
      pnec::out::SavePose(results_path + "ablation/", name,
                          timestamped_pose.first, timestamped_pose.second,
                          openmode);
      openmode = std::ios_base::app;
    }
  }

  std::ofstream out;
  out.open(results_path + "timing.txt", std::ios_base::trunc);

  out << timing;

  BOOST_LOG_TRIVIAL(info) << "Saving VO poses";
  view_graph->savePoses();
  return 0;
}