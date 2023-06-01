#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <vector>

#include <nec_ceres.h>
#include <pnec_ceres.h>

#include "base_frame.h"
#include "base_matcher.h"
#include "config.h"
#include "tracking_frame.h"
#include "tracking_matcher.h"
#include "visualization.h"

namespace py = pybind11;

// #define STRINGIFY(x) #x
// #define MACRO_STRINGIFY(x) STRINGIFY(x)

void log_init() {
  boost::log::add_console_log(std::cout, boost::log::keywords::format =
                                             "[%Severity%] %Message%");
  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::warning);
}

int add(int i, int j) { return i + j; }

Eigen::MatrixXd mat2(Eigen::MatrixXd &mat) {
  std::cout << mat << std::endl;
  return mat * 2;
}

std::vector<Eigen::MatrixXd> matrices(std::vector<Eigen::MatrixXd> mats) {
  std::vector<Eigen::MatrixXd> results;
  for (const auto &mat : mats) {
    std::cout << mat << std::endl;
    results.push_back(mat * 2);
  }
  return results;
}

Eigen::Matrix4d pyceres(std::vector<Eigen::Vector3d> host_bvs,
                        std::vector<Eigen::Vector3d> target_bvs,
                        std::vector<Eigen::Matrix3d> host_covariances,
                        std::vector<Eigen::Matrix3d> target_covariances,
                        Eigen::Matrix4d init_pose, double regularization) {
  pnec::optimization::PNECCeres optimizer;
  Sophus::SE3d sp_init_pose(Eigen::Quaterniond(init_pose.block<3, 3>(0, 0))
                                .normalized()
                                .toRotationMatrix(),
                            init_pose.block<3, 1>(0, 3));
  optimizer.InitValues(Eigen::Quaterniond(sp_init_pose.rotationMatrix()),
                       sp_init_pose.translation());
  optimizer.Optimize(host_bvs, target_bvs, host_covariances, target_covariances,
                     regularization);

  return optimizer.Result().matrix();
}

Eigen::Matrix4d pyceresnec(std::vector<Eigen::Vector3d> host_bvs,
                           std::vector<Eigen::Vector3d> target_bvs,
                           Eigen::Matrix4d init_pose) {
  pnec::optimization::NECCeres optimizer;
  // Sophus::SE3d sp_init_pose(init_pose);
  Sophus::SE3d sp_init_pose(Eigen::Quaterniond(init_pose.block<3, 3>(0, 0))
                                .normalized()
                                .toRotationMatrix(),
                            init_pose.block<3, 1>(0, 3));
  optimizer.InitValues(Eigen::Quaterniond(sp_init_pose.rotationMatrix()),
                       sp_init_pose.translation());
  optimizer.Optimize(host_bvs, target_bvs);

  return optimizer.Result().matrix();
}

void pyKLTMatching(std::string img_1_path, std::string img_2_path,
                   std::vector<Eigen::Vector2d> &host_kps,
                   std::vector<Eigen::Vector2d> &target_kps,
                   bool verbose = false) {
  log_init();
  pnec::features::BaseMatcher::Ptr matcher;
  pnec::features::TrackingMatcher::Ptr tracking_matcher =
      std::make_shared<pnec::features::TrackingMatcher>(50, 5.0);
  matcher = tracking_matcher;

  basalt::Calibration<double> tracking_calib = pnec::input::LoadTrackingCalib(
      "/usr/wiss/muhled/Documents/projects/pnec/pnec/data/tracking/KITTI/"
      "kitti_calib.json");
  basalt::VioConfig tracking_config =
      pnec::input::LoadTrackingConfig("/usr/wiss/muhled/Documents/projects/"
                                      "pnec/pnec/data/tracking/KITTI/00.json");

  basalt::KLTPatchOpticalFlow<float, basalt::Pattern52> tracking(
      tracking_config, tracking_calib, true, false);

  pnec::frames::BaseFrame::Ptr host_f;
  pnec::common::FrameTiming prev_timing(0);
  host_f.reset(new pnec::frames::TrackingFrame(0, 0.0, img_1_path, tracking,
                                               prev_timing));

  pnec::common::FrameTiming frame_timing(1);
  pnec::frames::BaseFrame::Ptr target_f;
  target_f.reset(new pnec::frames::TrackingFrame(1, 1.0, img_2_path, tracking,
                                                 frame_timing));

  bool skipping_frame;
  pnec::FeatureMatches matches =
      matcher->FindMatches(host_f, target_f, skipping_frame);
  if (verbose) {
    std::cout << "Found " << matches.size() << " matches" << std::endl;
  }

  std::vector<size_t> host_matches;
  std::vector<size_t> target_matches;
  for (const auto &match : matches) {
    host_matches.push_back(match.queryIdx);
    target_matches.push_back(match.trainIdx);
  }
  pnec::features::KeyPoints host_keypoints = host_f->keypoints(host_matches);
  pnec::features::KeyPoints target_keypoints =
      target_f->keypoints(target_matches);

  host_kps.clear();
  target_kps.clear();
  for (const auto &keypoint : host_keypoints) {
    host_kps.push_back(keypoint.second.point_);
  }
  for (const auto &keypoint : target_keypoints) {
    target_kps.push_back(keypoint.second.point_);
  }
}

void pyKLTImageMatching(py::array_t<uint8_t> &img1, py::array_t<uint8_t> &img2,
                        std::vector<Eigen::Vector2d> &host_kps,
                        std::vector<Eigen::Vector2d> &target_kps,
                        Eigen::Vector2d offset = Eigen::Vector2d(0.0, 0.0),
                        bool verbose = false) {
  log_init();

  auto rows1 = img1.shape(0);
  auto cols1 = img1.shape(1);
  // std::cout << rows1 << " " << cols1 << std::endl;
  auto type = CV_8UC1;
  cv::Mat cvimg1(rows1, cols1, type, (unsigned char *)img1.data());
  auto rows2 = img2.shape(0);
  auto cols2 = img2.shape(1);
  // std::cout << rows2 << " " << cols2 << std::endl;
  cv::Mat cvimg2(rows2, cols2, type, (unsigned char *)img2.data());

  pnec::features::BaseMatcher::Ptr matcher;
  pnec::features::TrackingMatcher::Ptr tracking_matcher =
      std::make_shared<pnec::features::TrackingMatcher>(50, 5.0);
  matcher = tracking_matcher;

  basalt::Calibration<double> tracking_calib = pnec::input::LoadTrackingCalib(
      "/usr/wiss/muhled/Documents/projects/pnec/pnec/data/tracking/KITTI/"
      "kitti_calib.json");
  basalt::VioConfig tracking_config =
      pnec::input::LoadTrackingConfig("/usr/wiss/muhled/Documents/projects/"
                                      "pnec/pnec/data/tracking/KITTI/00.json");

  basalt::KLTPatchOpticalFlow<float, basalt::Pattern52> tracking(
      tracking_config, tracking_calib, true, false);

  pnec::frames::BaseFrame::Ptr host_f;
  pnec::common::FrameTiming prev_timing(0);
  host_f.reset(new pnec::frames::TrackingFrame(cvimg1, tracking));
  // std::cout << "Host frame" << std::endl;

  pnec::common::FrameTiming frame_timing(1);
  pnec::frames::BaseFrame::Ptr target_f;
  target_f.reset(new pnec::frames::TrackingFrame(cvimg2, tracking, offset));
  // std::cout << "Target frame" << std::endl;

  bool skipping_frame;
  pnec::FeatureMatches matches =
      matcher->FindMatches(host_f, target_f, skipping_frame);
  if (verbose) {
    std::cout << "Found " << matches.size() << " matches" << std::endl;
  }

  std::vector<size_t> host_matches;
  std::vector<size_t> target_matches;
  for (const auto &match : matches) {
    host_matches.push_back(match.queryIdx);
    target_matches.push_back(match.trainIdx);
  }
  pnec::features::KeyPoints host_keypoints = host_f->keypoints(host_matches);
  pnec::features::KeyPoints target_keypoints =
      target_f->keypoints(target_matches);

  host_kps.clear();
  target_kps.clear();
  for (const auto &keypoint : host_keypoints) {
    host_kps.push_back(keypoint.second.point_);
  }
  for (const auto &keypoint : target_keypoints) {
    target_kps.push_back(keypoint.second.point_);
  }

  if (verbose) {
    std::vector<int> inliers;
    char key = pnec::visualization::plotMatches(
        host_f, cvimg1, target_f, cvimg2, matches, inliers,
        pnec::visualization::Options(
            "/storage/user/muhled/outputs/images/",
            pnec::visualization::Options::VisualizationLevel::TRACKED,
            pnec::visualization::Options::VisualizationLevel::NO),
        "pypnec");
  }
}

void cpp_img(std::string img_path, py::array_t<uint8_t> &img) {
  cv::Mat img1 = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

  // auto im = img.unchecked<3>();
  auto rows = img.shape(0);
  auto cols = img.shape(1);
  auto type = CV_8UC1;

  cv::Mat img2(rows, cols, type, (unsigned char *)img.data());

  cv::Mat total_img;

  std::cout << img1.dims << " " << img1.channels() << " " << img1.rows << " "
            << img1.cols << std::endl;
  std::cout << img2.dims << " " << img2.channels() << " " << img2.rows << " "
            << img2.cols << std::endl;
  cv::hconcat(img1, img2, total_img);
  cv::imwrite("test.png", total_img);
  return;
}

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2d>);

PYBIND11_MODULE(pypnec, m) {
  py::bind_vector<std::vector<Eigen::Vector2d>>(m, "Keypoints");

  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("add", &add, "A function that adds two numbers");
  m.def("mat2", &mat2, "A function prints out and multiplies by two");
  m.def("matrices", &matrices, "A function prints out and multiplies by two");
  m.def("pyceres", &pyceres, "A function prints out and multiplies by two");
  m.def("pyceresnec", &pyceresnec,
        "A function prints out and multiplies by two");
  m.def("cppimg", &cpp_img, "Test Function for loading images to cpp");
  m.def("KLTMatching", &pyKLTMatching, "Match with KLT Tracker");
  m.def("KLTImageMatching", &pyKLTImageMatching, "Match with KLT Tracker");
}