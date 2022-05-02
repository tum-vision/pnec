#include "config.h"

#include <boost/filesystem.hpp>
#include <iostream>

#include <basalt/calibration/calibration.hpp>
#include <basalt/utils/vio_config.h>
#include <basalt/serialization/headers_serialization.h>
#include <opencv2/core/core.hpp>

#include <camera.h>

namespace pnec {
namespace input {

// TODO: ceres options
basalt::Calibration<double> LoadTrackingCalib(const std::string &calib_path) {
  std::ifstream os(calib_path, std::ios::binary);

  basalt::Calibration<double> tracking_calib;
  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(tracking_calib);
    std::cout << "Loaded camera with " << tracking_calib.intrinsics.size()
              << " cameras" << std::endl;
  } else {
    std::cerr << "could not load camera calibration " << calib_path
              << std::endl;
    std::abort();
  }
  return tracking_calib;
}

basalt::VioConfig LoadTrackingConfig(const std::string &config_path) {
  basalt::VioConfig tracking_config;
  tracking_config.load(config_path);
  return tracking_config;
}

pnec::rel_pose_estimation::Options
LoadPNECConfig(const std::string &config_path) {
  // check settings file
  cv::FileStorage settings(config_path.c_str(), cv::FileStorage::READ);
  if (!settings.isOpened()) {
    std::cerr << "Failed to open settings file: " << config_path << std::endl;
    std::exit(-1);
  }

  std::string noise_frame_str = settings["PNEC.noiseFrame"];
  pnec::common::NoiseFrame noise_frame = pnec::common::Target;
  if (noise_frame_str == "Host") {
    noise_frame = pnec::common::Host;
  }
  if (noise_frame_str == "Target") {
    noise_frame = pnec::common::Target;
  }
  const int nec_int = settings["PNEC.NEC"];
  const int max_weighted_it = settings["PNEC.weightedIterations"];
  const int scf_int = settings["PNEC.SCF"];
  const int ceres_int = settings["PNEC.ceres"];
  const int ransac_int = settings["PNEC.ransac"];

  pnec::rel_pose_estimation::Options options;
  options.use_nec_ = (nec_int == 0) ? false : true;

  options.noise_frame_ = noise_frame;
  options.regularization_ = settings["PNEC.regularization"];

  options.weighted_iterations_ = settings["PNEC.weightedIterations"];
  options.use_scf_ = (scf_int == 0) ? false : true;

  options.use_ceres_ = (ceres_int == 0) ? false : true;
  options.ceres_options_ = ceres::Solver::Options();

  options.use_ransac_ = (ransac_int == 0) ? false : true;
  options.max_ransac_iterations_ = settings["PNEC.maxRANSACIterations"];
  options.ransac_sample_size_ = settings["PNEC.RANSACSampleSize"];

  return options;
}

pnec::CameraParameters LoadCameraConfig(const std::string &config_path) {
  // check settings file
  cv::FileStorage settings(config_path.c_str(), cv::FileStorage::READ);
  if (!settings.isOpened()) {
    std::cerr << "Failed to open settings file: " << config_path << std::endl;
    std::exit(-1);
  }

  cv::Matx33d K = cv::Matx33d::eye();

  const double fx = settings["Camera.fx"];
  const double fy = settings["Camera.fy"];
  const double cx = settings["Camera.cx"];
  const double cy = settings["Camera.cy"];

  K(0, 0) = fx;
  K(1, 1) = fy;
  K(0, 2) = cx;
  K(1, 2) = cy;

  cv::Vec4d dist_coef;

  dist_coef(0) = settings["Camera.k1"];
  dist_coef(1) = settings["Camera.k2"];
  dist_coef(2) = settings["Camera.p1"];
  dist_coef(3) = settings["Camera.p2"];

  return pnec::CameraParameters(K, dist_coef);
}
} // namespace input
} // namespace pnec