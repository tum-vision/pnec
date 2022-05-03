#ifndef IO_CONFIG_H_
#define IO_CONFIG_H_

#include <string.h>

#include <basalt/calibration/calibration.hpp>
#include <basalt/utils/vio_config.h>

#include "camera.h"
#include "pnec_config.h"

namespace pnec {
namespace input {
basalt::Calibration<double> LoadTrackingCalib(const std::string &calib_path);

basalt::VioConfig LoadTrackingConfig(const std::string &config_path);

pnec::rel_pose_estimation::Options
LoadPNECConfig(const std::string &config_path);

pnec::CameraParameters LoadCameraConfig(const std::string &config_path);

} // namespace input
} // namespace pnec

#endif // IO_CONFIG_H_
