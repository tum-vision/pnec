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

#ifndef COMMON_CAMERA_H_
#define COMMON_CAMERA_H_

#include <opencv2/opencv.hpp>

namespace pnec {
class CameraParameters {
public:
  CameraParameters()
      : intrisics_(cv::Matx33d::eye()), distortion_coef_(cv::Vec4d::all(0)) {}

  CameraParameters(cv::Matx33d K) : intrisics_(K) {}

  CameraParameters(cv::Matx33d K, cv::Vec4d dist_coef)
      : intrisics_(K), distortion_coef_(dist_coef) {}

  const cv::Matx33d &intrinsic() const { return intrisics_; }
  const cv::Vec4d &dist_coef() const { return distortion_coef_; }

private:
  cv::Matx33d intrisics_;
  cv::Vec4d distortion_coef_;
};

// singleton
// https://stackoverflow.com/questions/1008019/c-singleton-design-pattern

class Camera {
public:
  Camera(Camera const &) = delete;

  void operator=(Camera const &) = delete;

  static Camera &instance() {
    static Camera instance;
    return instance;
  }

  void init(CameraParameters camera_parameters);

  const CameraParameters cameraParameters() const { return cam_parameters_; }

private:
  Camera() {}

  CameraParameters cam_parameters_;
};

} // namespace pnec

#endif // COMMON_CAMERA_H_
