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

#include "tracking_frame.h"

#include <ctime>

#include "camera.h"
#include "converter.h"
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

namespace pnec {
namespace frames {
void TrackingFrame::FindFeatures(pnec::common::FrameTiming &frame_timing) {
  auto tic = std::chrono::high_resolution_clock::now(),
       toc = std::chrono::high_resolution_clock::now();
  cv::Mat image = getImage();

  basalt::OpticalFlowInput::Ptr img_ptr =
      pnec::converter::OpticalFlowFromOpenCV(image, id_);

  toc = std::chrono::high_resolution_clock::now();
  frame_timing.frame_loading_ =
      std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

  basalt::OpticalFlowResult::Ptr result = tracking_.processFrame(id_, img_ptr);

  pnec::converter::KeypointsFromOpticalFlow(result, keypoints_, keypoint_ids_);

 
  auto result_cov = tracking_.Covariances();
  auto result_hessian = tracking_.Hessians();

  if (result_cov.size() != 0) {
    for (const auto &covariance : result_cov) {
      covariances_.push_back(covariance.second.cast<double>());
    }
  }
  if (result_hessian.size() != 0) {
    for (const auto &hessian : result_hessian) {
      hessians_.push_back(hessian.second.cast<double>());
    }
  }
  std::cout << "extracted " << std::endl << keypoints_.size() << " keypoints" << 
  std::endl << covariances_.size() << " covariances" << std::endl << 
  hessians_.size() << " hessians" << std::endl;

  // tracking_.DeleteOldKeypoints();
}
} // namespace frames
} // namespace pnec
