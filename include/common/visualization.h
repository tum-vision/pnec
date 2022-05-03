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

#ifndef COMMON_VISUALIZATION_H_
#define COMMON_VISUALIZATION_H_

#include <string.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "base_frame.h"
#include "common.h"

namespace pnec {
namespace visualization {
cv::RotatedRect GetErrorEllipse(double chisquare_val, cv::Point2f mean,
                                cv::Mat covmat);

char plotMatches(pnec::frames::BaseFrame::Ptr prev_frame,
                 pnec::frames::BaseFrame::Ptr curr_frame,
                 pnec::FeatureMatches &matches, std::vector<int> &inliers,
                 std::string suffix);

char plotCovariancess(pnec::frames::BaseFrame::Ptr curr_frame,
                      pnec::FeatureMatches &matches, std::vector<int> &inliers,
                      std::string suffix);
} // namespace visualization
} // namespace pnec

#endif // COMMON_VISUALIZATION_H_
