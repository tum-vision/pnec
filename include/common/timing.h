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

#ifndef COMMON_TIMING_H_
#define COMMON_TIMING_H_

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace pnec {
namespace common {

struct FrameTiming {
  FrameTiming(int id) : id_(id) {}
  ~FrameTiming() {}

  static std::string TimingHeader() {
    return "ID FrameLoading FeatureCreation NEC-ES IT-ES AVG-IT-ES "
           "CERES OPTIMIZATION TOTAL";
  }

  int OptimizationTime() const;
  int TotalTime() const;

  int id_;
  std::chrono::milliseconds frame_loading_ = std::chrono::milliseconds(0);
  std::chrono::milliseconds feature_creation_ = std::chrono::milliseconds(0);
  std::chrono::milliseconds nec_es_ = std::chrono::milliseconds(0);
  std::chrono::milliseconds it_es_ = std::chrono::milliseconds(0);
  std::chrono::milliseconds avg_it_es_ = std::chrono::milliseconds(0);
  std::chrono::milliseconds ceres_ = std::chrono::milliseconds(0);
};

std::ostream &operator<<(std::ostream &os, const FrameTiming &frame_timing);

class Timing {
public:
  void push_back(const FrameTiming &frame_timing) {
    frame_timings_.push_back(frame_timing);
  }

  friend std::ostream &operator<<(std::ostream &os, const Timing timing);

private:
  std::vector<FrameTiming> frame_timings_;
};

std::ostream &operator<<(std::ostream &os, const Timing timing);

} // namespace common
} // namespace pnec

#endif // COMMON_TIMING_H_