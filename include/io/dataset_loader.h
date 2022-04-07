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

#ifndef IO_DATASET_LOADER_H_
#define IO_DATASET_LOADER_H_

#include <boost/filesystem.hpp>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <sophus/se3.hpp>

namespace pnec {
namespace input {

class DatasetLoader {
public:
  struct Image {
    Image(double timestamp, boost::filesystem::path path)
        : timestamp_(timestamp), path_(path) {}
    ~Image() {}

    // overloaded < operator
    bool operator<(const Image &img) {
      if (timestamp_ < img.timestamp_) {
        return true;
      }
      return false;
    }
    int id_;
    double timestamp_;
    boost::filesystem::path path_;
  };

  DatasetLoader(std::string img_path, std::string im_ext,
                std::string times_path = "", bool has_index = false,
                double timescale = 1.0);

  std::vector<DatasetLoader::Image>::const_iterator begin() const {
    return images_.begin();
  }

  std::vector<Image>::const_iterator end() const { return images_.end(); }

private:
  void LoadTogether(std::string img_path, std::string im_ext, double timescale);
  void LoadSeperately(std::string img_path, std::string im_ext,
                      std::string times_path, bool has_index, double timescale);

  std::vector<Image> images_;
};

bool LoadGroundTruth(std::string gt_path, std::vector<Sophus::SE3d> &gt_poses,
                     std::vector<Sophus::SE3d> &rel_gt_poses);

} // namespace input
} // namespace pnec

#endif // IO_DATASET_LOADER_H_
