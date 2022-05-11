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

#include "dataset_loader.h"

namespace pnec {
namespace input {
DatasetLoader::DatasetLoader(std::string img_path, std::string im_ext,
                             std::string times_path, bool has_index,
                             double timescale) {
  if (times_path == "") {
    std::cout << "Loading images and timestamps from\n"
              << img_path << std::endl;
    LoadTogether(img_path, im_ext, timescale);
  } else {
    std::cout << "Loading images from\n" << img_path << std::endl;
    std::cout << "Loading timestamps from\n" << times_path << std::endl;
    LoadSeperately(img_path, im_ext, times_path, has_index, timescale);
  }

  std::sort(images_.begin(), images_.end(), [](const Image &a, const Image &b) {
    return a.timestamp_ < b.timestamp_;
  });
  int id = 0;
  for (auto &image : images_) {
    image.id_ = id;
    id++;
  }
}

void DatasetLoader::LoadTogether(std::string img_path, std::string im_ext,
                                 double timescale) {

  boost::filesystem::path p(img_path);
  for (auto it = boost::filesystem::directory_iterator(p);
       it != boost::filesystem::directory_iterator(); it++) {
    if (boost::filesystem::is_regular_file(*it) &&
        it->path().extension().string() == im_ext) {

      // skip hidden files
      if (it->path().stem().string()[0] == '.') {
        continue;
      }

      double timestamp = std::stod(it->path().stem().string()) / timescale;
      images_.push_back(Image(timestamp, it->path()));
    }
  }
}

void DatasetLoader::LoadSeperately(std::string img_path, std::string im_ext,
                                   std::string times_path, bool has_index,
                                   double timescale) {
  boost::filesystem::path p(img_path);
  auto it = boost::filesystem::directory_iterator(p);

  std::vector<std::pair<int, boost::filesystem::path>> images;
  for (auto it = boost::filesystem::directory_iterator(p);
       it != boost::filesystem::directory_iterator(); it++) {
    if (boost::filesystem::is_regular_file(*it) &&
        it->path().extension().string() == im_ext) {

      // skip hidden files
      if (it->path().stem().string()[0] == '.') {
        continue;
      }

      int img_name = std::stod(it->path().stem().string());
      images.push_back(std::make_pair(img_name, it->path()));
    }
  }
  std::sort(images.begin(), images.end());

  std::ifstream fTimes;
  fTimes.open(times_path);
  for (const auto &image : images) {
    double timestamp;
    std::string s = "";
    while ((s.empty() || s == "") && !fTimes.eof()) {
      getline(fTimes, s);
    }
    if (fTimes.eof()) {
      break;
    }
    std::stringstream ss;
    ss << s;

    if (has_index) {
      int idx;
      ss >> idx;
    }
    ss >> timestamp;
    timestamp /= timescale;

    images_.push_back(Image(timestamp, image.second));
  }
}

bool LoadGroundTruth(std::string gt_path, std::vector<Sophus::SE3d> &gt_poses,
                     std::vector<Sophus::SE3d> &rel_gt_poses) {
  std::cout << "Loading ground truth from " << gt_path << std::endl;
  std::ifstream myfile(gt_path);
  if (!myfile.is_open()) {
    std::cerr << "Unable to open file " << gt_path << std::endl;
    return false;
  }
  double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;
  while (myfile >> r1 >> r2 >> r3 >> r4 >> r5 >> r6 >> r7 >> r8 >> r9 >> r10 >>
         r11 >> r12) {
    Eigen::Matrix3d R;
    R << r1, r2, r3, r5, r6, r7, r9, r10, r11;
    Eigen::Vector3d t(r4, r8, r12);

    Eigen::Matrix3d fixed_rotation =
        Eigen::Quaterniond(R).normalized().toRotationMatrix();

    gt_poses.push_back(Sophus::SE3d(fixed_rotation, t));
  }
  myfile.close();
  std::cout << "loaded " << gt_poses.size() << " gt poses" << std::endl;

  for (size_t i = 0; i < gt_poses.size() - 1; i++) {
    rel_gt_poses.push_back(gt_poses[i].inverse() * gt_poses[i + 1]);
  }
  std::cout << "computed " << rel_gt_poses.size() << " relative ground truth poses" << std::endl;
  return true;
}
} // namespace input
} // namespace pnec