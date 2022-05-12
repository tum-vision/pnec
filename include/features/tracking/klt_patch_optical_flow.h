/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
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

// There are changes compared to the original file at
// https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/include/basalt/optical_flow/patch_optical_flow.h

#pragma once

#include <iostream>
#include <thread>

#include <sophus/se2.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <basalt/optical_flow/optical_flow.h>
// #include <basalt/optical_flow/patch.h>
#include "pnec_patch.h"

#include <basalt/image/image_pyr.h>
#include <basalt/utils/keypoints.h>

namespace basalt {

template <typename Scalar, template <typename> typename Pattern>
class KLTPatchOpticalFlow : public OpticalFlowBase {
public:
  typedef POpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Sophus::SE2<Scalar> SE2;

  KLTPatchOpticalFlow(const VioConfig &config,
                      const basalt::Calibration<double> &calib,
                      const bool use_mahalanobis = true,
                      const bool numerical_cov = true)
      : t_ns(-1), frame_counter(0), use_mahalanobis_{use_mahalanobis},
        numerical_cov_(numerical_cov), last_keypoint_id(0), config(config),
        calib(calib) {
    patches.reserve(3000);
    input_queue.set_capacity(10);

    patch_coord = PatchT::pattern2.template cast<float>();

    if (/*calib.intrinsics.size()*/ 1 > 1) {
      Sophus::SE3d T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[1];
      computeEssential(T_i_j, E);
    }
  }

  ~KLTPatchOpticalFlow() {}

  void DeleteOldKeypoints() {
    for (const auto &id : deleteKeypoints) {
      if (patches.find(id) != patches.end()) {
        patches.erase(id);
      } else {
        std::cout << "couldn't delete Keypoint" << std::endl;
      }
    }
    deleteKeypoints.clear();
  }

  OpticalFlowResult::Ptr processFrame(int64_t curr_t_ns,
                                      OpticalFlowInput::Ptr &new_img_vec) {
    for (const auto &v : new_img_vec->img_data) {
      if (!v.img.get())
        return transforms;
    }

    int n_tracked_points = 0;
    if (t_ns < 0) {
      t_ns = curr_t_ns;

      transforms.reset(new OpticalFlowResult);
      std::cout << /*calib.intrinsics.size()*/ 1 << std::endl;
      transforms->observations.resize(/*calib.intrinsics.size()*/ 1);
      transforms->t_ns = t_ns;
      covariances.resize(1);

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<u_int16_t>>);
      pyramid->resize(/*calib.intrinsics.size()*/ 1);
      for (size_t i = 0; i < /*calib.intrinsics.size()*/ 1; i++) {
        pyramid->at(i).setFromImage(*new_img_vec->img_data[i].img,
                                    config.optical_flow_levels);
      }

      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();
    } else {
      t_ns = curr_t_ns;

      old_pyramid = pyramid;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<u_int16_t>>);
      pyramid->resize(/*calib.intrinsics.size()*/ 1);
      for (size_t i = 0; i < /*calib.intrinsics.size()*/ 1; i++) {
        pyramid->at(i).setFromImage(*new_img_vec->img_data[i].img,
                                    config.optical_flow_levels);
      }

      OpticalFlowResult::Ptr new_transforms;
      new_transforms.reset(new OpticalFlowResult);
      new_transforms->observations.resize(new_img_vec->img_data.size());
      new_transforms->t_ns = t_ns;

      for (size_t i = 0; i < /*calib.intrinsics.size()*/ 1; i++) {
        trackPoints(old_pyramid->at(i), pyramid->at(i),
                    transforms->observations[i],
                    new_transforms->observations[i], covariances[i]);
      }

      transforms = new_transforms;
      n_tracked_points = transforms->observations[0].size();
      std::cout << "Tracked " << n_tracked_points
                << " patches from previous frame." << std::endl;
      transforms->input_images = new_img_vec;

      addPoints();
      filterPoints();
    }
    frame_counter++;
    return transforms;
  }

  int PatchesSize() { return patches.size(); }

  void trackPoints(
      const basalt::ManagedImagePyr<u_int16_t> &pyr_1,
      const basalt::ManagedImagePyr<u_int16_t> &pyr_2,
      const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>
          &transform_map_1,
      Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> &transform_map_2,
      Eigen::aligned_map<KeypointId, Matrix2> &covariance_map) {
    size_t num_points = transform_map_1.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);

    for (const auto &kv : transform_map_1) {
      ids.push_back(kv.first);
      init_vec.push_back(kv.second);
    }

    tbb::concurrent_unordered_map<KeypointId, Eigen::AffineCompact2f,
                                  std::hash<KeypointId>>
        result;
    tbb::concurrent_unordered_map<KeypointId, Matrix2, std::hash<KeypointId>>
        result_cov;

    auto compute_func = [&](const tbb::blocked_range<size_t> &range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];

        const Eigen::AffineCompact2f &transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = transform_1;

        const Eigen::aligned_vector<PatchT> &patch_vec = patches.at(id);

        bool valid = trackPoint(pyr_2, patch_vec, transform_2, id, 0);

        if (valid) {
          // forward_tracking[r] = 1;
          Eigen::AffineCompact2f transform_1_recovered = transform_2;
          valid = trackPoint(pyr_1, patch_vec, transform_1_recovered, id, 1);

          if (valid) {
            // backward_tracking[r] = 1;
            Scalar dist2;
            if (use_mahalanobis_) {
              dist2 = (transform_1.translation() -
                       transform_1_recovered.translation())
                          .transpose() *
                      transform_1.rotation() *
                      patch_vec[min_level].Cov.inverse() *
                      transform_1.rotation().transpose() *
                      (transform_1.translation() -
                       transform_1_recovered.translation());
            } else {
              dist2 = (transform_1.translation() -
                       transform_1_recovered.translation())
                          .squaredNorm();
            }

            if (dist2 < config.optical_flow_max_recovered_dist2) {
              // distance_tracking[r] = 1;
              result[id] = transform_2;
              if (numerical_cov_) {
                typename PatchT::VectorP res;
                bool valid_res;
                PatchT dp = patch_vec[min_level];
                typename PatchT::Matrix2P transformed_pat =
                    transform_2.linear().matrix() * PatchT::pattern2;
                transformed_pat.colwise() += transform_2.translation();
                typename PatchT::Matrix2P offset_transformed_pat;

                valid_res =
                    dp.residual(pyr_2.lvl(min_level), transformed_pat, res);
                Scalar e_0 = res.mean();
                offset_transformed_pat = transformed_pat;
                offset_transformed_pat.colwise() += Eigen::Vector2f(1.0, 0.0);
                valid_res = dp.residual(pyr_2.lvl(min_level),
                                        offset_transformed_pat, res);
                Scalar e_pos_x = res.transpose() * res;
                offset_transformed_pat = transformed_pat;
                offset_transformed_pat.colwise() += Eigen::Vector2f(-1.0, 0.0);
                valid_res = dp.residual(pyr_2.lvl(min_level),
                                        offset_transformed_pat, res);
                Scalar e_neg_x = res.transpose() * res;
                offset_transformed_pat = transformed_pat;
                offset_transformed_pat.colwise() += Eigen::Vector2f(0.0, 1.0);
                valid_res = dp.residual(pyr_2.lvl(min_level),
                                        offset_transformed_pat, res);
                Scalar e_pos_y = res.transpose() * res;
                offset_transformed_pat = transformed_pat;
                offset_transformed_pat.colwise() += Eigen::Vector2f(0.0, -1.0);
                valid_res = dp.residual(pyr_2.lvl(min_level),
                                        offset_transformed_pat, res);
                Scalar e_neg_y = res.transpose() * res;
                offset_transformed_pat = transformed_pat;
                offset_transformed_pat.colwise() += Eigen::Vector2f(1.0, 1.0);
                valid_res = dp.residual(pyr_2.lvl(min_level),
                                        offset_transformed_pat, res);
                Scalar e_pos_x_pos_y = res.transpose() * res;
                offset_transformed_pat = transformed_pat;
                offset_transformed_pat.colwise() += Eigen::Vector2f(1.0, -1.0);
                valid_res = dp.residual(pyr_2.lvl(min_level),
                                        offset_transformed_pat, res);
                Scalar e_pos_x_neg_y = res.transpose() * res;
                offset_transformed_pat = transformed_pat;
                offset_transformed_pat.colwise() += Eigen::Vector2f(-1.0, 1.0);
                valid_res = dp.residual(pyr_2.lvl(min_level),
                                        offset_transformed_pat, res);
                Scalar e_neg_x_pos_y = res.transpose() * res;
                offset_transformed_pat = transformed_pat;
                offset_transformed_pat.colwise() += Eigen::Vector2f(-1.0, 1.0);
                valid_res = dp.residual(pyr_2.lvl(min_level),
                                        offset_transformed_pat, res);
                Scalar e_neg_x_neg_y = res.transpose() * res;

                Matrix2 H;
                H(0, 0) = e_pos_x - 2 * e_0 + e_neg_x;
                H(1, 0) = (e_pos_x_pos_y - e_pos_x_neg_y - e_neg_x_pos_y +
                           e_neg_x_neg_y) /
                          4.0;
                H(0, 1) = H(1, 0);
                H(1, 1) = e_pos_y - 2 * e_0 + e_neg_y;
                result_cov[id] = H.inverse() * 52.0;
              } else {
                result_cov[id] = transform_2.rotation() *
                                 (patch_vec[min_level].Cov / 10.0) *
                                 transform_2.rotation()
                                     .transpose(); // factor for trace norming
              }
            } else {
              // remove id from patches map
              valid = false;
            }
          }
        }

        // remove id from patches to save memory footprint
        if (!valid) {
          deleteKeypoints.push_back(id);
        }
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);

    tbb::parallel_for(range, compute_func);
    // compute_func(range);

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
    if (covariance_map.size() != 0) {
      covariance_map.clear();
    }
    covariance_map.insert(result_cov.begin(), result_cov.end());
  }

  inline bool trackPoint(const basalt::ManagedImagePyr<uint16_t> &pyr,
                         const Eigen::aligned_vector<PatchT> &patch_vec,
                         Eigen::AffineCompact2f &transform, KeypointId id,
                         int backward) const {
    bool patch_valid = true;

    for (int level = config.optical_flow_levels;
         level >= min_level && patch_valid; level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      // Perform tracking on current level
      patch_valid &= trackPointAtLevel(pyr.lvl(level), patch_vec[level],
                                       transform, id, scale, backward);

      transform.translation() *= scale;
    }

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const u_int16_t> &img_2,
                                const PatchT &dp,
                                Eigen::AffineCompact2f &transform,
                                KeypointId id, Scalar scale,
                                int backward) const {
    bool patch_valid = true;

    for (int iteration = 0;
         patch_valid && iteration < config.optical_flow_max_iterations;
         iteration++) {
      typename PatchT::VectorP res;

      typename PatchT::Matrix2P transformed_pat =
          transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      bool valid = dp.residual(img_2, transformed_pat, res);

      if (valid) {
        Vector3 inc = -dp.H_se2_inv_J_se2_T * res;
        transform *= SE2::exp(inc).matrix();

        const int filter_margin = 2;

        if (!img_2.InBounds(transform.translation(), filter_margin))
          patch_valid = false;
      } else {
        patch_valid = false;
      }
    }

    return patch_valid;
  }

  void addPoints() {
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    for (const auto &kv : transforms->observations.at(0)) {
      pts0.emplace_back(kv.second.translation().cast<double>());
    }

    KeypointsData kd;
    Scalar scaling = (1 << min_level);

    detectKeypoints(pyramid->at(0).lvl(min_level), kd,
                    config.optical_flow_detection_grid_size, 1, pts0);

    Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> new_poses0,
        new_poses1;

    for (size_t i = 0; i < kd.corners.size(); i++) {
      kd.corners[i] *= scaling;
      Eigen::aligned_vector<PatchT> &p = patches[last_keypoint_id];

      Vector2 pos = kd.corners[i].cast<Scalar>();

      for (int l = 0; l <= config.optical_flow_levels; l++) {
        Scalar scale = 1 << l;
        Vector2 pos_scaled = pos / scale;
        p.emplace_back(pyramid->at(0).lvl(l), pos_scaled);
      }

      Eigen::AffineCompact2f transform;
      transform.setIdentity();
      transform.translation() = kd.corners[i].cast<Scalar>();

      transforms->observations.at(0)[last_keypoint_id] = transform;
      new_poses0[last_keypoint_id] = transform;

      last_keypoint_id++;
    }

    if (/*calib.intrinsics.size()*/ 1 > 1) {
      trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1,
                  covariances[0]);

      for (const auto &kv : new_poses1) {
        transforms->observations.at(1).emplace(kv);
      }
    }
  }

  void filterPoints() {
    if (/*calib.intrinsics.size()*/ 1 < 2)
      return;

    std::set<KeypointId> lm_to_remove;

    std::vector<KeypointId> kpid;
    Eigen::aligned_vector<Eigen::Vector2d> proj0, proj1;

    for (const auto &kv : transforms->observations.at(1)) {
      auto it = transforms->observations.at(0).find(kv.first);

      if (it != transforms->observations.at(0).end()) {
        proj0.emplace_back(it->second.translation().cast<double>());
        proj1.emplace_back(kv.second.translation().cast<double>());
        kpid.emplace_back(kv.first);
      }
    }

    Eigen::aligned_vector<Eigen::Vector4d> p3d0, p3d1;
    std::vector<bool> p3d0_success, p3d1_success;

    calib.intrinsics[0].unproject(proj0, p3d0, p3d0_success);
    calib.intrinsics[1].unproject(proj1, p3d1, p3d1_success);

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) {
        const double epipolar_error =
            std::abs(p3d0[i].transpose() * E * p3d1[i]);

        if (epipolar_error > config.optical_flow_epipolar_error) {
          lm_to_remove.emplace(kpid[i]);
        }
      } else {
        lm_to_remove.emplace(kpid[i]);
      }
    }

    for (int id : lm_to_remove) {
      transforms->observations.at(1).erase(id);
    }
  }

  std::vector<Eigen::aligned_map<KeypointId, Matrix2>> Covariances() const {
    return covariances;
  }

  OpticalFlowResult::Ptr Transforms() const { return transforms; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  std::vector<KeypointId> deleteKeypoints;

  int64_t t_ns;

  std::string filename = "translation_diffs.csv";

  size_t frame_counter;

  int min_level = 0;

  bool use_mahalanobis_;

  bool numerical_cov_;

  KeypointId last_keypoint_id;

  VioConfig config;
  basalt::Calibration<double> calib;

  Eigen::aligned_unordered_map<KeypointId, Eigen::aligned_vector<PatchT>>
      patches;

  OpticalFlowResult::Ptr transforms;
  std::vector<Eigen::aligned_map<KeypointId, Matrix2>> covariances;

  std::shared_ptr<std::vector<basalt::ManagedImagePyr<u_int16_t>>> old_pyramid,
      pyramid;

  Eigen::Matrix4d E;
};

} // namespace basalt
