/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_USE_SYCL
#error This file must only be included when building TensorFlow with SYCL support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_SYCL_DNN_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_SYCL_DNN_UTILS_H_

#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/sycl_blas_utils.h"

#if defined(SYCL_SNN_USE_EIGEN_BACKEND)
#include "sycldnn/backend/eigen_backend.h"
#else
#include "sycldnn/backend/sycl_blas_backend.h"
#endif

#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/depthwise_conv2d/params.h"
#include "sycldnn/pooling/params.h"

namespace tensorflow {

// Backends cannot be copied or moved so they are constructed in a macro
#if defined(SYCL_SNN_USE_EIGEN_BACKEND)
#define CREATE_SNN_BACKEND(BACKEND, DEVICE) \
  sycldnn::backend::EigenBackend BACKEND(DEVICE)
#else
#define CREATE_SNN_BACKEND(BACKEND, DEVICE) \
  sycldnn::backend::SyclBLASBackend BACKEND(DEVICE.sycl_queue())
#endif

template <class SDStatus>
inline Status get_sd_err_msg(const SDStatus& s) {
  return errors::Internal("Internal error from SYCL-DNN code " +
      std::to_string(static_cast<int>(s.status)));
}

inline void vlog_conv2d_params(const sycldnn::conv2d::Conv2DParams& sd_params,
                               TensorFormat data_format,
                               const std::string& type) {
  VLOG(1) << "[SYCL-DNN] thread=" << std::this_thread::get_id()
          << " type=" << type
          << " TF_data_format=" << data_format
          << " channels=" << sd_params.channels
          << " features=" << sd_params.features
          << " batch=" << sd_params.batch
          << " in_rows=" << sd_params.in_rows
          << " in_cols=" << sd_params.in_cols
          << " window_rows=" << sd_params.window_rows
          << " window_cols=" << sd_params.window_cols
          << " stride_rows=" << sd_params.stride_rows
          << " stride_cols=" << sd_params.stride_cols
          << " out_rows=" << sd_params.out_rows
          << " out_cols=" << sd_params.out_cols
          << " pad_rows=" << sd_params.pad_rows
          << " pad_cols=" << sd_params.pad_cols
          << " dilation_rows=" << sd_params.dilation_rows
          << " dilation_cols=" << sd_params.dilation_cols
          << std::endl;
}

inline void vlog_depthwise_params(
    const sycldnn::depthwise_conv2d::DepthwiseConv2DParams& sd_params,
    TensorFormat data_format, const std::string& type) {
  VLOG(1) << "[SYCL-DNN] thread=" << std::this_thread::get_id()
          << " type=" << type
          << " TF_data_format=" << data_format
          << " channels=" << sd_params.channels
          << " channel_multiplier=" << sd_params.channel_multiplier
          << " batch=" << sd_params.batch
          << " in_rows=" << sd_params.in_rows
          << " in_cols=" << sd_params.in_cols
          << " window_rows=" << sd_params.window_rows
          << " window_cols=" << sd_params.window_cols
          << " stride_rows=" << sd_params.stride_rows
          << " stride_cols=" << sd_params.stride_cols
          << " out_rows=" << sd_params.out_rows
          << " out_cols=" << sd_params.out_cols
          << " pad_rows=" << sd_params.pad_rows
          << " pad_cols=" << sd_params.pad_cols
          << std::endl;
}

inline void vlog_pooling_params(const sycldnn::pooling::PoolingParams& sd_params,
                                const std::string& type,
                                bool propagate_nan = false) {
  VLOG(1) << "[SYCL-DNN] thread=" << std::this_thread::get_id()
          << " type=" << type
          << " propagate_nan=" << propagate_nan
          << " channels=" << sd_params.channels
          << " batch=" << sd_params.batch
          << " in_rows=" << sd_params.in_rows
          << " in_cols=" << sd_params.in_cols
          << " window_rows=" << sd_params.window_rows
          << " window_cols=" << sd_params.window_cols
          << " stride_rows=" << sd_params.stride_rows
          << " stride_cols=" << sd_params.stride_cols
          << " out_rows=" << sd_params.out_rows
          << " out_cols=" << sd_params.out_cols
          << " pad_rows=" << sd_params.pad_rows
          << " pad_cols=" << sd_params.pad_cols
          << std::endl;
}

}  // tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SYCL_DNN_UTILS_H_
