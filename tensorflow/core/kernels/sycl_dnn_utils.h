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
//TODO(codeplay): remove later
#include <cstdlib>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/sycl_blas_utils.h"

#if defined(SYCL_SNN_USE_EIGEN_BACKEND)
#include "sycldnn/backend/eigen_backend.h"
#else // Use SYCL-BLAS backend
#define SYCL_SNN_USE_BLAS_BACKEND 1
#include "sycldnn/backend/sycl_blas_backend.h"
#endif

namespace tensorflow {

template <class SDStatus>
inline Status get_sd_err_msg(const SDStatus& s) {
  return errors::Internal("Internal error from SYCL-DNN code " +
      std::to_string(static_cast<int>(s.status)));
}

//TODO(codeplay): remove later
inline bool is_snn_enabled() {
  static const char* use_snn_cstr = std::getenv("TF_SYCL_USE_SNN");
  static bool use_snn = use_snn_cstr == nullptr || std::string(use_snn_cstr) != "0";
  return use_snn;
}

#if defined(SYCL_SNN_USE_BLAS_BACKEND)
#define CREATE_SNN_BACKEND(BACKEND, DEVICE) \
  sycldnn::backend::SyclBLASBackend BACKEND(DEVICE.sycl_queue())
#else // Use Eigen backend
#define CREATE_SNN_BACKEND(BACKEND, DEVICE) \
  sycldnn::backend::EigenBackend BACKEND(DEVICE)
#endif

}  // tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SYCL_DNN_UTILS_H_
