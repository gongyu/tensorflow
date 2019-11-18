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

#ifndef TENSORFLOW_CORE_KERNELS_SYCL_BLAS_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_SYCL_BLAS_UTILS_H_

#include <utility>
#include <thread>
#include <cstdlib>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// Workaround to use Eigen's PointerMapper instead of including one from a
// public repository
namespace cl {
namespace sycl {
namespace codeplay {
class PointerMapper : public Eigen::TensorSycl::internal::PointerMapper {};
inline void *SYCLmalloc(size_t size, PointerMapper &pMap) {
  return Eigen::TensorSycl::internal::SYCLmalloc(size, pMap);
}
template <bool ReUse = true, typename PointerMapper>
inline void SYCLfree(void *ptr, PointerMapper &pMap) {
  Eigen::TensorSycl::internal::SYCLfree<ReUse>(ptr, pMap);
}
template <typename PointerMapper>
inline void SYCLfreeAll(PointerMapper &pMap) {
  Eigen::TensorSycl::internal::SYCLfreeAll(pMap);
}
}
}
}

#define SYCL_BLAS_ALWAYS_INLINE 1
#include "sycl_blas.hpp"

#include "tensorflow/core/platform/default/logging.h"

namespace tensorflow {

using SYCLBlasPolicy = blas::PolicyHandler<blas::codeplay_policy>;
using SYCLBlasExecutor = blas::Executor<SYCLBlasPolicy>;

template <class T>
inline blas::BufferIterator<T, blas::codeplay_policy>
    get_buffer_iterator(const Eigen::SyclDevice& d, const T* ptr) {
  auto original_buffer = d.get_sycl_buffer(ptr);
  auto offset = d.get_offset(ptr);
  auto buffer = original_buffer.template reinterpret<T>(
      cl::sycl::range<1>(original_buffer.get_size() / sizeof(T)));
  return blas::BufferIterator<T, blas::codeplay_policy>(buffer, offset);
}

template <typename IndexT>
inline void vlog_blas_params(const std::string& type, IndexT m, IndexT n,
                             IndexT k, char t_x, char t_y,
                             IndexT batch=IndexT(1)) {
  VLOG(1) << "[SYCL-BLAS] thread=" << std::this_thread::get_id()
          << " type=" << type
          << " batch=" << batch
          << " m=" << m
          << " n=" << n
          << " k=" << k
          << " t_x=" << t_x
          << " t_y=" << t_y
          << std::endl;
}

}  // tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SYCL_BLAS_UTILS_H_
