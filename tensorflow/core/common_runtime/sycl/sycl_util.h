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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/notification.h"

namespace tensorflow {

inline void const* GetBase(const Tensor* src) { return DMAHelper::base(src); }
inline void* GetBase(Tensor* dst) { return DMAHelper::base(dst); }

class SYCLUtil {
 public:
  static void copyCPUTensorToDevice(
      const Eigen::SyclDevice& sycl_device, const Tensor& cpu_tensor,
      Tensor& device_tensor, StatusCallback done = [](const Status&) {});

  static void copyDeviceTensorToCPU(
      const Eigen::SyclDevice& sycl_device, const Tensor& device_tensor,
      Tensor& cpu_tensor, StatusCallback done = [](const Status&) {});

  static void copyDeviceTensorToDevice(const Eigen::SyclDevice& sycl_device,
                                       const Tensor& src_tensor,
                                       Tensor& dst_tensor);

  static inline void blockingCopyCPUTensorToDevice(
      const Eigen::SyclDevice& sycl_device, const Tensor& cpu_tensor,
      Tensor& device_tensor) {
    Notification done_copy;
    SYCLUtil::copyCPUTensorToDevice(sycl_device, cpu_tensor, device_tensor,
        [&done_copy](const Status& s) { done_copy.Notify(); });
    done_copy.WaitForNotification();
  }

  static inline void blockingCopyDeviceTensorToCPU(
      const Eigen::SyclDevice& sycl_device, const Tensor& device_tensor,
      Tensor& cpu_tensor) {
    Notification done_copy;
    SYCLUtil::copyDeviceTensorToCPU(sycl_device, device_tensor, cpu_tensor,
        [&done_copy](const Status& s) { done_copy.Notify(); });
    done_copy.WaitForNotification();
  }

  static inline void copyCPUTensorToDevice(
      Device* device, const Tensor& cpu_tensor, Tensor& device_tensor,
      StatusCallback done = [](const Status&) {}) {
    copyCPUTensorToDevice(*device->eigen_sycl_device(), cpu_tensor,
                          device_tensor, done);
  }

  static inline void copyDeviceTensorToCPU(
      Device* device, const Tensor& device_tensor, Tensor& cpu_tensor,
      StatusCallback done = [](const Status&) {}) {
    copyDeviceTensorToCPU(*device->eigen_sycl_device(), device_tensor,
                          cpu_tensor, done);
  }

  static inline void copyDeviceTensorToDevice(Device *device,
                                              const Tensor& src_tensor,
                                              Tensor& dst_tensor) {
    copyDeviceTensorToDevice(*device->eigen_sycl_device(), src_tensor,
                             dst_tensor);
  }

  static inline void blockingCopyCPUTensorToDevice(Device* device,
                                                   const Tensor& cpu_tensor,
                                                   Tensor& device_tensor) {
    blockingCopyCPUTensorToDevice(*device->eigen_sycl_device(), cpu_tensor,
                                  device_tensor);
  }

  static inline void blockingCopyDeviceTensorToCPU(Device* device,
                                                   const Tensor& device_tensor,
                                                   Tensor& cpu_tensor) {
    blockingCopyDeviceTensorToCPU(*device->eigen_sycl_device(), device_tensor,
                                  cpu_tensor);
  }

  static inline const cl::sycl::id<3>
    get_max_work_item_tuple(const Eigen::SyclDevice& d) {
    const auto& device = d.sycl_queue().get_device();
    return device.template get_info<cl::sycl::info::device::max_work_item_sizes>();
  }

  template <class T>
  static inline cl::sycl::nd_range<1> get_nd_range(const Eigen::SyclDevice& d,
                                                   const T items) {
    const size_t nb_items = static_cast<size_t>(items);
    const size_t group_size = std::min(nb_items,
        SYCLUtil::get_max_work_item_tuple(d)[0]);
    const size_t group_count = (nb_items + group_size - 1) / group_size;

    return cl::sycl::nd_range<1>(cl::sycl::range<1>(group_count * group_size),
                                 cl::sycl::range<1>(group_size));
  }

  template <class T>
  static inline cl::sycl::nd_range<2> get_nd_range(const Eigen::SyclDevice& d,
                                                   const T item_dim0,
                                                   const T item_dim1) {
    const size_t nb_items = static_cast<size_t>(item_dim0);
    const size_t group_size = std::min(nb_items,
        SYCLUtil::get_max_work_item_tuple(d)[0]);
    const size_t group_count = (nb_items + group_size - 1) / group_size;

    return cl::sycl::nd_range<2>(
        cl::sycl::range<2>(group_count * group_size, item_dim1),
        cl::sycl::range<2>(group_size, 1));
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_UTIL_H_
