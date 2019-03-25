/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_SYCL

#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace tensorflow {

void SYCLUtil::copyCPUTensorToDevice(const Eigen::SyclDevice& sycl_device,
                                     const Tensor& cpu_tensor,
                                     Tensor& device_tensor,
                                     StatusCallback done) {
  const int64 total_bytes = cpu_tensor.TotalBytes();
  const void *src_ptr = GetBase(&cpu_tensor);
  void *dst_ptr = GetBase(&device_tensor);
#ifdef EIGEN_SYCL_ASYNC_EXECUTION
  TensorReference input_ref(cpu_tensor);
  auto callback = [done, input_ref]() {
    input_ref.Unref();
    done(Status::OK());
  };
#else
  // This will make the copy blocking
  auto callback = std::function<void()>();
#endif
  sycl_device.memcpyHostToDevice(dst_ptr, src_ptr, total_bytes,
                                 std::move(callback));
#ifndef EIGEN_SYCL_ASYNC_EXECUTION
  done(Status::OK());
#endif
}

void SYCLUtil::copyDeviceTensorToCPU(const Eigen::SyclDevice& sycl_device,
                                     const Tensor& device_tensor,
                                     Tensor& cpu_tensor,
                                     StatusCallback done) {
  const int64 total_bytes = device_tensor.TotalBytes();
  const void *src_ptr = GetBase(&device_tensor);
  void *dst_ptr = GetBase(&cpu_tensor);
#ifdef EIGEN_SYCL_ASYNC_EXECUTION
  TensorReference input_ref(device_tensor);
  // For now this copy has to be blocking no matter what, later cone_copy
  // could be removed.
  Notification done_copy;
  auto callback = [&done_copy, done, input_ref]() {
    input_ref.Unref();
    done(Status::OK());
    done_copy.Notify();
  };
#else
  // This will make the copy blocking
  auto callback = std::function<void()>();
#endif
  sycl_device.memcpyDeviceToHost(dst_ptr, src_ptr, total_bytes,
                                 std::move(callback));
#ifdef EIGEN_SYCL_ASYNC_EXECUTION
  done_copy.WaitForNotification();
#else
  done(Status::OK());
#endif
}

void SYCLUtil::copyDeviceTensorToDevice(const Eigen::SyclDevice& sycl_device,
                                        const Tensor& src_tensor,
                                        Tensor& dst_tensor) {
  const int64 total_bytes = src_tensor.TotalBytes();
  const void* src_ptr = GetBase(&src_tensor);
  void* dst_ptr = GetBase(&dst_tensor);
  sycl_device.memcpy(dst_ptr, src_ptr, total_bytes);
}

}  // namespace tensorflow
#endif  // TENSORFLOW_USE_SYCL
