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
  TensorReference input_ref(cpu_tensor);
  sycl_device.memcpyHostToDevice(dst_ptr, src_ptr, total_bytes,
                                 [done, input_ref]() {
    input_ref.Unref();
    done(Status::OK());
  });
}

void SYCLUtil::copyDeviceTensorToCPU(const Eigen::SyclDevice& sycl_device,
                                     const Tensor& device_tensor,
                                     Tensor& cpu_tensor,
                                     StatusCallback done) {
  const int64 total_bytes = device_tensor.TotalBytes();
  const void *src_ptr = GetBase(&device_tensor);
  void *dst_ptr = GetBase(&cpu_tensor);
  TensorReference input_ref(device_tensor);
  sycl_device.memcpyDeviceToHost(dst_ptr, src_ptr, total_bytes,
                                 [done, input_ref]() {
    input_ref.Unref();
    done(Status::OK());
  });
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
