/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_WHERE_OP_H_
#define TENSORFLOW_KERNELS_WHERE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

#define TF_CALL_WHERE_GPU_TYPES(m) \
  TF_CALL_int8(m);                 \
  TF_CALL_uint8(m);                \
  TF_CALL_int32(m);                \
  TF_CALL_int64(m);                \
  TF_CALL_float(m);                \
  TF_CALL_double(m);               \
  TF_CALL_complex64(m);            \
  TF_CALL_complex128(m);           \
  TF_CALL_bool(m);

namespace functor {

template <typename Device, typename T, typename TIndex>
struct NumTrue {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const Device& d,
      typename TTypes<T>::ConstFlat input,
      typename TTypes<TIndex>::Scalar num_true);
};

template <typename Device, int NDIM, typename T, typename TIndex>
struct Where {
  // Copies indices of true values in input into output.  The pointer
  // found_true should sit on the host.  Compute should copy the
  // number of true elements found into it.  At the end, if
  //   *found_true != output.dimension(0),
  // then the input may have changed between the initial counting of
  // the true values and the call to Where.
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const Device& d,
      typename TTypes<T, NDIM>::ConstTensor input,
      typename TTypes<int64>::Matrix output, TIndex* found_true);
};

template <typename TIndex, typename T, int NDIM>
Eigen::array<TIndex, NDIM> CalculateStrides(
    typename TTypes<T, NDIM>::ConstTensor input) {
  const Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
  Eigen::array<TIndex, NDIM> strides;
  EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                       static_cast<int>(Eigen::RowMajor)),
                      INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);
  strides[NDIM - 1] = 1;
  for (int i = NDIM - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_WHERE_OP_H_
