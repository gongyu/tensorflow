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

#ifndef TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_

// Functor definitions for ScatterND ops, must be compilable by nvcc.

#define EIGEN_USE_THREADS

#include <atomic>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/scatter_nd_op.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/util.h"

#ifdef TENSORFLOW_USE_SYCL
#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#endif  // TENSORFLOW_USE_SYCL

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

class OpKernelContext;

// Specialization of UpdateExecutor to CPU
namespace update_executor {

template <typename Input, typename Update, typename Output,
          scatter_nd_op::UpdateOp OP>
class UpdateExecutor {
 public:
  EIGEN_STRONG_INLINE static void Execute(Input value, Update update,
                                          Output output);
};

template <typename Input, typename Update, typename Output>
class UpdateExecutor<Input, Update, Output, scatter_nd_op::UpdateOp::ASSIGN> {
 public:
  EIGEN_STRONG_INLINE static void Execute(Input /* input */, Update update,
                                          Output output) {
    output = update;
  }
};

template <typename Input, typename Update, typename Output>
class UpdateExecutor<Input, Update, Output, scatter_nd_op::UpdateOp::ADD> {
 public:
  EIGEN_STRONG_INLINE static void Execute(Input /* input */, Update update,
                                          Output output) {
    output += update;
  }
};

template <typename Input, typename Update, typename Output>
class UpdateExecutor<Input, Update, Output, scatter_nd_op::UpdateOp::SUB> {
 public:
  EIGEN_STRONG_INLINE static void Execute(Input /* input */, Update update,
                                          Output output) {
    output -= update;
  }
};

}  // namespace update_executor

namespace functor {

// Implementation of update functor for CPU.
template <typename T, typename Index, scatter_nd_op::UpdateOp OP, int IXDIM>
struct ScatterNdFunctor<CPUDevice, T, Index, OP, IXDIM> {
  Index operator()(
      const CPUDevice& d, const Index slice_size,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      typename TTypes<T, 2>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, 2>::Tensor Toutput) {
    // error_loc is -1 if there's no out-of-bounds index,
    // otherwise it is the location of an OOB index in Tindices.
    Index error_loc = -1;

    const Eigen::DenseIndex batch_size = Tindices.dimension(0);

    Index batch_strides[IXDIM];
    for (int dim = IXDIM - 1; dim >= 0; --dim) {
      if (dim == IXDIM - 1) {
        batch_strides[dim] = 1;
      } else {
        batch_strides[dim] =
            batch_strides[dim + 1] * output_shape_prefix[dim + 1];
      }
    }

    for (Eigen::DenseIndex loc = 0; loc < batch_size; ++loc) {
      Index i = 0;
      bool out_of_bounds = false;
      for (int dim = 0; dim < IXDIM; ++dim) {
        const Index ix_d = internal::SubtleMustCopy(Tindices(loc, dim));
        out_of_bounds |= !FastBoundsCheck(ix_d, output_shape_prefix[dim]);
        i += ix_d * batch_strides[dim];
      }
      if (TF_PREDICT_FALSE(out_of_bounds)) {
        error_loc = loc;
        break;
      } else {
        auto input_chip = Toutput.template chip<0>(i);
        auto output_chip = input_chip.device(d);
        auto update_chip = Tupdates.template chip<0>(loc);
        update_executor::UpdateExecutor<
            decltype(input_chip), decltype(update_chip), decltype(output_chip),
            OP>::Execute(input_chip, update_chip, output_chip);
      }
    }

    return error_loc;
  }
};

#define REGISTER_SCATTER_ND_FULL(T, Index, op)                               \
  template Index                                                             \
  ScatterNdFunctor<CPUDevice, T, Index, op, CPU_PROVIDED_IXDIM>::operator()( \
      const CPUDevice& d, const Index slice_size,                            \
      const Eigen::array<Eigen::DenseIndex, CPU_PROVIDED_IXDIM>              \
          output_shape_prefix,                                               \
      typename TTypes<T, 2>::Tensor Tparams,                                 \
      typename TTypes<Index, 2>::ConstTensor Tindices,                       \
      typename TTypes<T, 2>::ConstTensor Tupdates,                           \
      typename TTypes<T, 2>::Tensor Toutput)

#define REGISTER_SCATTER_ND_INDEX(type, op)  \
  REGISTER_SCATTER_ND_FULL(type, int32, op); \
  REGISTER_SCATTER_ND_FULL(type, int64, op)

#define REGISTER_SCATTER_ND_UPDATE(type) \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::ASSIGN);

#define REGISTER_SCATTER_ND_MATH(type)                           \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::ADD); \
  REGISTER_SCATTER_ND_INDEX(type, scatter_nd_op::UpdateOp::SUB);

TF_CALL_ALL_TYPES(REGISTER_SCATTER_ND_UPDATE);
REGISTER_SCATTER_ND_INDEX(string, scatter_nd_op::UpdateOp::ADD);
TF_CALL_NUMBER_TYPES(REGISTER_SCATTER_ND_MATH)

#undef REGISTER_SCATTER_ND_MATH
#undef REGISTER_SCATTER_ND_UPDATE
#undef REGISTER_SCATTER_ND_INDEX
#undef REGISTER_SCATTER_ND_FULL

#ifdef TENSORFLOW_USE_SYCL
namespace {
template <typename PTR, typename T, scatter_nd_op::UpdateOp Op>
struct LeftUpdateSYCL {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(PTR out, const T& val);
};

template <typename PTR, typename T>
struct LeftUpdateSYCL<PTR, T, scatter_nd_op::UpdateOp::ASSIGN> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(PTR out, const T& val) {
    *out = val;
  }
};

template <typename PTR, typename T>
struct LeftUpdateSYCL<PTR, T, scatter_nd_op::UpdateOp::ADD> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(PTR out, const T& val) {
    *out += val;
  }
};

template <typename PTR, typename T>
struct LeftUpdateSYCL<PTR, T, scatter_nd_op::UpdateOp::SUB> {
  EIGEN_STRONG_INLINE EIGEN_DEVICE_FUNC void operator()(PTR out, const T& val) {
    *out -= val;
  }
};
}  // namespace

template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
struct ScatterNdKernel {
  using write_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>;
  using read_accessor =
      cl::sycl::accessor<uint8_t, 1, cl::sycl::access::mode::read,
                         cl::sycl::access::target::global_buffer>;

  ScatterNdKernel(
      const read_accessor indices, const read_accessor updates,
      write_accessor out,
      const Index batch_size, const Index slice_size,
      const unsigned int out_size,
      const cl::sycl::cl_int batch_strides[IXDIM])
      : indices_(indices),
        updates_(updates),
        out_(out),
        batch_size_(batch_size),
        slice_size_(slice_size),
        out_size_(out_size)
        {
          for (int i = 0; i < IXDIM; ++i)
            batch_strides_[i] = batch_strides[i];
        }

  void operator()(cl::sycl::nd_item<1> item) {
    const auto curr_item = item.get_global_id(0);
    if (curr_item >= slice_size_)
      return;

    const T* updates = ConvertToActualTypeSycl(T, updates_);
    const Index* indices = ConvertToActualTypeSycl(Index, indices_);
    T* out = ConvertToActualTypeSycl(T, out_);

    auto update_op = LeftUpdateSYCL<decltype(out), T, op>();

    // Iterate through every index and update the curr_item corresponding to that slice
    for (int idx = 0; idx < batch_size_; ++idx)
    {
      // update_idx is the index of the element that needs to be changed in out
      auto update_idx = 0;
      for (int i = 0; i < IXDIM; ++i)
        update_idx += indices[(idx * IXDIM) + i] * batch_strides_[i];
      update_idx += curr_item;

      if (update_idx < out_size_)
        update_op(out + update_idx, updates[idx * slice_size_ + curr_item]);
    }
  }

 private:
  const read_accessor indices_;
  const read_accessor updates_;
  write_accessor out_;

  const Index batch_size_;
  const Index slice_size_;
  const unsigned int out_size_;
  cl::sycl::cl_int batch_strides_[IXDIM];
};

// Implementation of update functor for SYCL.
template <typename T, typename Index, scatter_nd_op::UpdateOp OP, int IXDIM>
struct ScatterNdFunctor<SYCLDevice, T, Index, OP, IXDIM> {
  Index operator()(
      const SYCLDevice& d, const Index slice_size,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      typename TTypes<T, 2>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, 2>::Tensor Toutput) {

    const Eigen::DenseIndex batch_size = Tindices.dimension(0);
    const int num_threads = batch_size;

    // batch_strides are the number of dimensions in each rank, is used to know
    // how many elements to skip on each rank when accessing indices
    cl::sycl::cl_int batch_strides[IXDIM];
    for (int dim = 0; dim < IXDIM; ++dim)
    {
      batch_strides[dim] = slice_size;
      for (int i = dim + 1; i < IXDIM; ++i)
        batch_strides[dim] *= output_shape_prefix[i]; 
    }

    auto indices_buffer = d.get_sycl_buffer(Tindices.data());
    auto updates_buffer = d.get_sycl_buffer(Tupdates.data());
    auto output_buffer = d.get_sycl_buffer(Toutput.data());

    d.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto indices_access =
          indices_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
      auto updates_access =
          updates_buffer.template get_access<cl::sycl::access::mode::read>(cgh);

      auto output_access =
          output_buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);

      cl::sycl::nd_range<1> nd_rng = get_sycl_nd_range(d, slice_size);

      ScatterNdKernel<T, Index, OP, IXDIM> kernel(
          indices_access, updates_access, output_access, batch_size,
          slice_size, Toutput.size(), batch_strides);

      cgh.parallel_for(nd_rng, kernel);
    });

    return -1;
  }
};

#define REGISTER_SCATTER_ND_FULL_SYCL(T, Index, op)                           \
  template Index                                                              \
  ScatterNdFunctor<SYCLDevice, T, Index, op, CPU_PROVIDED_IXDIM>::operator()( \
      const SYCLDevice& d, const Index slice_size,                            \
      const Eigen::array<Eigen::DenseIndex, CPU_PROVIDED_IXDIM>               \
          output_shape_prefix,                                                \
      typename TTypes<T, 2>::Tensor Tparams,                                  \
      typename TTypes<Index, 2>::ConstTensor Tindices,                        \
      typename TTypes<T, 2>::ConstTensor Tupdates,                            \
      typename TTypes<T, 2>::Tensor Toutput)

#define REGISTER_SCATTER_ND_INDEX_SYCL(type, op)  \
  REGISTER_SCATTER_ND_FULL_SYCL(type, int32, op); \
  REGISTER_SCATTER_ND_FULL_SYCL(type, int64, op)

#define REGISTER_SCATTER_ND_UPDATE_SYCL(type) \
  REGISTER_SCATTER_ND_INDEX_SYCL(type, scatter_nd_op::UpdateOp::ASSIGN);

#define REGISTER_SCATTER_ND_MATH_SYCL(type)                           \
  REGISTER_SCATTER_ND_INDEX_SYCL(type, scatter_nd_op::UpdateOp::ADD); \
  REGISTER_SCATTER_ND_INDEX_SYCL(type, scatter_nd_op::UpdateOp::SUB);

TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SCATTER_ND_UPDATE_SYCL)
TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SCATTER_ND_MATH_SYCL)

#undef REGISTER_SCATTER_ND_MATH_SYCL
#undef REGISTER_SCATTER_ND_UPDATE_SYCL
#undef REGISTER_SCATTER_ND_INDEX_SYCL
#undef REGISTER_SCATTER_ND_FULL_SYCL

#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCATTER_ND_OP_CPU_IMPL_H_
