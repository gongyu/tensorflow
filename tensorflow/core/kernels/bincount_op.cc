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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/bincount_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

using thread::ThreadPool;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace functor {

template <typename T>
struct BincountFunctor<CPUDevice, T> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<int32, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output) {
    int size = output.size();

    Tensor all_nonneg_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_BOOL, TensorShape({}), &all_nonneg_t, AllocatorAttributes()));
    all_nonneg_t.scalar<bool>().device(context->eigen_cpu_device()) =
        (arr >= 0).all();
    if (!all_nonneg_t.scalar<bool>()()) {
      return errors::InvalidArgument("Input arr must be non-negative!");
    }

    // Allocate partial output bin sums for each worker thread. Worker ids in
    // ParallelForWithWorkerId range from 0 to NumThreads() inclusive.
    ThreadPool* thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int64 num_threads = thread_pool->NumThreads() + 1;
    Tensor partial_bins_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<T>::value,
                                              TensorShape({num_threads, size}),
                                              &partial_bins_t));
    auto partial_bins = partial_bins_t.matrix<T>();
    partial_bins.setZero();
    thread_pool->ParallelForWithWorkerId(
        arr.size(), 8 /* cost */,
        [&](int64 start_ind, int64 limit_ind, int64 worker_id) {
          for (int64 i = start_ind; i < limit_ind; i++) {
            int32 value = arr(i);
            if (value < size) {
              if (weights.size()) {
                partial_bins(worker_id, value) += weights(i);
              } else {
                // Complex numbers don't support "++".
                partial_bins(worker_id, value) += T(1);
              }
            }
          }
        });

    // Sum the partial bins along the 0th axis.
    Eigen::array<int, 1> reduce_dims({0});
    output.device(context->eigen_cpu_device()) = partial_bins.sum(reduce_dims);
    return Status::OK();
  }
};

#ifdef TENSORFLOW_USE_SYCL

namespace {
//Generate a matrix which values are the index of the columns
template <typename T>
struct ColumnsIndicesGenerator {
  inline T operator()(const Eigen::array<Eigen::DenseIndex, 2>& idx) const {
    return idx[1];
  }
};
} // namespace

template <typename T>
struct BincountFunctor<SYCLDevice, T> {
  static Status Compute(OpKernelContext* context,
                        const typename TTypes<int32, 1>::ConstTensor& arr,
                        const typename TTypes<T, 1>::ConstTensor& weights,
                        typename TTypes<T, 1>::Tensor& output) {

    const SYCLDevice& d = context->eigen_device<SYCLDevice>();

    auto input_size = arr.size();
    auto output_size = output.size();

    if (output_size == 0)
      return Status::OK();

    if (input_size == 0)
    {
      output.device(d) = output.constant(T(0));
      return Status::OK();
    }

    // Checks if all values in arr are positive
    Tensor all_nonneg_t;
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_BOOL, TensorShape({}), &all_nonneg_t, AllocatorAttributes()));
    auto all_nonneg = all_nonneg_t.scalar<bool>();
    all_nonneg.device(d) = (arr >= 0).all();
    bool all_nonneg_host = false;
    d.memcpyDeviceToHost(&all_nonneg_host, all_nonneg.data(), sizeof(bool));
    if (!all_nonneg_host) {
      return errors::InvalidArgument("Input arr must be non-negative!");
    }

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> input_size_by_one({input_size, 1});
    Eigen::DSizes<Eigen::Index, 2> one_by_output_size({1, output_size});
    Eigen::DSizes<Eigen::Index, 1> sum_dim(0);
#else
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1>> input_size_by_one;
    input_size_by_one.set(0, input_size);
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_output_size;
    one_by_output_size.set(1, output_size);
    Eigen::IndexList<Eigen::type2index<0>> sum_dim;
#endif

    auto bcast_input = arr.reshape(input_size_by_one).
                           broadcast(one_by_output_size);
    auto index_matrix = bcast_input.generate(ColumnsIndicesGenerator<int32>());

    auto result_mat = (bcast_input == index_matrix).template cast<T>();

    if (weights.size()) {
      auto bcast_weights = weights.reshape(input_size_by_one).
                                   broadcast(one_by_output_size);
      output.device(d) = (result_mat * bcast_weights).sum(sum_dim);
    }
    else {
      output.device(d) = result_mat.sum(sum_dim);
    }

    return Status::OK();
  }
};

#endif //TENSORFLOW_USE_SYCL

}  // namespace functor

template <typename Device, typename T>
class BincountOp : public OpKernel {
 public:
  explicit BincountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& arr_t = ctx->input(0);
    const Tensor& size_tensor = ctx->input(1);
    const Tensor& weights_t = ctx->input(2);

    int32 size = size_tensor.scalar<int32>()();
    OP_REQUIRES(
        ctx, size >= 0,
        errors::InvalidArgument("size (", size, ") must be non-negative"));

    const auto arr = arr_t.flat<int32>();
    const auto weights = weights_t.flat<T>();
    Tensor* output_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({size}), &output_t));
    auto output = output_t->flat<T>();
    OP_REQUIRES_OK(ctx, functor::BincountFunctor<Device, T>::Compute(
                            ctx, arr, weights, output));
  }
};

#define REGISTER_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Bincount").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BincountOp<CPUDevice, type>)

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("Bincount")                \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("size")         \
                              .TypeConstraint<type>("T"), \
                          BincountOp<GPUDevice, type>)

TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_float(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("Bincount")                \
                              .Device(DEVICE_SYCL)        \
                              .HostMemory("size")         \
                              .TypeConstraint<type>("T"), \
                          BincountOp<SYCLDevice, type>)

TF_CALL_int32(REGISTER_KERNELS);
TF_CALL_float(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // TENSORFLOW_USE_SYCL

}  // end namespace tensorflow
