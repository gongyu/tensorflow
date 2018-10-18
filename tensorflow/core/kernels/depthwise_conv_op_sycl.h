#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_DEPTHWISE_CONV_OP_SYCL_H_
#define TENSORFLOW_KERNELS_DEPTHWISE_CONV_OP_SYCL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#include "tensorflow/core/kernels/depthwise_conv_op.h"

#include "sycldnn/accessor_types.h"
#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/depthwise_conv2d/launch.h"
#include "sycldnn/backend/eigen_backend.h"

namespace tensorflow {

typedef Eigen::SyclDevice SYCLDevice;

template <typename T>
struct LaunchDepthwiseConvOp<SYCLDevice, T> {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* input, const T* depthwise_filter, T* output,
                  TensorFormat data_format) noexcept {
    OP_REQUIRES(
        ctx, data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Depthwise convolution on SYCL is only supported for NHWC format"));

    namespace sd = sycldnn::depthwise_conv2d;
    sd::DepthwiseConv2DParams sd_params = get_sd_params(args);

    auto device = ctx->template eigen_device<SYCLDevice>();
    sycldnn::backend::EigenBackend backend(device);
    auto status = sd::launch<T, sycldnn::conv2d::conv_type::Forward>(input,
        depthwise_filter, output, sd_params, backend);
    if (status.status != sycldnn::StatusCode::OK) {
      ctx->SetStatus(get_sd_err_msg(status));
      return;
    }
    status.event.wait();
  }
};

template <typename T>
struct LaunchDepthwiseConvBackpropInputOp<SYCLDevice, T> {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* out_backprop, const T* depthwise_filter,
                  T* in_backprop, TensorFormat data_format) noexcept {
    OP_REQUIRES(
        ctx, data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Depthwise convolution on SYCL is only supported for NHWC format"));

    namespace sd = sycldnn::depthwise_conv2d;
    sd::DepthwiseConv2DParams sd_params = get_sd_params(args);

    auto device = ctx->template eigen_device<SYCLDevice>();
    sycldnn::backend::EigenBackend backend(device);
    auto status = sd::launch<T, sycldnn::conv2d::conv_type::InputBackprop>(
        out_backprop, depthwise_filter, in_backprop, sd_params, backend);
    if (status.status != sycldnn::StatusCode::OK) {
      ctx->SetStatus(get_sd_err_msg(status));
      return;
    }
    status.event.wait();
  }
};

template <typename T>
struct LaunchDepthwiseConvBackpropFilterOp<SYCLDevice, T> {
  void operator()(OpKernelContext* ctx, const DepthwiseArgs& args,
                  const T* out_backprop, const T* input, T* filter_backprop,
                  TensorFormat data_format) noexcept {
    OP_REQUIRES(
        ctx, data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Depthwise convolution on SYCL is only supported for NHWC format"));

    namespace sd = sycldnn::depthwise_conv2d;
    sd::DepthwiseConv2DParams sd_params = get_sd_params(args);

    auto device = ctx->template eigen_device<SYCLDevice>();
    sycldnn::backend::EigenBackend backend(device);
    auto status = sd::launch<T, sycldnn::conv2d::conv_type::FilterBackprop>(
        input, out_backprop, filter_backprop, sd_params, backend);
    if (status.status != sycldnn::StatusCode::OK) {
      ctx->SetStatus(get_sd_err_msg(status));
      return;
    }
    status.event.wait();
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_DEPTHWISE_CONV_OP_SYCL_H_
