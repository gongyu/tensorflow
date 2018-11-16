#ifndef TENSORFLOW_USE_SYCL
#error This file should only be included when compiling with SYCL support
#endif

#ifndef TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
#define TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_

#ifdef ARM_NON_MOBILE
#define SNN_ARM 1
#define SNN_SELECTOR arm_selector
#else
#define SNN_SELECTOR default_selector
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/common_runtime/sycl/sycl_util.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/conv_ops_sycl_launcher.h"

#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/backend/eigen_backend.h"

namespace tensorflow {
typedef Eigen::SyclDevice SYCLDevice;
// Forward declarations needed for later specializations.
template <typename Device, typename T>
struct LaunchConv2DOp;

template <typename T>
struct LaunchConv2DOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& input,
                  const Tensor& filter, int row_dilation, int col_dilation,
                  int stride_rows_, int stride_cols_, const Padding& padding,
                  Tensor* output, TensorFormat data_format) {
    if (row_dilation > 1 || col_dilation > 1) {
      context->SetStatus(
          errors::Unimplemented("The current SYCL convolution implementation "
                                "only supports dilated rate of 1 for now."));
      return;
    }
    const int64 batch = GetTensorDim(input, data_format, 'N');
    const int64 input_rows = GetTensorDim(input, data_format, 'H');
    const int64 input_cols = GetTensorDim(input, data_format, 'W');

    const int64 stride_rows = stride_rows_;
    const int64 stride_cols = stride_cols_;
    const int64 filter_rows = filter.dim_size(0);
    const int64 filter_cols = filter.dim_size(1);
    const int64 in_depth = filter.dim_size(2);
    const int64 out_depth = filter.dim_size(3);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding, &out_cols, &pad_cols));

    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  filter_rows, filter_cols, stride_rows,
                            stride_cols, out_rows,    out_cols,    pad_rows,
                            pad_cols};

    // TODO: Remove the SYCLConv2DParams struct and always use the sycldnn one instead.
    namespace sd = sycldnn::conv2d;
    static_assert(sizeof(SYCLConv2DParams) == sizeof(sd::Conv2DParams),
                  "SYCLConv2DParams shouldn't be used anymore");
    auto sd_params = *reinterpret_cast<sd::Conv2DParams*>(&params);
    auto device = context->eigen_device<SYCLDevice>();
    sycldnn::backend::EigenBackend backend(device);

    T const* const in_ptr = input.template flat<T>().data();
    T const* const fil_ptr = filter.template flat<T>().data();
    T* const out_ptr = output->template flat<T>().data();

    if (data_format == FORMAT_NCHW) {
      SNN_SELECTOR sel;
      launch_conv2d_nchw<T, ConvType::Forward>(
          context->eigen_device<SYCLDevice>(),
          in_ptr, fil_ptr, params, out_ptr, sel);
    } else {
      if (!is_snn_enabled()) {
        SNN_SELECTOR sel;
        launch_conv2d<T, ConvType::Forward>(context->eigen_device<SYCLDevice>(),
                                          in_ptr, fil_ptr, params, out_ptr, sel);
      }
      else {
        auto selector = sycldnn::conv2d::get_default_selector(
            device.sycl_queue().get_device());
        auto status = sd::launch<T, sd::conv_type::Forward>(
            in_ptr, fil_ptr, out_ptr, sd_params, *selector, backend);
        if (status.status != sycldnn::StatusCode::OK) {
          context->SetStatus(get_sd_err_msg(status));
          return;
        }
        status.event.wait();
      }
    }
  }
};

template <typename T>
struct LaunchConv2DBackpropInputOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& out_backprop,
                  const Tensor& filter, int row_dilation, int col_dilation,
                  int stride_rows_, int stride_cols_, const Padding& padding,
                  Tensor* in_backprop, TensorFormat data_format) {
    if (row_dilation > 1 || col_dilation > 1) {
      context->SetStatus(
          errors::Unimplemented("The current SYCL convolution implementation "
                                "only supports dilated rate of 1 for now."));
      return;
    }
    const int64 batch = GetTensorDim(*in_backprop, data_format, 'N');
    const int64 input_rows = GetTensorDim(*in_backprop, data_format, 'H');
    const int64 input_cols = GetTensorDim(*in_backprop, data_format, 'W');

    const int64 stride_rows = stride_rows_;
    const int64 stride_cols = stride_cols_;
    const int64 filter_rows = filter.dim_size(0);
    const int64 filter_cols = filter.dim_size(1);
    const int64 in_depth = filter.dim_size(2);
    const int64 out_depth = filter.dim_size(3);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding, &out_cols, &pad_cols));

    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  filter_rows, filter_cols, stride_rows,
                            stride_cols, out_rows,    out_cols,    pad_rows,
                            pad_cols};

    // TODO: Remove the SYCLConv2DParams struct and always use the sycldnn one instead.
    namespace sd = sycldnn::conv2d;
    static_assert(sizeof(SYCLConv2DParams) == sizeof(sd::Conv2DParams),
                  "SYCLConv2DParams shouldn't be used anymore");
    auto sd_params = *reinterpret_cast<sd::Conv2DParams*>(&params);
    auto device = context->eigen_device<SYCLDevice>();
    sycldnn::backend::EigenBackend backend(device);

    T const* const in_ptr = out_backprop.template flat<T>().data();
    T const* const fil_ptr = filter.template flat<T>().data();
    T* const out_ptr = in_backprop->template flat<T>().data();

    if (data_format == FORMAT_NCHW) {
      SNN_SELECTOR sel;
      launch_conv2d_nchw<T, ConvType::InputBackprop>(
          context->eigen_device<SYCLDevice>(),
          in_ptr, fil_ptr, params, out_ptr, sel);
    } else {
      if (!is_snn_enabled()) {
        SNN_SELECTOR sel;
        launch_conv2d<T, ConvType::InputBackprop>(context->eigen_device<SYCLDevice>(),
                                          in_ptr, fil_ptr, params, out_ptr, sel);
      }
      else {
        auto selector = sycldnn::conv2d::get_default_selector(
            device.sycl_queue().get_device());
        auto status = sd::launch<T, sd::conv_type::InputBackprop>(
            in_ptr, fil_ptr, out_ptr, sd_params, *selector, backend);
        if (status.status != sycldnn::StatusCode::OK) {
          context->SetStatus(get_sd_err_msg(status));
          return;
        }
        status.event.wait();
      }
    }
  }
};

template <typename T>
struct LaunchConv2DBackpropFilterOp<SYCLDevice, T> {
  void operator()(OpKernelContext* context, bool /*use_cudnn*/,
                  bool /*cudnn_use_autotune*/, const Tensor& out_backprop,
                  const Tensor& input, int row_dilation, int col_dilation,
                  int stride_rows_, int stride_cols_, const Padding& padding,
                  Tensor* filter_backprop, TensorFormat data_format) {
    if (row_dilation > 1 || col_dilation > 1) {
      context->SetStatus(
          errors::Unimplemented("The current SYCL convolution implementation "
                                "only supports dilated rate of 1 for now."));
      return;
    }
    const int64 batch = GetTensorDim(input, data_format, 'N');
    const int64 input_rows = GetTensorDim(input, data_format, 'H');
    const int64 input_cols = GetTensorDim(input, data_format, 'W');

    const int64 stride_rows = stride_rows_;
    const int64 stride_cols = stride_cols_;
    const int64 filter_rows = filter_backprop->dim_size(0);
    const int64 filter_cols = filter_backprop->dim_size(1);
    const int64 in_depth = filter_backprop->dim_size(2);
    const int64 out_depth = filter_backprop->dim_size(3);

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                         padding, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context,
                   GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                         padding, &out_cols, &pad_cols));

    SYCLConv2DParams params{in_depth,    out_depth,   batch,       input_rows,
                            input_cols,  filter_rows, filter_cols, stride_rows,
                            stride_cols, out_rows,    out_cols,    pad_rows,
                            pad_cols};

    // TODO: Remove the SYCLConv2DParams struct and always use the sycldnn one instead.
    namespace sd = sycldnn::conv2d;
    static_assert(sizeof(SYCLConv2DParams) == sizeof(sd::Conv2DParams),
                  "SYCLConv2DParams shouldn't be used anymore");
    auto sd_params = *reinterpret_cast<sd::Conv2DParams*>(&params);
    auto device = context->eigen_device<SYCLDevice>();
    sycldnn::backend::EigenBackend backend(device);

    T const* const in_ptr = input.template flat<T>().data();
    T const* const fil_ptr = out_backprop.template flat<T>().data();
    T* const out_ptr = filter_backprop->template flat<T>().data();

    if (data_format == FORMAT_NCHW) {
      SNN_SELECTOR sel;
      launch_conv2d_nchw<T, ConvType::FilterBackprop>(
          context->eigen_device<SYCLDevice>(),
          in_ptr, fil_ptr, params, out_ptr, sel);
    } else {
      if (!is_snn_enabled()) {
        SNN_SELECTOR sel;
        launch_conv2d<T, ConvType::FilterBackprop>(context->eigen_device<SYCLDevice>(),
                                          in_ptr, fil_ptr, params, out_ptr, sel);
      }
      else {
        auto selector = sycldnn::conv2d::get_default_selector(
            device.sycl_queue().get_device());
        auto status = sd::launch<T, sd::conv_type::FilterBackprop>(
            in_ptr, fil_ptr, out_ptr, sd_params, *selector, backend);
        if (status.status != sycldnn::StatusCode::OK) {
          context->SetStatus(get_sd_err_msg(status));
          return;
        }
        status.event.wait();
      }
    }
  }
};

}  // namespace tensorflow

#undef SNN_SELECTOR

#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
