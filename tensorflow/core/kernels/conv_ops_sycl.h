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

#include "tensorflow/core/kernels/sycl_dnn_utils.h"
#include "tensorflow/core/kernels/conv_grad_ops.h"
#include "tensorflow/core/kernels/conv_ops_sycl_launcher.h"

#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/conv2d/launch.h"

namespace tensorflow {

namespace snn {
namespace conv2d = sycldnn::conv2d;
struct SNNSelectorGen final : public conv2d::Selector {
  conv2d::Algorithm select(const conv2d::Conv2DParams& params) override {
    if (params.stride_rows == 1 && params.stride_cols == 1) {
      if (params.window_rows == 1 && params.window_cols == 1) {
        return conv2d::Algorithm::Matmul;
      }
      else if ((params.window_rows == 1 && params.window_cols == 3) ||
               (params.window_rows == 3 && params.window_cols == 1) ||
               (params.window_rows == 3 && params.window_cols == 3)) {
        return conv2d::Algorithm::Winograd;
      }
    }
#ifdef ARM_NON_MOBILE
    return conv2d::Algorithm::Tiled;
#else
    return conv2d::Algorithm::Im2col;
#endif
  }

  const char* name() const override { return "SNNSelectorGen"; }
};

// FilterBackprop is not supported with tiled convolution so make sure not to select that here
struct SNNSelectorFilterBackprop final : public conv2d::Selector {
  conv2d::Algorithm select(const conv2d::Conv2DParams& params) override {
    if (params.stride_rows == 1 && params.stride_cols == 1) {
      if (params.window_rows == 1 && params.window_cols == 1) {
        return conv2d::Algorithm::Matmul;
      }
      else if ((params.window_rows == 1 && params.window_cols == 3) ||
               (params.window_rows == 3 && params.window_cols == 1) ||
               (params.window_rows == 3 && params.window_cols == 3)) {
        return conv2d::Algorithm::Winograd;
      }
    }
    return conv2d::Algorithm::Im2col;
  }

  const char* name() const override { return "SNNSelectorFilterBackprop"; }
};

inline conv2d::Conv2DParams sycl_to_sd_params(const SYCLConv2DParams& params) {
  conv2d::Conv2DParams sd_params;
  sd_params.channels = params.channels_;
  sd_params.features = params.features_;
  sd_params.batch = params.batch_;
  sd_params.in_rows = params.in_rows_;
  sd_params.in_cols = params.in_cols_;
  sd_params.window_rows = params.window_rows_;
  sd_params.window_cols = params.window_cols_;
  sd_params.stride_rows = params.stride_rows_;
  sd_params.stride_cols = params.stride_cols_;
  sd_params.out_rows = params.out_rows_;
  sd_params.out_cols = params.out_cols_;
  sd_params.pad_rows = params.pad_rows_;
  sd_params.pad_cols = params.pad_cols_;
  sd_params.dilation_rows = params.dilation_rows_;
  sd_params.dilation_cols = params.dilation_cols_;
  return sd_params;
}
}  // namespace snn

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
    auto sd_params = snn::sycl_to_sd_params(params);
    auto device = context->eigen_device<SYCLDevice>();

    auto in_t = input.template flat<T>();
    auto fil_t = filter.template flat<T>();
    auto out_t = output->template flat<T>();

    auto in_ptr = in_t.data();
    auto fil_ptr = fil_t.data();
    auto out_ptr = out_t.data();

    if (data_format == FORMAT_NCHW) {
      SNN_SELECTOR sel;
      launch_conv2d_nchw<T, ConvType::Forward>(device, in_ptr, fil_ptr,
                                               params, out_ptr, sel);
    } else {
      if (!is_snn_enabled()) {
        SNN_SELECTOR sel;
        launch_conv2d<T, ConvType::Forward>(device, in_ptr, fil_ptr,
                                            params, out_ptr, sel);
      } else {
        snn::SNNSelectorGen selector;
        CREATE_SNN_BACKEND(backend, device);
#ifdef SYCL_SNN_USE_BLAS_BACKEND
        auto ph = backend.get_executor().get_policy_handler();
        in_ptr = attach_pointer<T>(device, ph, in_ptr);
        fil_ptr = attach_pointer<T>(device, ph, fil_ptr);
        out_ptr = attach_pointer<T>(device, ph, out_ptr);
#endif
        sycldnn::SNNStatus status = sd::launch<T, sd::conv_type::Forward>(
            in_ptr, fil_ptr, out_ptr, sd_params, selector, backend);
        if (status.status != sycldnn::StatusCode::OK) {
          context->SetStatus(get_sd_err_msg(status));
          return;
        }
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
    auto sd_params = snn::sycl_to_sd_params(params);
    auto device = context->eigen_device<SYCLDevice>();

    auto in_t = out_backprop.template flat<T>();
    auto fil_t = filter.template flat<T>();
    auto out_t = in_backprop->template flat<T>();

    auto in_ptr = in_t.data();
    auto fil_ptr = fil_t.data();
    auto out_ptr = out_t.data();

    if (data_format == FORMAT_NCHW) {
      SNN_SELECTOR sel;
      launch_conv2d_nchw<T, ConvType::InputBackprop>(device, in_ptr, fil_ptr,
                                                     params, out_ptr, sel);
    } else {
      if (!is_snn_enabled()) {
        SNN_SELECTOR sel;
        launch_conv2d<T, ConvType::InputBackprop>(device, in_ptr, fil_ptr,
                                                  params, out_ptr, sel);
      } else {
        snn::SNNSelectorGen selector;
        CREATE_SNN_BACKEND(backend, device);
#ifdef SYCL_SNN_USE_BLAS_BACKEND
        auto ph = backend.get_executor().get_policy_handler();
        in_ptr = attach_pointer<T>(device, ph, in_ptr);
        fil_ptr = attach_pointer<T>(device, ph, fil_ptr);
        out_ptr = attach_pointer<T>(device, ph, out_ptr);
#endif
        sycldnn::SNNStatus status = sd::launch<T, sd::conv_type::InputBackprop>(
            in_ptr, fil_ptr, out_ptr, sd_params, selector, backend);
        if (status.status != sycldnn::StatusCode::OK) {
          context->SetStatus(get_sd_err_msg(status));
          return;
        }
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
    auto sd_params = snn::sycl_to_sd_params(params);
    auto device = context->eigen_device<SYCLDevice>();

    auto in_t = input.template flat<T>();
    auto fil_t = out_backprop.template flat<T>();
    auto out_t = filter_backprop->template flat<T>();

    auto in_ptr = in_t.data();
    auto fil_ptr = fil_t.data();
    auto out_ptr = out_t.data();

    if (data_format == FORMAT_NCHW) {
      SNN_SELECTOR sel;
      launch_conv2d_nchw<T, ConvType::FilterBackprop>(device, in_ptr, fil_ptr,
                                                      params, out_ptr, sel);
    } else {
      if (!is_snn_enabled()) {
        SNN_SELECTOR sel;
        launch_conv2d<T, ConvType::FilterBackprop>(device, in_ptr, fil_ptr,
                                                   params, out_ptr, sel);
      } else {
        snn::SNNSelectorFilterBackprop selector;
        CREATE_SNN_BACKEND(backend, device);
#ifdef SYCL_SNN_USE_BLAS_BACKEND
        auto ph = backend.get_executor().get_policy_handler();
        in_ptr = attach_pointer<T>(device, ph, in_ptr);
        fil_ptr = attach_pointer<T>(device, ph, fil_ptr);
        out_ptr = attach_pointer<T>(device, ph, out_ptr);
#endif
        sycldnn::SNNStatus status =
          sd::launch<T, sd::conv_type::FilterBackprop>(
            in_ptr, fil_ptr, out_ptr, sd_params, selector, backend);
        if (status.status != sycldnn::StatusCode::OK) {
          context->SetStatus(get_sd_err_msg(status));
          return;
        }
      }
    }
  }
};

}  // namespace tensorflow

#undef SNN_SELECTOR

#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
