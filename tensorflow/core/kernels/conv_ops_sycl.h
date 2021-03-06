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

template <class S, class P>
std::string algo_to_str(const S& s, const P& p) {
  switch(s.select(p)) {
    case conv2d::Algorithm::Direct:
      return "Direct";
    case conv2d::Algorithm::Tiled:
      return "Tiled";
    case conv2d::Algorithm::Im2col:
      return "Im2col";
    case conv2d::Algorithm::Winograd:
      return "Winograd";
    case conv2d::Algorithm::Matmul:
      return "Matmul";
    default:
      return "NotSupported";
  }
}

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

    auto in_ptr = input.template flat<T>().data();
    auto fil_ptr = filter.template flat<T>().data();
    auto out_ptr = output->template flat<T>().data();

    if (data_format == FORMAT_NCHW) {
      SNN_SELECTOR sel;
      launch_conv2d_nchw<T, ConvType::Forward>(device, in_ptr, fil_ptr,
                                               params, out_ptr, sel);
    } else {
      vlog_conv2d_params(sd_params, data_format, "forward_conv2d");
      auto sycl_device = device.sycl_queue().get_device();
      static auto sd_selector = sd::get_default_selector(sycl_device);
      CREATE_SNN_BACKEND(sd_backend, device);
      auto sd_in = get_sycl_dnn_input<T>(device, in_ptr);
      auto sd_fil = get_sycl_dnn_input<T>(device, fil_ptr);
      auto sd_out = get_sycl_dnn_input<T>(device, out_ptr);
      sycldnn::SNNStatus sd_status = sd::launch<T, sd::conv_type::Forward>(
          sd_in, sd_fil, sd_out, sd_params, *sd_selector, sd_backend);
      if (sd_status.status != sycldnn::StatusCode::OK) {
        context->SetStatus(get_sd_err_msg(sd_status));
        return;
      }
    }
    device.async_synchronize();
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

    auto in_ptr = out_backprop.template flat<T>().data();
    auto fil_ptr = filter.template flat<T>().data();
    auto out_ptr = in_backprop->template flat<T>().data();

    if (data_format == FORMAT_NCHW) {
      SNN_SELECTOR sel;
      launch_conv2d_nchw<T, ConvType::InputBackprop>(device, in_ptr, fil_ptr,
                                                     params, out_ptr, sel);
    } else {
      vlog_conv2d_params(sd_params, data_format, "input_backprop_conv2d");
      auto sycl_device = device.sycl_queue().get_device();
      static auto sd_selector = sd::get_default_selector(sycl_device);
      CREATE_SNN_BACKEND(sd_backend, device);
      auto sd_in = get_sycl_dnn_input<T>(device, in_ptr);
      auto sd_fil = get_sycl_dnn_input<T>(device, fil_ptr);
      auto sd_out = get_sycl_dnn_input<T>(device, out_ptr);
      sycldnn::SNNStatus sd_status = sd::launch<T, sd::conv_type::InputBackprop>(
          sd_in, sd_fil, sd_out, sd_params, *sd_selector, sd_backend);
      if (sd_status.status != sycldnn::StatusCode::OK) {
        context->SetStatus(get_sd_err_msg(sd_status));
        return;
      }
    }
    device.async_synchronize();
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

    auto in_ptr = input.template flat<T>().data();
    auto fil_ptr = out_backprop.template flat<T>().data();
    auto out_ptr = filter_backprop->template flat<T>().data();

    if (data_format == FORMAT_NCHW) {
      SNN_SELECTOR sel;
      launch_conv2d_nchw<T, ConvType::FilterBackprop>(device, in_ptr, fil_ptr,
                                                      params, out_ptr, sel);
    } else {
      vlog_conv2d_params(sd_params, data_format, "filter_backprop_conv2d");
      auto sycl_device = device.sycl_queue().get_device();
      static auto sd_selector = sd::get_default_selector(sycl_device);
      CREATE_SNN_BACKEND(sd_backend, device);
      auto sd_in = get_sycl_dnn_input<T>(device, in_ptr);
      auto sd_fil = get_sycl_dnn_input<T>(device, fil_ptr);
      auto sd_out = get_sycl_dnn_input<T>(device, out_ptr);
      sycldnn::SNNStatus sd_status =
        sd::launch<T, sd::conv_type::FilterBackprop>(
          sd_in, sd_fil, sd_out, sd_params, *sd_selector, sd_backend);
      if (sd_status.status != sycldnn::StatusCode::OK) {
        context->SetStatus(get_sd_err_msg(sd_status));
        return;
      }
    }
    device.async_synchronize();
  }
};

}  // namespace tensorflow

#undef SNN_SELECTOR

#endif  // TENSORFLOW_KERNELS_CONV_OPS_SYCL_H_
