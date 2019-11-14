/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
   Copyright (C) Codeplay Software Limited

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

#include "tensorflow/contrib/tensoropt/kernels/convert_nodes.h"

#include <algorithm>
#include <bitset>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/node_def.pb.h"  // NOLINT
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"  // NOLINT
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/types.h"

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT

namespace tensorflow {
namespace tensoropt {
namespace convert {
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;
namespace {

template <class T>
std::string VecToStr(const std::vector<T>& v) {
  std::stringstream ss("[");
  if (!v.empty()) {
    ss << v[0];
    for (unsigned i = 1; i < v.size(); ++i)
      ss << ", " << v[i];
  }
  ss << "]";
  return ss.str();
}

/*
 * TensorOrWeights represents data that can be easier on the device (Tensor)
 * or the host (Weight).
 */
class TensorOrWeights {
 public:
  using dim_t = uint32_t;
  using dims_t = std::vector<dim_t>;

  TensorOrWeights() : dimensions_(), op_(), op_idx_(-1), data_(nullptr) {}

  template <class DimT = uint32_t>
  explicit TensorOrWeights(
      const std::vector<DimT>& shape,
      ANeuralNetworksOperandCode type =
          ANeuralNetworksOperandCode::ANEURALNETWORKS_INVALID,
      const void* data = nullptr)
      : dimensions_(), op_(), op_idx_(UINT_MAX), data_(data) {
    dimensions_.reserve(shape.size());
    for (DimT d : shape)
      dimensions_.push_back(static_cast<dim_t>(d));
    op_.type = type;
    op_.dimensionCount = dimensions_.size();
    op_.dimensions = dimensions_.data();
  }

  explicit TensorOrWeights(const tensorflow::Tensor& tensor)
      : TensorOrWeights() {
    dimensions_.reserve(tensor.dims());
    for (dim_t i = 0; i < op_.dimensionCount; i++) {
      dimensions_.push_back(tensor.dim_size(i));
    }
    TF_CHECK_OK(ConvertDTypeToTOPTTensorType(tensor.dtype(), &op_.type));
    op_.dimensionCount = dimensions_.size();
    op_.dimensions = dimensions_.data();
  }

  TensorOrWeights(const TensorOrWeights& other)
      : dimensions_(other.dimensions_),
        op_(other.op_),
        op_idx_(other.op_idx_),
        data_(other.data_) {
    op_.dimensions = dimensions_.data();
  }
  TensorOrWeights& operator=(const TensorOrWeights& other) {
    dimensions_ = other.dimensions_;
    op_ = other.op_;
    op_idx_ = other.op_idx_;
    data_ = other.data_;
    op_.dimensions = dimensions_.data();
    return *this;
  }

  TensorOrWeights(TensorOrWeights&&) = default;
  TensorOrWeights& operator=(TensorOrWeights&&) = default;

  ~TensorOrWeights() = default;

  inline void set_tensor_type(tensorflow::DataType type) {
    TF_CHECK_OK(ConvertDTypeToTOPTTensorType(type, &op_.type));
  }

  inline void set_host_type(tensorflow::DataType type) {
    TF_CHECK_OK(ConvertDTypeToTOPTHostType(type, &op_.type));
  }

  inline bool is_tensor() const {
    return op_.type ==
               ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_INT32 ||
           op_.type ==
               ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_FLOAT32;
  }

  inline bool is_host() const {
    return op_.type == ANeuralNetworksOperandCode::ANEURALNETWORKS_INT32 ||
           op_.type == ANeuralNetworksOperandCode::ANEURALNETWORKS_UINT32 ||
           op_.type == ANeuralNetworksOperandCode::ANEURALNETWORKS_FLOAT32;
  }

  inline const dims_t& dimensions() const { return dimensions_; }
  inline const ANeuralNetworksOperandType& op() const { return op_; }
  inline operator ANeuralNetworksOperandType() const { return op_; }

  inline dim_t& idx() { return op_idx_; }
  inline const dim_t& get_valid_idx() const {
    CHECK_NE(op_idx_, UINT_MAX);
    return op_idx_;
  }
  inline void set_idx(dim_t idx) { op_idx_ = idx; }

  // Returns total number of elements in shape
  uint32_t count() const {
    if (op_.dimensionCount == 0) {
      return 0;
    }
    uint32_t count = 1;
    for (dim_t d = 0; d < op_.dimensionCount; ++d) {
      count *= op_.dimensions[d];
    }
    return count;
  }

  inline size_t size_bytes() const {
    return count() * tensorflow::DataTypeSize(ConvertTOPTTypeToDType(op_.type));
  }
  inline const void* get_data() const { return data_; }
  inline void set_data(const void* data) { data_ = data; }
  inline tensorflow::DataType dtype() const {
    return ConvertTOPTTypeToDType(op_.type);
  }

  template <class T>
  std::ostream& print_data(std::ostream& os) const {
    auto data_count = count();
    auto printed_count = std::min(10U, data_count);
    CHECK_EQ(is_host(), true);
    os << "[";
    if (printed_count > 0) {
      auto data = static_cast<const T*>(data_);
      os << data[0];
      for (unsigned i = 1; i < printed_count; ++i) {
        os << ", " << data[i];
      }
      if (data_count > printed_count) {
        os << ", ...";
      }
    }
    return os << "]";
  }

 private:
  // op dimensions points to this dimensions_ field so that memory is handled
  // by RAII
  dims_t dimensions_;
  ANeuralNetworksOperandType op_;
  dim_t op_idx_;
  const void* data_;
};

std::ostream& operator<<(std::ostream& os, const TensorOrWeights& tw) {
  auto topt_type = tw.op().type;
  os << "{topt_type=" << topt_type << ", shape=[";
  const auto& dims = tw.dimensions();
  if (!dims.empty()) {
    os << dims[0];
    for (unsigned i = 1; i < dims.size(); ++i) {
      os << ", " << dims[i];
    }
  }
  os << "], data=";
  if (tw.is_tensor()) {
    os << "[device tensor]";
  } else {
    switch (topt_type) {
      case ANEURALNETWORKS_INT32:
        tw.print_data<int32_t>(os);
        break;
      case ANEURALNETWORKS_UINT32:
        tw.print_data<uint32_t>(os);
        break;
      case ANEURALNETWORKS_FLOAT32:
        tw.print_data<float>(os);
        break;
      default:
        os << "[unsupported host type " << topt_type << "]";
        break;
    }
  }
  return os << "}";
}

/*
 * Helper class to read nodes attributes.
 */
class TFAttrs {
 public:
  explicit TFAttrs(const tensorflow::NodeDef& tf_node) {
    for (const auto& attr : tf_node.attr()) {
      attrs_.insert({attr.first, &attr.second});
    }
  }
  bool ount(string key) const { return attrs_.count(key); }
  tensorflow::AttrValue const* at(string key) const {
    if (!attrs_.count(key)) {
      LOG(FATAL) << "Attribute not found: " << key;
    }
    return attrs_.at(key);
  }
  template <typename T>
  T get(const string& key) const;
  template <typename T>
  T get(const string& key, const T& default_value) const {
    return attrs_.count(key) ? this->get<T>(key) : default_value;
  }

  std::vector<string> GetAllAttrKey() {
    std::vector<string> attr_list;
    for (const auto& attr_item : attrs_) {
      attr_list.emplace_back(attr_item.first);
    }
    return attr_list;
  }

 private:
  typedef std::map<string, tensorflow::AttrValue const*> AttrMap;
  AttrMap attrs_;
};

template <>
string TFAttrs::get<string>(const string& key) const {
  return this->at(key)->s();
}

template <>
std::vector<int> TFAttrs::get<std::vector<int>>(const string& key) const {
  auto attr = this->at(key)->list().i();
  return std::vector<int>(attr.begin(), attr.end());
}

template <>
std::vector<float> TFAttrs::get<std::vector<float>>(const string& key) const {
  auto attr = this->at(key)->list().f();
  return std::vector<float>(attr.begin(), attr.end());
}

template <>
std::vector<string> TFAttrs::get<std::vector<string>>(const string& key) const {
  auto attr = this->at(key)->list().s();
  return std::vector<string>(attr.begin(), attr.end());
}

template <>
tensorflow::DataType TFAttrs::get<tensorflow::DataType>(
    const string& key) const {
  return this->at(key)->type();
}

template <>
float TFAttrs::get<float>(const string& key) const {
  return this->at(key)->f();
}

template <>
int TFAttrs::get<int>(const string& key) const {
  return static_cast<int>(this->at(key)->i());
}

template <>
bool TFAttrs::get<bool>(const string& key) const {
  return this->at(key)->b();
}

class Converter;

using OpConverter = std::function<tensorflow::Status(
    Converter&, const tensorflow::NodeDef&, const std::vector<TensorOrWeights>&,
    std::vector<TensorOrWeights>*)>;

class Converter {
  std::unordered_map<string, TensorOrWeights> topt_tensors_;
  std::unordered_map<string, OpConverter> op_registry_;
  ANeuralNetworksModel* topt_model_;
  std::vector<std::vector<uint8_t>>& weight_store_;

  void register_op_converters();

  tensorflow::Status get_inputs(const tensorflow::NodeDef& node_def,
                                std::vector<TensorOrWeights>* inputs) {
    for (auto const& input_name : node_def.input()) {
      /*************************************************************************
       * TODO handle case 1) here
       * Normalizes the inputs and extracts associated metadata:
       * 1) Inputs can contain a colon followed by a suffix of characters.
       *    That suffix may be a single number (e.g. inputName:1) or several
       *    word characters separated from a number by a colon
       *    (e.g. inputName:foo:1). The
       *    latter case is used to denote inputs and outputs of functions.
       * 2) Control dependency inputs contain caret at the beginning and we
       *    remove this and annotate the edge as a control dependency.
       ************************************************************************/
      // skip control nodes
      if (input_name[0] == '^') {
        continue;
      }
      string name = input_name;
      auto first = name.find_first_of(':');
      if (first != string::npos && first + 2 == name.size() &&
          name[first + 1] == '0') {
        name.erase(first);
      }

      if (topt_tensors_.count(name)) {
        auto& input = topt_tensors_.at(name);
        inputs->push_back(input);
        VLOG(2) << "retrieve input: " << name << " " << input;
      } else {
        string str("Node ");
        StrAppend(&str, node_def.name(), " should have an input named '", name,
                  "' but it is not available");
        LOG(WARNING) << "input: " << name << " not available for node at "
                     << node_def.name();
        return tensorflow::errors::InvalidArgument(str);
      }
    }
    return tensorflow::Status::OK();
  }

 public:
  explicit Converter(ANeuralNetworksModel* model,
                     std::vector<std::vector<uint8_t>>& weight_store)
      : topt_model_(model), weight_store_(weight_store) {
    this->register_op_converters();
  }

  TensorOrWeights get_temp_weights(tensorflow::DataType type,
                                   const std::vector<uint32_t>& shape,
                                   void** store_data = nullptr) {
    TensorOrWeights weights(shape);
    weights.set_host_type(type);
    weight_store_.emplace_back(weights.size_bytes());
    weights.set_data(weight_store_.back().data());
    if (store_data) {
      *store_data = weight_store_.back().data();
    }
    return weights;
  }

  TensorOrWeights get_temp_weights_like(const TensorOrWeights& weights,
                                        void** store_data = nullptr) {
    return this->get_temp_weights(weights.dtype(), weights.dimensions(),
                                  store_data);
  }

  tensorflow::Status convert_node(const tensorflow::NodeDef& node_def) {
    std::vector<TensorOrWeights> inputs;
    TF_RETURN_IF_ERROR(this->get_inputs(node_def, &inputs));
    string op = node_def.op();
    std::vector<TensorOrWeights> outputs;

    if (!op_registry_.count(op)) {
      return tensorflow::errors::Unimplemented(
          "No converter registered for op: " + op);
    }
    OpConverter op_converter = op_registry_.at(op);
    TF_RETURN_IF_ERROR(op_converter(*this, node_def, inputs, &outputs));

    for (size_t i = 0; i < outputs.size(); ++i) {
      TensorOrWeights output = outputs.at(i);
      string output_name = node_def.name();
      if (i != 0) {
        output_name = StrCat(output_name, ":", i);
      }
      VLOG(2) << "Write out tensor: " << output_name << " " << output;
      if (!topt_tensors_.insert({output_name, output}).second) {
        return tensorflow::errors::AlreadyExists(
            "Output tensor already exists for op: " + op);
      }
    }
    return tensorflow::Status::OK();
  }

  ANeuralNetworksModel* model() { return topt_model_; }

  TensorOrWeights get_tensor(string name) { return topt_tensors_.at(name); }

  bool insert_input_tensor(string name, const TensorOrWeights& topt_tensor) {
    return topt_tensors_.insert({name, topt_tensor}).second;
  }

  template <class T>
  inline TensorOrWeights create_scalar_host_operand(T value) {
    // By default cast enum types to int32
    return create_scalar_host_operand(static_cast<int32_t>(value));
  }

  inline void set_host_operand_value(const TensorOrWeights& weight) {
    TOPT_CHECK_OK(Model_setOperandValue(topt_model_, weight.get_valid_idx(),
                                        weight.get_data(),
                                        weight.size_bytes()));
  }

  inline void create_host_operand(TensorOrWeights& weight) {
    TOPT_CHECK_OK(Model_addOperand(topt_model_, &weight.op(), &weight.idx()));
    set_host_operand_value(weight);
  }

 private:
  template <class T>
  inline TensorOrWeights create_scalar_host_operand_helper(
      T value, ANeuralNetworksOperandCode op_code) {
    static_assert(
        sizeof(T) < ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES,
        "create_scalar_host_operand can only be used for small scalar inputs");
    // TensorOpt will copy "value" since it is smaller than
    // ANEURALNETWORKS_MAX_SIZE_OF_IMMEDIATELY_COPIED_VALUES
    // so the pointer to value won't be used outside of this scope.
    TensorOrWeights host_operand({1}, op_code, &value);
    create_host_operand(host_operand);
    return host_operand;
  }
};

template <>
inline TensorOrWeights Converter::create_scalar_host_operand<bool>(bool value) {
  return create_scalar_host_operand_helper(
      value, ANeuralNetworksOperandCode::ANEURALNETWORKS_BOOL);
}

template <>
inline TensorOrWeights Converter::create_scalar_host_operand<int32_t>(
    int32_t value) {
  return create_scalar_host_operand_helper(
      value, ANeuralNetworksOperandCode::ANEURALNETWORKS_INT32);
}

template <>
inline TensorOrWeights Converter::create_scalar_host_operand<uint32_t>(
    uint32_t value) {
  return create_scalar_host_operand_helper(
      value, ANeuralNetworksOperandCode::ANEURALNETWORKS_UINT32);
}

template <>
inline TensorOrWeights Converter::create_scalar_host_operand<float>(
    float value) {
  return create_scalar_host_operand_helper(
      value, ANeuralNetworksOperandCode::ANEURALNETWORKS_FLOAT32);
}

tensorflow::Status ConvertConv2D(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TensorOrWeights>& inputs,
                                 std::vector<TensorOrWeights>* outputs) {
  ANeuralNetworksOperationType op_code;
  if (node_def.op() == "Conv2D") {
    op_code = ANEURALNETWORKS_CONV_2D;
  } else if (node_def.op() == "DepthwiseConv2dNative") {
    op_code = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
  } else {
    return tensorflow::errors::Unimplemented("Unsupported operation: " +
                                             node_def.op());
  }

  if (inputs.size() != 2) {
    return tensorflow::errors::FailedPrecondition(
        "Convolution ops require two tensor inputs, at " + node_def.name());
  }
  const auto& input = inputs.at(0);
  const auto& filter = inputs.at(1);

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  const auto tf_data_format = attrs.get<string>("data_format");
  const auto tf_strides = attrs.get<std::vector<int>>("strides");
  const auto tf_padding = attrs.get<string>("padding");
  const auto tf_dilatations = attrs.get<std::vector<int>>("dilations");
  CHECK_EQ(input.dtype(), node_dtype);
  CHECK_EQ(filter.dtype(), node_dtype);
  CHECK_EQ(tf_strides.size(), 4);
  CHECK_EQ(tf_strides[0], 1);
  CHECK_EQ(tf_strides[3], 1);
  CHECK_EQ(tf_dilatations.size(), 4);
  CHECK_EQ(tf_dilatations[0], 1);
  CHECK_EQ(tf_dilatations[3], 1);

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return tensorflow::errors::Unimplemented(
        "Unsupported data format " + tf_padding + ", at " + node_def.name());
  }
  if (tf_padding != "SAME" && tf_padding != "VALID") {
    return tensorflow::errors::Unimplemented(
        "Unsupported padding " + tf_padding + ", at " + node_def.name());
  }

  const auto& input_dims = input.dimensions();
  if (input_dims.size() != 4) {
    string err_str("Convolution ops require 4D input but got ");
    StrAppend(&err_str, input_dims.size(), "D at ", node_def.name());
    return tensorflow::errors::FailedPrecondition(err_str);
  }
  const auto& filter_dims = filter.dimensions();
  if (filter_dims.size() != 4) {
    string err_str("Convolution ops require 4D filter but got ");
    StrAppend(&err_str, filter_dims.size(), "D at ", node_def.name());
    return tensorflow::errors::FailedPrecondition(err_str);
  }

  const bool is_nhwc = tf_data_format == "NHWC";
  const bool is_same = tf_padding == "SAME";
  const unsigned n_dim = 0;
  const unsigned h_dim = is_nhwc ? 1 : 2;
  const unsigned w_dim = h_dim + 1;
  const unsigned c_dim = is_nhwc ? 3 : 1;
  TensorOrWeights::dims_t output_dims(4);
  output_dims[n_dim] = input_dims[n_dim];
  // Filter format is always [FilterH, FilterW, InChannels, OutChannels]
  // for convolution or [FilterH, FilterW, InChannels, ChannelMultiplier]
  // for depthwise convolutions
  CHECK_EQ(filter_dims[2], input_dims[c_dim]);
  output_dims[c_dim] = filter_dims[3];
  if (op_code == ANEURALNETWORKS_DEPTHWISE_CONV_2D) {
    output_dims[c_dim] *= filter_dims[2];
  }
  const std::array<unsigned, 2> hw_dims{h_dim, w_dim};
  for (unsigned i = 0; i < hw_dims.size(); ++i) {
    unsigned dim = hw_dims[i];
    int64 padding_before;
    int64 padding_after;
    int64 output_size;
    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        input_dims[dim], filter_dims[i], tf_dilatations[dim],
        tf_strides[dim], is_same ? Padding::SAME : Padding::VALID, &output_size,
        &padding_before, &padding_after));
    output_dims[dim] = static_cast<TensorOrWeights::dim_t>(output_size);
    VLOG(2) << "Convolution i=" << i << " input=" << input_dims[dim]
            << " ksize=" << filter_dims[i] << " dilation=" << tf_dilatations[dim]
            << " stride=" << tf_strides[dim]
            << " padding=" << tf_padding << " output_size=" << output_size
            << " padding_before=" << padding_before << " padding_after=" << padding_after;
  }

  // Create unused bias operand
  TensorOrWeights bias({}, input.op().type);
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &bias.op(), &bias.idx()));

  // Create input host operands
  const auto padding = ctx.create_scalar_host_operand(
      is_same ? ANEURALNETWORKS_PADDING_SAME : ANEURALNETWORKS_PADDING_VALID);
  const auto stride_w = ctx.create_scalar_host_operand(tf_strides[w_dim]);
  const auto stride_h = ctx.create_scalar_host_operand(tf_strides[h_dim]);
  const auto fuse_code = ctx.create_scalar_host_operand(ANEURALNETWORKS_FUSED_NONE);
  const auto use_nchw = ctx.create_scalar_host_operand(!is_nhwc);
  const auto use_filter_hwio = ctx.create_scalar_host_operand(true);
  const auto dilation_w = ctx.create_scalar_host_operand(tf_dilatations[w_dim]);
  const auto dilation_h = ctx.create_scalar_host_operand(tf_dilatations[h_dim]);

  // Create an output operand
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(output_dims, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS = 11;
  const uint32_t operation_inputs[NB_INPUTS] = {
      input.get_valid_idx(),
      filter.get_valid_idx(),
      bias.get_valid_idx(),
      padding.get_valid_idx(),
      stride_w.get_valid_idx(),
      stride_h.get_valid_idx(),
      fuse_code.get_valid_idx(),
      use_nchw.get_valid_idx(),
      use_filter_hwio.get_valid_idx(),
      dilation_w.get_valid_idx(),
      dilation_h.get_valid_idx(),
  };
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), op_code, NB_INPUTS,
                                   operation_inputs, 1,
                                   &output.get_valid_idx()));

  return tensorflow::Status::OK();
}

tensorflow::Status ConvertPool(Converter& ctx,
                               const tensorflow::NodeDef& node_def,
                               const std::vector<TensorOrWeights>& inputs,
                               std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 1) {
    return tensorflow::errors::FailedPrecondition(
        "Pooling ops require single tensor input, at " + node_def.name());
  }
  const auto& input = inputs.at(0);

  ANeuralNetworksOperationType op_code;
  if (node_def.op() == "MaxPool") {
    op_code = ANEURALNETWORKS_MAX_POOL_2D;
  } else if (node_def.op() == "AvgPool") {
    op_code = ANEURALNETWORKS_AVERAGE_POOL_2D;
  } else {
    return tensorflow::errors::Unimplemented("Unsupported operation: " +
                                             node_def.op());
  }

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  const auto tf_data_format = attrs.get<string>("data_format");
  const auto tf_strides = attrs.get<std::vector<int>>("strides");
  const auto tf_ksize = attrs.get<std::vector<int>>("ksize");
  const auto tf_padding = attrs.get<string>("padding");
  CHECK_EQ(input.dtype(), node_dtype);
  CHECK_EQ(tf_strides.size(), 4);
  CHECK_EQ(tf_strides[0], 1);
  CHECK_EQ(tf_strides[3], 1);
  CHECK_EQ(tf_ksize.size(), 4);
  CHECK_EQ(tf_ksize[0], 1);
  CHECK_EQ(tf_ksize[3], 1);

  if (tf_data_format != "NHWC" && tf_data_format != "NCHW") {
    return tensorflow::errors::Unimplemented(
        "Unsupported data format " + tf_padding + ", at " + node_def.name());
  }
  if (tf_padding != "SAME" && tf_padding != "VALID") {
    return tensorflow::errors::Unimplemented(
        "Unsupported padding " + tf_padding + ", at " + node_def.name());
  }

  const auto& input_dims = input.dimensions();
  if (input_dims.size() != 4) {
    string err_str("Pooling ops require 4D input but got ");
    StrAppend(&err_str, input_dims.size(), " at ", node_def.name());
    return tensorflow::errors::FailedPrecondition(err_str);
  }

  const bool is_nhwc = tf_data_format == "NHWC";
  const bool is_same = tf_padding == "SAME";
  const unsigned n_dim = 0;
  const unsigned h_dim = is_nhwc ? 1 : 2;
  const unsigned w_dim = h_dim + 1;
  const unsigned c_dim = is_nhwc ? 3 : 1;
  TensorOrWeights::dims_t output_dims(4);
  output_dims[n_dim] = input_dims[n_dim];
  output_dims[c_dim] = input_dims[c_dim];
  static constexpr int64 POOLING_DILATION = 1;
  const std::array<unsigned, 2> hw_dims{h_dim, w_dim};
  for (unsigned i = 0; i < hw_dims.size(); ++i) {
    unsigned dim = hw_dims[i];
    int64 padding_before;
    int64 padding_after;
    int64 output_size;
    TF_CHECK_OK(GetWindowedOutputSizeVerboseV2(
        input_dims[dim], tf_ksize[dim], POOLING_DILATION,
        tf_strides[dim], is_same ? Padding::SAME : Padding::VALID, &output_size,
        &padding_before, &padding_after));
    output_dims[dim] = static_cast<TensorOrWeights::dim_t>(output_size);
    VLOG(2) << "Pooling i=" << i << " input=" << input_dims[dim]
            << " ksize=" << tf_ksize[dim] << " stride=" << tf_strides[dim]
            << " padding=" << tf_padding << " output_size=" << output_size
            << " padding_before=" << padding_before
            << " padding_after=" << padding_after;
  }

  // Create input host operands
  const auto padding = ctx.create_scalar_host_operand(
      is_same ? ANEURALNETWORKS_PADDING_SAME : ANEURALNETWORKS_PADDING_VALID);
  const auto stride_w = ctx.create_scalar_host_operand(tf_strides[w_dim]);
  const auto stride_h = ctx.create_scalar_host_operand(tf_strides[h_dim]);
  const auto filter_w = ctx.create_scalar_host_operand(tf_ksize[w_dim]);
  const auto filter_h = ctx.create_scalar_host_operand(tf_ksize[h_dim]);
  const auto fuse_code = ctx.create_scalar_host_operand(ANEURALNETWORKS_FUSED_NONE);
  const auto use_nchw = ctx.create_scalar_host_operand(!is_nhwc);

  // Create an output operand
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(output_dims, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS = 8;
  const uint32_t operation_inputs[NB_INPUTS] = {
      input.get_valid_idx(),     padding.get_valid_idx(),
      stride_w.get_valid_idx(),  stride_h.get_valid_idx(),
      filter_w.get_valid_idx(),  filter_h.get_valid_idx(),
      fuse_code.get_valid_idx(), use_nchw.get_valid_idx(),
  };
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), op_code, NB_INPUTS,
                                   operation_inputs, 1,
                                   &output.get_valid_idx()));

  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConst(Converter& ctx,
                                const tensorflow::NodeDef& node_def,
                                const std::vector<TensorOrWeights>& inputs,
                                std::vector<TensorOrWeights>* outputs) {
  const auto& proto_tensor = node_def.attr().at("value").tensor();

  // Get trt type & shape
  const TFAttrs attrs(node_def);
  const tensorflow::DataType node_dtype =
      attrs.get<tensorflow::DataType>("dtype");

  // Create shaped weights as output
  tensorflow::Tensor tensor;
  if (!tensor.FromProto(proto_tensor)) {
    return tensorflow::errors::Internal("Cannot parse weight tensor proto: " +
                                        node_def.name());
  }

  VLOG(2) << "Const shape=" << tensor.shape();
  TensorOrWeights::dims_t scalar_shape;
  scalar_shape.reserve(tensor.dims());
  for (auto dim : tensor.shape()) {
    scalar_shape.push_back(dim.size);
  }
  TensorOrWeights weights(scalar_shape);
  if (!proto_tensor.float_val().empty()) {
    VLOG(2) << "Converting scalar float weight: " << node_def.name();
    if (weights.count() != proto_tensor.float_val_size()) {
      if (tensor.dims() == 0 || proto_tensor.float_val_size() == 1 ||
          scalar_shape[0] == proto_tensor.float_val_size()) {
        // no dimension provided. flatten it
        scalar_shape.resize(1);
        scalar_shape[0] = proto_tensor.float_val_size();
      } else {
        string err_str("Cannot broadcast Const node '");
        StrAppend(&err_str, node_def.name(), "' of type ", node_def.op());
        return tensorflow::errors::InvalidArgument(err_str);
      }
    }
    void* store_data;
    weights = ctx.get_temp_weights(node_dtype, scalar_shape, &store_data);
    // Make a local copy first to flatten
    std::vector<float> tensor_data(proto_tensor.float_val().begin(),
                                   proto_tensor.float_val().end());
    std::memcpy(store_data, tensor_data.data(), weights.size_bytes());
  } else if (!proto_tensor.int_val().empty()) {
    VLOG(2) << "Converting scalar int weight: " << node_def.name();
    if (weights.count() != proto_tensor.int_val_size()) {
      if (tensor.dims() == 0 || proto_tensor.int_val_size() == 1 ||
          scalar_shape[0] == proto_tensor.int_val_size()) {
        // no dimension provided. flatten it
        scalar_shape.resize(1);
        scalar_shape[0] = proto_tensor.int_val_size();
      } else {
        string err_str("Cannot broadcast Const node '");
        StrAppend(&err_str, node_def.name(), "' of type ", node_def.op());
        return tensorflow::errors::InvalidArgument(err_str);
      }
    }
    void* store_data;
    weights = ctx.get_temp_weights(node_dtype, scalar_shape, &store_data);
    // Make a local copy first to flatten
    std::vector<int32> tensor_data(proto_tensor.int_val().begin(),
                                   proto_tensor.int_val().end());
    std::memcpy(store_data, tensor_data.data(), weights.size_bytes());
  } else if (!proto_tensor.tensor_content().empty()) {
    VLOG(2) << "Converting Tensor weight: " << node_def.name();
    const auto& content = proto_tensor.tensor_content();
    const int dtype_size = tensorflow::DataTypeSize(node_dtype);
    auto content_count = content.size() / dtype_size;
    if (content_count > 0 && weights.count() != content_count) {
      // no dimension provided. flatten it
      scalar_shape.resize(1);
      scalar_shape[0] = content_count;
    }

    void* store_data;
    weights = ctx.get_temp_weights(node_dtype, scalar_shape, &store_data);
    if (content.size() > 0) {
      CHECK_EQ(0, content.size() % dtype_size)
          << "Tensor content size (" << content.size()
          << ") is not a multiple of " << dtype_size;
      port::CopyToArray(content, static_cast<char*>(store_data));
    }
  } else {
    return tensorflow::errors::Unimplemented(
        "Not supported constant type, at " + node_def.name());
  }
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &weights.op(), &weights.idx()));
  TOPT_CHECK_OK(Model_setOperandValue(ctx.model(), weights.get_valid_idx(),
                                      weights.get_data(),
                                      weights.size_bytes()));
  outputs->push_back(weights);
  VLOG(2) << "Output Const node '" << node_def.name() << "': " << weights;
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertIdentity(Converter&, const tensorflow::NodeDef&,
                                   const std::vector<TensorOrWeights>& inputs,
                                   std::vector<TensorOrWeights>* outputs) {
  outputs->push_back(inputs.at(0));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertBinary(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TensorOrWeights>& inputs,
                                 std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 2) {
    return tensorflow::errors::FailedPrecondition(
        "Binary ops require two tensor inputs, at " + node_def.name());
  }
  const auto& lhs = inputs.at(0);
  const auto& rhs = inputs.at(1);

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  CHECK_EQ(lhs.dtype(), node_dtype);
  CHECK_EQ(rhs.dtype(), node_dtype);

  const std::unordered_map<string, ANeuralNetworksOperationType> node_to_op{
      {"Add", ANeuralNetworksOperationType::ANEURALNETWORKS_ADD},
      {"BiasAdd", ANeuralNetworksOperationType::ANEURALNETWORKS_ADD},
      {"Mul", ANeuralNetworksOperationType::ANEURALNETWORKS_MUL},
      {"Sub", ANeuralNetworksOperationType::ANEURALNETWORKS_SUB},
      {"Div", ANeuralNetworksOperationType::ANEURALNETWORKS_DIV},
      {"RealDiv", ANeuralNetworksOperationType::ANEURALNETWORKS_DIV},
      {"Minimum", ANeuralNetworksOperationType::ANEURALNETWORKS_MIN},
      {"Maximum", ANeuralNetworksOperationType::ANEURALNETWORKS_MAX},
  };
  const auto op_pair = node_to_op.find(node_def.op());
  if (op_pair == node_to_op.end()) {
    return tensorflow::errors::Unimplemented(
        "Binary op: " + node_def.op() +
        " not supported at: " + node_def.name());
  }

  // Create an output operand
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  // The output dimension is the input's dimensions with the most elements
  // as one of the input can be broadcasted. If lhs and rhs have the same
  // number of elements keep the one with the most dimensions as it holds
  // more information.
  auto output_dims_ptr = &lhs.dimensions();
  const auto lhs_count = lhs.count();
  const auto rhs_count = rhs.count();
  if (rhs_count > lhs_count ||
      (rhs_count == lhs_count &&
       rhs.dimensions().size() > lhs.dimensions().size())) {
    output_dims_ptr = &rhs.dimensions();
  }
  outputs->emplace_back(*output_dims_ptr, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS = 2;
  const uint32_t operation_inputs[NB_INPUTS] = {lhs.get_valid_idx(),
                                          rhs.get_valid_idx()};
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), op_pair->second, NB_INPUTS,
                                   operation_inputs, 1,
                                   &output.get_valid_idx()));

  return tensorflow::Status::OK();
}

tensorflow::Status ConvertUnary(Converter& ctx,
                                const tensorflow::NodeDef& node_def,
                                const std::vector<TensorOrWeights>& inputs,
                                std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 1) {
    return tensorflow::errors::FailedPrecondition(
        "Unary ops require single tensor input, at " + node_def.name());
  }
  const auto& input = inputs.at(0);

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  CHECK_EQ(input.dtype(), node_dtype);

  const std::unordered_map<string, ANeuralNetworksOperationType> node_to_op{
      {"Relu", ANeuralNetworksOperationType::ANEURALNETWORKS_RELU},
      {"Relu1", ANeuralNetworksOperationType::ANEURALNETWORKS_RELU1},
      {"Relu6", ANeuralNetworksOperationType::ANEURALNETWORKS_RELU6},
      {"Exp", ANeuralNetworksOperationType::ANEURALNETWORKS_EXP},
      {"Sqrt", ANeuralNetworksOperationType::ANEURALNETWORKS_SQRT},
      {"Rsqrt", ANeuralNetworksOperationType::ANEURALNETWORKS_RSQRT},
  };
  const auto op_pair = node_to_op.find(node_def.op());
  if (op_pair == node_to_op.end()) {
    return tensorflow::errors::Unimplemented(
        "Unary op: " + node_def.op() + " not supported at: " + node_def.name());
  }

  // Create an output operand
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(input.dimensions(), output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  TOPT_CHECK_OK(Model_addOperation(ctx.model(), op_pair->second, 1,
                                   &input.get_valid_idx(), 1,
                                   &output.get_valid_idx()));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSoftmax(Converter& ctx,
                                  const tensorflow::NodeDef& node_def,
                                  const std::vector<TensorOrWeights>& inputs,
                                  std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 1) {
    return tensorflow::errors::FailedPrecondition(
        "SoftmaxOp requires single tensor input, at " + node_def.name());
  }
  const auto& input = inputs.at(0);

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  CHECK_EQ(input.dtype(), node_dtype);

  // Create an output operand
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(input.dimensions(), output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  TOPT_CHECK_OK(Model_addOperation(
      ctx.model(), ANeuralNetworksOperationType::ANEURALNETWORKS_SOFTMAX, 1,
      &input.get_valid_idx(), 1, &output.get_valid_idx()));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertConcat(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TensorOrWeights>& inputs,
                                 std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() < 2) {
    return tensorflow::errors::FailedPrecondition(
        "ConcatOp requires at least 2 inputs, at " + node_def.name());
  }

  const TFAttrs attrs(node_def);
  const auto index_type = attrs.get<tensorflow::DataType>("Tidx");
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  const std::size_t concat_size = inputs.size() - 1;

  for (std::size_t i = 0; i < concat_size; ++i) {
    CHECK_EQ(inputs.at(i).dtype(), node_dtype);
  }

  // Only expect to handle INT32 as index attributes for now
  if (index_type != tensorflow::DataType::DT_INT32) {
    return tensorflow::errors::Unimplemented(
        "Concat supports only Tidx of DT_INT32, at " + node_def.name());
  }

  const auto axis = inputs.at(concat_size);
  if (!axis.is_host()) {
    return tensorflow::errors::InvalidArgument(
        "Concat axis is not a host value, at " + node_def.name());
  }

  // Create input host operands
  ctx.set_host_operand_value(axis);
  VLOG(2) << "Concat axis: " << axis;

  // Create an output operand
  auto shape_vec = inputs.at(0).dimensions();
  int tf_axis = *static_cast<const int*>(axis.get_data());
  if (tf_axis < 0) {
    tf_axis += shape_vec.size();
  }
  for (unsigned i = 1; i < concat_size; ++i) {
    shape_vec[tf_axis] += inputs.at(i).dimensions()[tf_axis];
  }
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(shape_vec, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  std::vector<uint32_t> operation_inputs(inputs.size());
  for (unsigned i = 0; i < inputs.size(); ++i) {
    operation_inputs[i] = inputs.at(i).get_valid_idx();
  }
  TOPT_CHECK_OK(Model_addOperation(
      ctx.model(), ANEURALNETWORKS_CONCATENATION, operation_inputs.size(),
      operation_inputs.data(), 1, &output.get_valid_idx()));

  return tensorflow::Status::OK();
}

tensorflow::Status ConvertPack(Converter& ctx,
                               const tensorflow::NodeDef& node_def,
                               const std::vector<TensorOrWeights>& inputs,
                               std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() < 1) {
    return tensorflow::errors::FailedPrecondition(
        "PackOp requires at least 1 input, at " + node_def.name());
  }

  const TFAttrs attrs(node_def);
  const auto attr_n = attrs.get<int>("N", inputs.size());
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  auto tf_axis = attrs.get<int>("axis");
  CHECK_EQ(inputs.size(), attr_n);

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs.at(i).dtype(), node_dtype);
  }

  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));

  // Create reshape operations
  const auto& input0_shape = inputs.at(0).dimensions();
  const auto input_rank = static_cast<uint32_t>(input0_shape.size());
   auto new_input_shape = input0_shape;
  if (tf_axis < 0) {
    tf_axis += input_rank;
  }
  CHECK_GE(tf_axis, 0);
  CHECK_LE(tf_axis, input_rank);
  // First reshape the inputs to add a new dimension of size 1
  new_input_shape.insert(new_input_shape.begin() + tf_axis, 1);
  TensorOrWeights topt_new_shape(
      {static_cast<uint32_t>(new_input_shape.size())}, ANEURALNETWORKS_INT32,
      new_input_shape.data());
  ctx.create_host_operand(topt_new_shape);
  std::vector<TensorOrWeights> reshaped_inputs;
  static constexpr unsigned NB_INPUTS_RESHAPE = 2;
  uint32_t reshape_inputs[NB_INPUTS_RESHAPE];
  reshape_inputs[1] = topt_new_shape.get_valid_idx();
  for (unsigned i = 0; i < inputs.size(); ++i) {
    reshaped_inputs.emplace_back(new_input_shape, output_type);
    auto& reshaped_input = reshaped_inputs.back();
    TOPT_CHECK_OK(Model_addOperand(ctx.model(), &reshaped_input.op(),
                                   &reshaped_input.idx()));
    reshape_inputs[0] = inputs.at(i).get_valid_idx();
    TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_RESHAPE,
                                     NB_INPUTS_RESHAPE, reshape_inputs, 1,
                                     &reshaped_input.get_valid_idx()));
  }

  // Create input host operands
  const auto axis = ctx.create_scalar_host_operand(tf_axis);

  // Create an output operand
  new_input_shape[tf_axis] = inputs.size();
  outputs->emplace_back(new_input_shape, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  std::vector<uint32_t> operation_inputs(inputs.size() + 1);
  for (unsigned i = 0; i < inputs.size(); ++i) {
    operation_inputs[i] = reshaped_inputs.at(i).get_valid_idx();
  }
  operation_inputs[inputs.size()] = axis.get_valid_idx();
  TOPT_CHECK_OK(Model_addOperation(
      ctx.model(), ANEURALNETWORKS_CONCATENATION, operation_inputs.size(),
      operation_inputs.data(), 1, &output.get_valid_idx()));

  return tensorflow::Status::OK();
}

tensorflow::Status ConvertUnpack(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TensorOrWeights>& inputs,
                                 std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 1) {
    return tensorflow::errors::FailedPrecondition(
        "UnpackOp requires 1 input, at " + node_def.name());
  }

  const auto& input = inputs.at(0);
  const auto& input_shape = input.dimensions();
  const auto input_rank = static_cast<uint32_t>(input_shape.size());

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  auto tf_axis = attrs.get<int>("axis");
  if (tf_axis < 0) {
    tf_axis += input_rank;
  }
  CHECK_GE(tf_axis, 0);
  CHECK_LT(tf_axis, input_rank);
  const unsigned num_slices = input_shape[tf_axis];
  const auto attr_num = attrs.get<int>("num", num_slices);
  CHECK_EQ(num_slices, attr_num);

  for (std::size_t i = 0; i < inputs.size(); ++i) {
    CHECK_EQ(inputs.at(i).dtype(), node_dtype);
  }

  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));

  // Create input host operand
  auto sizes = input_shape;
  sizes[tf_axis] = 1;
  TensorOrWeights topt_size({input_rank},
                            ANEURALNETWORKS_INT32, sizes.data());
  ctx.create_host_operand(topt_size);

  // Create slice and reshape operations
  auto sliced_shape = input_shape;
  sliced_shape[tf_axis] = 1;
  TensorOrWeights sliced_input(sliced_shape, output_type);
  auto outputs_shape = input_shape;
  outputs_shape.erase(outputs_shape.begin() + tf_axis);
  TensorOrWeights topt_new_shape({static_cast<uint32_t>(outputs_shape.size())},
                                 ANEURALNETWORKS_INT32, outputs_shape.data());
  ctx.create_host_operand(topt_new_shape);
  std::vector<int> begins(input_rank);
  TensorOrWeights topt_begin({static_cast<uint32_t>(begins.size())},
                             ANEURALNETWORKS_INT32, begins.data());
  static constexpr unsigned NB_INPUTS_SLICE = 3;
  uint32_t slice_inputs[NB_INPUTS_SLICE];
  slice_inputs[0] = input.get_valid_idx();
  slice_inputs[2] = topt_size.get_valid_idx();
  static constexpr unsigned NB_INPUTS_RESHAPE = 2;
  uint32_t reshape_inputs[NB_INPUTS_RESHAPE];
  reshape_inputs[1] = topt_new_shape.get_valid_idx();
  for (unsigned i = 0; i < num_slices; ++i) {
    // Slice
    TOPT_CHECK_OK(
        Model_addOperand(ctx.model(), &sliced_input.op(), &sliced_input.idx()));
    begins[tf_axis] = i;
    ctx.create_host_operand(topt_begin);
    slice_inputs[1] = topt_begin.get_valid_idx();
    TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_SLICE,
                                     NB_INPUTS_SLICE, slice_inputs, 1,
                                     &sliced_input.get_valid_idx()));

    // Create output operand
    outputs->emplace_back(outputs_shape, output_type);
    auto& output = outputs->back();
    TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

    // Reshape
    reshape_inputs[0] = sliced_input.get_valid_idx();
    TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_RESHAPE,
                                     NB_INPUTS_RESHAPE, reshape_inputs, 1,
                                     &output.get_valid_idx()));
  }

  return tensorflow::Status::OK();
}

tensorflow::Status ConvertFusedBatchNorm(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TensorOrWeights>& inputs,
    std::vector<TensorOrWeights>* outputs) {
  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  const float epsilon = attrs.get<float>("epsilon", 0.f);
  const auto data_format = attrs.get<string>("data_format");
  if (data_format != "NHWC") {
    return tensorflow::errors::Unimplemented(
        "only data_format=NHWC is supported, at " + node_def.name());
  }
  const bool is_training = attrs.get<bool>("is_training");
  if (is_training) {
    return tensorflow::errors::Unimplemented(
        "only is_training=false is supported, at " + node_def.name());
  }
  const auto& input = inputs.at(0);
  CHECK_EQ(input.dtype(), node_dtype);

  //  Check parameter types
  const auto parameter_type = inputs.at(1).dtype();
  if (parameter_type != tensorflow::DataType::DT_FLOAT) {
    return tensorflow::errors::Unimplemented(
        "only float32 data type is supported, for node " + node_def.name() +
        " got " + tensorflow::DataTypeString(parameter_type));
  }
  for (int i = 1; i < 5; i++) {
    const auto& input = inputs.at(i);
    if (!input.is_host()) {
      return tensorflow::errors::InvalidArgument(
          "FusedBatchNorm expects inputs >0 to be host values, at " +
          node_def.name());
    }
    if (input.dtype() != parameter_type) {
      return tensorflow::errors::Unimplemented(
          "Inconsistent parameter type for batchnormis not supported, at: " +
          node_def.name());
    }
    VLOG(2) << "FusedBatchNorm input[" << i << "]: " << input;
  }

  std::size_t nweight = 0;
  std::size_t input_counts[4];
  for (int i = 0; i < 4; i++) {
    input_counts[i] = static_cast<std::size_t>(inputs.at(i + 1).count());
    nweight = std::max(nweight, input_counts[i]);
  }
  const TensorOrWeights* ptr_shape_weights = nullptr;
  for (int i = 0; i < 4; i++) {
    if (input_counts[i] == nweight) {
      ptr_shape_weights = &inputs.at(i + 1);
    } else if (input_counts[i] != 1) {
      return tensorflow::errors::InvalidArgument(
          "Inconsistent batchnorm parameter count, at: " + node_def.name());
    }
  }
  void* scale_data;
  TensorOrWeights combined_scale_weights =
      ctx.get_temp_weights_like(*ptr_shape_weights, &scale_data);
  void* offset_data;
  TensorOrWeights combined_offset_weights =
      ctx.get_temp_weights_like(*ptr_shape_weights, &offset_data);

  const float* vals_array[4];
  for (int j = 0; j < 4; j++) {
    vals_array[j] = static_cast<float const*>(inputs.at(j + 1).get_data());
  }
  auto combined_scale_vals = static_cast<float*>(scale_data);
  auto combined_offset_vals = static_cast<float*>(offset_data);

  for (size_t i = 0; i < nweight; ++i) {
    float batchnorm_data[4];
    for (int j = 0; j < 4; j++) {
      if (input_counts[j] != 1) {
        batchnorm_data[j] = vals_array[j][i];
      } else {
        batchnorm_data[j] = vals_array[j][0];
      }
    }
    const float scale = batchnorm_data[0];
    const float offset = batchnorm_data[1];
    const float mean = batchnorm_data[2];
    const float variance = batchnorm_data[3];
    combined_scale_vals[i] = scale / sqrtf(variance + epsilon);
    combined_offset_vals[i] = offset - mean * combined_scale_vals[i];
  }

  // Create input host operands
  ctx.create_host_operand(combined_scale_weights);
  ctx.create_host_operand(combined_offset_weights);

  // Create an intermediate operand
  TensorOrWeights intermediate(input);
  TOPT_CHECK_OK(
      Model_addOperand(ctx.model(), &intermediate.op(), &intermediate.idx()));

  // Create an output operand
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(input.dimensions(), output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS_BIN = 2;
  const uint32_t operation_inputs_mul[NB_INPUTS_BIN] = {
      input.get_valid_idx(), combined_scale_weights.get_valid_idx()};
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_MUL,
                                   NB_INPUTS_BIN, operation_inputs_mul, 1,
                                   &intermediate.get_valid_idx()));

  const uint32_t operation_inputs_add[NB_INPUTS_BIN] = {
      intermediate.get_valid_idx(), combined_offset_weights.get_valid_idx()};
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_ADD,
                                   NB_INPUTS_BIN, operation_inputs_add, 1,
                                   &output.get_valid_idx()));

  return tensorflow::Status::OK();
}

tensorflow::Status ConvertMatMul(Converter& ctx,
                                 const tensorflow::NodeDef& node_def,
                                 const std::vector<TensorOrWeights>& inputs,
                                 std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 2) {
    return tensorflow::errors::FailedPrecondition(
        "MatMul op requires two tensor inputs, at " + node_def.name());
  }
  const auto& lhs_input = inputs.at(0);
  const auto& rhs_input = inputs.at(1);

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  CHECK_EQ(lhs_input.dtype(), node_dtype);
  CHECK_EQ(rhs_input.dtype(), node_dtype);

  const bool lhs_t = attrs.get<bool>("transpose_a");
  const bool rhs_t = attrs.get<bool>("transpose_b");

  const auto topt_lhs_t = ctx.create_scalar_host_operand(lhs_t);
  const auto topt_rhs_t = ctx.create_scalar_host_operand(rhs_t);

  // Create an output operand
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  const uint32_t m = lhs_t ? lhs_input.dimensions()[1] : lhs_input.dimensions()[0];
  const uint32_t n = rhs_t ? rhs_input.dimensions()[0] : rhs_input.dimensions()[1];
  const TensorOrWeights::dims_t output_dims{m, n};
  outputs->emplace_back(output_dims, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS = 4;
  const uint32_t operation_inputs[NB_INPUTS] = {
    lhs_input.get_valid_idx(), rhs_input.get_valid_idx(),
    topt_lhs_t.get_valid_idx(), topt_rhs_t.get_valid_idx()
  };
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_MATMUL,
                                   NB_INPUTS, operation_inputs, 1,
                                   &output.get_valid_idx()));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertTranspose(Converter& ctx,
                                    const tensorflow::NodeDef& node_def,
                                    const std::vector<TensorOrWeights>& inputs,
                                    std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 2) {
    return tensorflow::errors::FailedPrecondition(
        "Transpose op requires two tensor inputs, at " + node_def.name());
  }
  const auto& input = inputs.at(0);
  const auto& perm = inputs.at(1);
  if (!input.is_tensor() || !perm.is_host()) {
    return tensorflow::errors::InvalidArgument(
        "Transpose expects inputs to be a tensor and a host value, at " +
        node_def.name());
  }

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  const auto perm_dtype = attrs.get<tensorflow::DataType>("Tperm");
  CHECK_EQ(input.dtype(), node_dtype);
  CHECK_EQ(perm.dtype(), perm_dtype);

  // Create an output operand
  TensorOrWeights::dims_t output_dims;
  const auto& input_dims = input.dimensions();
  const auto host_perm = static_cast<const int*>(perm.get_data());
  for (unsigned i = 0; i < input_dims.size(); ++i) {
    output_dims.push_back(input_dims[host_perm[i]]);
  }
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(output_dims, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS = 2;
  const uint32_t operation_inputs[NB_INPUTS] = {input.get_valid_idx(),
                                          perm.get_valid_idx()};
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_TRANSPOSE,
                                   NB_INPUTS, operation_inputs, 1,
                                   &output.get_valid_idx()));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertReshape(Converter& ctx,
                                  const tensorflow::NodeDef& node_def,
                                  const std::vector<TensorOrWeights>& inputs,
                                  std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 2) {
    return tensorflow::errors::InvalidArgument(
        "Reshape op requires 2 inputs, at " + node_def.name());
  }
  const auto& input = inputs.at(0);
  const auto& shape = inputs.at(1);
  if (!input.is_tensor() || !shape.is_host()) {
    return tensorflow::errors::InvalidArgument(
        "Reshape expects inputs to be a tensor and a host value, at " +
        node_def.name());
  }

  if (shape.op().dimensionCount != 1) {
    return tensorflow::errors::InvalidArgument(
        "Reshape new shape is not 1 dimensional, at " + node_def.name());
  }

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  CHECK_EQ(input.dtype(), node_dtype);
  const auto shape_type = attrs.get<tensorflow::DataType>("Tshape");
  if (shape_type != tensorflow::DataType::DT_INT32) {
    return tensorflow::errors::Unimplemented(
        "Reshape's new shape supports only DT_INT32, at " + node_def.name());
  }

  std::vector<int> shape_vec;
  const auto shape_data = static_cast<const int*>(shape.get_data());
  shape_vec.assign(shape_data, shape_data + shape.dimensions()[0]);
  VLOG(2) << "Reshaping " << VecToStr(input.dimensions()) << " to "
          << VecToStr(shape_vec);

  const auto volume_t = input.count();
  // shape can have at most one -1 dimension
  int volume_w = 1;
  long negative_dim_idx = -1;
  for (unsigned i = 0; i < shape_vec.size(); ++i) {
    int dim = shape_vec[i];
    volume_w *= dim;
    if (dim < 0) {
      negative_dim_idx = i;
    }
  }
  if (negative_dim_idx >= 0) {
    volume_w *= -1;
    if (volume_t % volume_w != 0) {
      return tensorflow::errors::InvalidArgument("invalid new shape, at " +
                                                 node_def.name());
    }
    const int new_dim = volume_t / volume_w;
    shape_vec[negative_dim_idx] = new_dim;
    volume_w *= new_dim;
  }

  VLOG(2) << "Reshape volume: " << volume_t << " volume weights: " << volume_w;
  if (volume_w != volume_t) {
    return tensorflow::errors::InvalidArgument(
        "volume does not agree between tensor and new shape, at " +
        node_def.name());
  }

  // Create input host operands
  ctx.set_host_operand_value(shape);

  // Create an output operand
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(shape_vec, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS = 2;
  const uint32_t operation_inputs[NB_INPUTS] = {input.get_valid_idx(),
                                          shape.get_valid_idx()};
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_RESHAPE,
                                   NB_INPUTS, operation_inputs, 1,
                                   &output.get_valid_idx()));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertSqueeze(Converter& ctx,
                                  const tensorflow::NodeDef& node_def,
                                  const std::vector<TensorOrWeights>& inputs,
                                  std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 1) {
    return tensorflow::errors::InvalidArgument(
        "Squeeze op requires 1 input, at " + node_def.name());
  }
  const auto& input = inputs.at(0);

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  const auto tf_dims = attrs.get<std::vector<int>>("squeeze_dims");
  CHECK_EQ(input.dtype(), node_dtype);

  // Create input host operands
  TensorOrWeights topt_dims({static_cast<unsigned>(tf_dims.size())},
                            ANEURALNETWORKS_INT32, tf_dims.data());
  ctx.create_host_operand(topt_dims);

  // Create an output operand
  std::vector<int> output_shape;
  for (unsigned i = 0; i < input.dimensions().size(); ++i) {
    if (input.dimensions()[i] == 1 &&
        (tf_dims.empty() ||
         std::find(tf_dims.begin(), tf_dims.end(), i) != tf_dims.end())) {
      continue;
    }
    output_shape.push_back(input.dimensions()[i]);
  }
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(output_shape, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS = 2;
  const uint32_t operation_inputs[NB_INPUTS] = {input.get_valid_idx(),
                                          topt_dims.get_valid_idx()};
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_SQUEEZE,
                                   NB_INPUTS, operation_inputs, 1,
                                   &output.get_valid_idx()));
  return tensorflow::Status::OK();
}

template <class Index>
static void ComputeSliceShape(std::vector<uint32_t>& output_shape,
                              const TensorOrWeights& size) {
  const auto size_data = static_cast<const Index*>(size.get_data());
  for (unsigned i = 0; i < output_shape.size(); ++i) {
    if (size_data[i] > 0) {
      output_shape[i] = size_data[i];
    }
  }
}

tensorflow::Status ConvertSlice(Converter& ctx,
                                const tensorflow::NodeDef& node_def,
                                const std::vector<TensorOrWeights>& inputs,
                                std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 3) {
    return tensorflow::errors::InvalidArgument(
        "Slice op requires 3 inputs, at " + node_def.name());
  }
  const auto& input = inputs.at(0);
  const auto& begin = inputs.at(1);
  const auto& size = inputs.at(2);

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  const auto index_type = attrs.get<tensorflow::DataType>("Index");
  CHECK_EQ(input.dtype(), node_dtype);
  CHECK_EQ(begin.dtype(), index_type);
  CHECK_EQ(size.dtype(), index_type);

  if (!begin.is_host() || !size.is_host()) {
    return tensorflow::errors::InvalidArgument(
        "Slice inputs >0 must be host values, at " + node_def.name());
  }

  // Create input host operands
  ctx.set_host_operand_value(begin);
  ctx.set_host_operand_value(size);
  VLOG(2) << "Slice begin=" << begin << " size=" << size;

  // Create an output operand
  auto output_shape = input.dimensions();
  if (index_type == tensorflow::DataType::DT_INT32) {
    ComputeSliceShape<int32_t>(output_shape, size);
  } else if (index_type == tensorflow::DataType::DT_INT64) {
    ComputeSliceShape<int64_t>(output_shape, size);
  } else {
    return tensorflow::errors::Unimplemented(
        "Slice supports only Index type of DT_INT32 and DT_INT64, at " +
        node_def.name());
  }
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(output_shape, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS = 3;
  const uint32_t operation_inputs[NB_INPUTS] = {
      input.get_valid_idx(), begin.get_valid_idx(), size.get_valid_idx()};
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_SLICE,
                                   NB_INPUTS, operation_inputs, 1,
                                   &output.get_valid_idx()));
  return tensorflow::Status::OK();
}

template <class Index>
static void ComputeStridedSliceShape(std::vector<uint32_t>& output_shape,
                                     const TensorOrWeights& topt_begin,
                                     const TensorOrWeights& topt_end,
                                     const TensorOrWeights& topt_stride,
                                     int begin_mask_i, int end_mask_i,
                                     int shrink_axis_mask_i,
                                     int ellipsis_mask_i, int new_axis_mask_i) {
  const auto topt_count = topt_begin.count();
  CHECK_EQ(topt_count, topt_end.count());
  CHECK_EQ(topt_count, topt_stride.count());
  const auto begin_data = static_cast<const Index*>(topt_begin.get_data());
  const auto end_data = static_cast<const Index*>(topt_end.get_data());
  const auto stride_data = static_cast<const Index*>(topt_stride.get_data());
  std::vector<Index> begin_vec(begin_data, begin_data + topt_count);
  std::vector<Index> end_vec(end_data, end_data + topt_count);
  std::vector<Index> stride_vec(stride_data, stride_data + topt_count);
  const auto rank = output_shape.size();
  const std::bitset<32> begin_mask(begin_mask_i);
  const std::bitset<32> end_mask(end_mask_i);
  const std::bitset<32> shrink_axis_mask(shrink_axis_mask_i);
  const std::bitset<32> ellipsis_mask(ellipsis_mask_i);
  const std::bitset<32> new_axis_mask(new_axis_mask_i);
  if (ellipsis_mask.none()) {
    CHECK_EQ(topt_count, rank);
  } else if (topt_count < rank) {
    unsigned ellipsis = 0;
    while (!ellipsis_mask[ellipsis] && ellipsis < rank) {
      ++ellipsis;
    }
    if (ellipsis < rank) {
      std::size_t diff = rank - topt_count;
      begin_vec.insert(begin_vec.begin() + ellipsis, diff, 0);
      end_vec.insert(end_vec.begin() + ellipsis, diff, -1);
      stride_vec.insert(stride_vec.begin() + ellipsis, diff, 1);
    }
  }
  for (unsigned i = 0; i < rank; ++i) {
    if (shrink_axis_mask[i]) {
      output_shape[i] = 1;
      continue;
    }
    const uint32_t begin = begin_mask[i] || begin_vec[i] < 0 ? 0 : begin_vec[i];
    const uint32_t end = end_mask[i] || end_vec[i] < 0 ? rank : end_vec[i];
    const uint32_t stride = stride_vec[i];
    output_shape[i] = (begin - end + stride - 1) / stride;
  }
  for (long i = rank - 1; i >= 0; --i) {
    if (new_axis_mask[i]) {
      output_shape.insert(output_shape.begin() + i, 1);
    }
  }
}

tensorflow::Status ConvertStridedSlice(
    Converter& ctx, const tensorflow::NodeDef& node_def,
    const std::vector<TensorOrWeights>& inputs,
    std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 4) {
    return tensorflow::errors::InvalidArgument(
        "StridedSlice op requires 4 inputs, at " + node_def.name());
  }
  const auto& input = inputs.at(0);
  const auto& begin = inputs.at(1);
  const auto& end = inputs.at(2);
  const auto& stride = inputs.at(3);

  const TFAttrs attrs(node_def);
  const auto node_dtype = attrs.get<tensorflow::DataType>("T");
  const auto index_type = attrs.get<tensorflow::DataType>("Index");
  CHECK_EQ(input.dtype(), node_dtype);
  CHECK_EQ(begin.dtype(), index_type);
  CHECK_EQ(end.dtype(), index_type);
  CHECK_EQ(stride.dtype(), index_type);

  const int begin_mask = attrs.get<int>("begin_mask", 0);
  const int end_mask = attrs.get<int>("end_mask", 0);
  const int shrink_axis_mask = attrs.get<int>("shrink_axis_mask", 0);
  const int ellipsis_mask = attrs.get<int>("ellipsis_mask", 0);
  const int new_axis_mask = attrs.get<int>("new_axis_mask", 0);

  if (!begin.is_host() || !end.is_host() || !stride.is_host()) {
    return tensorflow::errors::InvalidArgument(
        "StridedSlice inputs >0 must be host values, at " + node_def.name());
  }

  // Create input host operands
  ctx.set_host_operand_value(begin);
  ctx.set_host_operand_value(end);
  ctx.set_host_operand_value(stride);
  const auto topt_begin_mask = ctx.create_scalar_host_operand(begin_mask);
  const auto topt_end_mask = ctx.create_scalar_host_operand(end_mask);
  const auto topt_shrink_axis_mask = ctx.create_scalar_host_operand(shrink_axis_mask);
  const auto topt_ellipsis_mask = ctx.create_scalar_host_operand(ellipsis_mask);
  const auto topt_new_axis_mask = ctx.create_scalar_host_operand(new_axis_mask);
  VLOG(2) << "StridedSlice begin=" << begin << " end=" << end
          << " stride=" << stride
          << " begin_mask=" << topt_begin_mask << " end_mask=" << topt_end_mask
          << " shrink_mask=" << topt_shrink_axis_mask
          << " ellipsis_mask=" << topt_ellipsis_mask
          << " new_axis_mask=" << topt_new_axis_mask;

  // Create an output operand
  auto output_shape = input.dimensions();
  if (index_type == tensorflow::DataType::DT_INT32) {
    ComputeStridedSliceShape<int32_t>(output_shape, begin, end, stride,
                                      begin_mask, end_mask, shrink_axis_mask,
                                      ellipsis_mask, new_axis_mask);
  } else if (index_type == tensorflow::DataType::DT_INT64) {
    ComputeStridedSliceShape<int64_t>(output_shape, begin, end, stride,
                                      begin_mask, end_mask, shrink_axis_mask,
                                      ellipsis_mask, new_axis_mask);
  } else {
    return tensorflow::errors::Unimplemented(
        "StridedSlice supports only Index type of DT_INT32 and DT_INT64, at " +
        node_def.name());
  }
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(node_dtype, &output_type));
  outputs->emplace_back(output_shape, output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  static constexpr unsigned NB_INPUTS = 9;
  const uint32_t operation_inputs[NB_INPUTS] = {input.get_valid_idx(),
                                          begin.get_valid_idx(),
                                          end.get_valid_idx(),
                                          stride.get_valid_idx(),
                                          topt_begin_mask.get_valid_idx(),
                                          topt_end_mask.get_valid_idx(),
                                          topt_shrink_axis_mask.get_valid_idx(),
                                          topt_ellipsis_mask.get_valid_idx(),
                                          topt_new_axis_mask.get_valid_idx()};
  TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_STRIDED_SLICE,
                                   NB_INPUTS, operation_inputs, 1,
                                   &output.get_valid_idx()));
  return tensorflow::Status::OK();
}

tensorflow::Status ConvertCast(Converter& ctx,
                               const tensorflow::NodeDef& node_def,
                               const std::vector<TensorOrWeights>& inputs,
                               std::vector<TensorOrWeights>* outputs) {
  if (inputs.size() != 1) {
    return tensorflow::errors::FailedPrecondition(
        "Cast op requires single tensor input, at " + node_def.name());
  }
  const auto& input = inputs.at(0);

  const TFAttrs attrs(node_def);
  const auto src_dtype = attrs.get<tensorflow::DataType>("SrcT");
  const auto dst_dtype = attrs.get<tensorflow::DataType>("DstT");
  CHECK_EQ(input.dtype(), src_dtype);

  // Create an output operand
  ANeuralNetworksOperandCode output_type;
  TF_CHECK_OK(ConvertDTypeToTOPTTensorType(dst_dtype, &output_type));
  outputs->emplace_back(input.dimensions(), output_type);
  auto& output = outputs->back();
  TOPT_CHECK_OK(Model_addOperand(ctx.model(), &output.op(), &output.idx()));

  TOPT_CHECK_OK(Model_addOperation(ctx.model(), ANEURALNETWORKS_CAST, 1,
                                   &input.get_valid_idx(), 1,
                                   &output.get_valid_idx()));
  return tensorflow::Status::OK();
}

void Converter::register_op_converters() {
  op_registry_["Const"] = ConvertConst;
  op_registry_["Conv2D"] = ConvertConv2D;
  op_registry_["DepthwiseConv2dNative"] = ConvertConv2D;
  op_registry_["MaxPool"] = ConvertPool;
  op_registry_["AvgPool"] = ConvertPool;
  op_registry_["MatMul"] = ConvertMatMul;
  op_registry_["Transpose"] = ConvertTranspose;
  op_registry_["Identity"] = ConvertIdentity;
  op_registry_["Snapshot"] = ConvertIdentity;
  op_registry_["BiasAdd"] = ConvertBinary;
  op_registry_["Add"] = ConvertBinary;
  op_registry_["Mul"] = ConvertBinary;
  op_registry_["Sub"] = ConvertBinary;
  op_registry_["Div"] = ConvertBinary;
  op_registry_["RealDiv"] = ConvertBinary;
  op_registry_["Minimum"] = ConvertBinary;
  op_registry_["Maximum"] = ConvertBinary;
  op_registry_["Relu"] = ConvertUnary;
  op_registry_["Relu1"] = ConvertUnary;
  op_registry_["Relu6"] = ConvertUnary;
  op_registry_["Exp"] = ConvertUnary;
  op_registry_["Sqrt"] = ConvertUnary;
  op_registry_["Rsqrt"] = ConvertUnary;
  op_registry_["Softmax"] = ConvertSoftmax;
  op_registry_["Reshape"] = ConvertReshape;
  op_registry_["Squeeze"] = ConvertSqueeze;
  op_registry_["Slice"] = ConvertSlice;
  op_registry_["StridedSlice"] = ConvertStridedSlice;
  op_registry_["ConcatV2"] = ConvertConcat;
  op_registry_["Pack"] = ConvertPack;
  op_registry_["Stack"] = ConvertPack;
  op_registry_["Unpack"] = ConvertUnpack;
  op_registry_["Unstack"] = ConvertUnpack;
  op_registry_["FusedBatchNorm"] = ConvertFusedBatchNorm;
  op_registry_["FusedBatchNormV2"] = ConvertFusedBatchNorm;
  op_registry_["Cast"] = ConvertCast;
}

}  // namespace

tensorflow::Status ConvertSubGraphToTensorOptModel(
    const std::list<tensorflow::Node*>& order,
    const std::vector<string>& input_names,
    const std::vector<TensorShape>& input_shapes,
    const std::vector<DataType>& input_types,
    const std::vector<string>& output_names,
    const std::vector<DataType>& output_types,
    std::vector<std::vector<uint8_t>>& weight_store,
    ANeuralNetworksModel** topt_model) {
  TOPT_CHECK_OK(Model_create(topt_model));

  Converter converter(*topt_model, weight_store);
  std::vector<uint32_t> topt_inputs;
  CHECK_EQ(input_names.size(), input_shapes.size());
  for (unsigned i = 0; i < input_names.size(); ++i) {
    auto& input_name = input_names[i];
    auto& input_shape = input_shapes[i];
    auto& input_type = input_types[i];
    VLOG(2) << "Parsing input '" << input_name << "'";
    std::vector<uint32_t> input_dim;
    for (const auto& dim : input_shape) {
      input_dim.push_back(dim.size);
    }

    TensorOrWeights input_tensor(input_dim);
    input_tensor.set_tensor_type(input_type);
    // TensorOrWeights does not need the name input_name.c_str() as
    // TensorOpt doesn't use the names as identifiers during the execution
    TOPT_CHECK_OK(Model_addOperand(converter.model(), &input_tensor.op(),
                                   &input_tensor.idx()));
    topt_inputs.push_back(input_tensor.get_valid_idx());

    if (!converter.insert_input_tensor(input_name, input_tensor)) {
      return tensorflow::errors::AlreadyExists(
          "Output tensor already exists for op: " + input_name);
    }
  }
  TOPT_CHECK_OK(Model_identifyInputs(converter.model(), topt_inputs.size(),
                                     topt_inputs.data()));

  for (const tensorflow::Node* node : order) {
    const tensorflow::NodeDef& node_def = node->def();
    VLOG(2) << "Converting node: " << node_def.name() << " (" << node_def.op()
            << ")";
    TF_RETURN_IF_ERROR(converter.convert_node(node_def));
  }

  VLOG(2) << "Finished conversion";

  // Gather output metadata
  std::vector<uint32_t> topt_outputs;
  for (unsigned i = 0; i < output_names.size(); ++i) {
    auto& output_name = output_names[i];
    auto& output_type = output_types[i];
    VLOG(2) << "Parsing output '" << output_name << "'";
    auto output_tensor = converter.get_tensor(output_name);
    topt_outputs.push_back(output_tensor.get_valid_idx());
    CHECK_EQ(output_tensor.dtype(), output_type);
  }
  TOPT_CHECK_OK(Model_identifyOutputs(converter.model(), topt_outputs.size(),
                                      topt_outputs.data()));
  TOPT_CHECK_OK(Model_finish(converter.model()));

  return tensorflow::Status::OK();
}

}  // namespace convert
}  // namespace tensoropt
}  // namespace tensorflow

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
