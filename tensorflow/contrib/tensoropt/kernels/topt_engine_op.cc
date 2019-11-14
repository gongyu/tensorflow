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
#include "tensorflow/contrib/tensoropt/kernels/topt_engine_op.h"

#include "tensorflow/contrib/tensoropt/kernels/convert_nodes.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
#include <SYCL/codeplay.hpp>

namespace tensorflow {
namespace tensoropt {

using ::tensorflow::str_util::StringReplace;
using ::tensorflow::str_util::StrContains;
using ::tensorflow::strings::StrCat;

TOPTEngineOp::TOPTEngineOp(OpKernelConstruction* context) : OpKernel(context) {
  OP_REQUIRES_OK(context, context->GetAttr("input_names", &input_names_));
  OP_REQUIRES_OK(context, context->GetAttr("InT", &input_types_));
  OP_REQUIRES_OK(context, context->GetAttr("output_names", &output_names_));
  OP_REQUIRES_OK(context, context->GetAttr("OutT", &output_types_));

  string proto_graph;
  OP_REQUIRES_OK(context, context->GetAttr("proto_graph", &proto_graph));
  tensorflow::GraphDef graph_def;
  graph_def.ParseFromString(proto_graph);
  // Add fake placeholder nodes for every input
  auto device_name = context->def().device();
  auto prefix = StrCat(context->def().name(), "/");
  for (unsigned i = 0; i < input_names_.size(); ++i) {
    auto& input_name = input_names_[i];
    auto node_name = input_name;
    // Placeholder can have only one input so some more work needs to be done
    // if a node has an input using a port_id > 0 (i.e. input:1)
    // In this case we create a Placeholder with the name prefix/input_1 and replace
    // any node that uses input:1 to prefix/input_1
    if (StrContains(input_name, ":")) {
      // Prefix the node name by the TOPTEngineOp node name to make sure
      // there are no name conflicts
      node_name = StrCat(prefix,
          StringReplace(input_name, ":", "_", true));
      for (auto& node_def : *graph_def.mutable_node()) {
        for (auto& node_input : *node_def.mutable_input()) {
          if (node_input == input_name) {
            node_input = node_name;
            input_name = node_name;
          }
        }
      }
    }
    NodeDef* node = graph_def.add_node();
    node->set_name(node_name);
    node->set_op("Placeholder");
    node->set_device(device_name);
    AddNodeAttr("dtype", input_types_[i], node);
  }

  tensorflow::FunctionLibraryDefinition flib(tensorflow::OpRegistry::Global(),
                                             graph_def.library());
  graph_ptr_ = std::unique_ptr<tensorflow::Graph>(
      new tensorflow::Graph(std::move(flib)));
  TF_CHECK_OK(ConvertGraphDefToGraph(GraphConstructorOptions(), graph_def,
                                     graph_ptr_.get()));

  std::vector<tensorflow::Node*> order_vec;
  tensorflow::GetPostOrder(*graph_ptr_, &order_vec);
  for (auto node : order_vec) {
    // Ignore _SOURCE, _SINK and fake Placeholder
    if (node->IsOp() && node->type_string() != "Placeholder")
      graph_order_.push_front(node);
  }
}

ANeuralNetworksMemory* TOPTEngineOp::CreateTOPTMemoryFromTensor(
    const Eigen::SyclDevice& eigen_device, const Tensor& t, uint32_t& offset,
    uint32_t& length) {
  auto buffer_ptr = DMAHelper::base(&t);
  auto sycl_buffer = eigen_device.get_sycl_buffer(buffer_ptr);
  offset = eigen_device.get_offset(buffer_ptr);
  length = t.TotalBytes();
  ANeuralNetworksMemory* memory = nullptr;
  TOPT_CHECK_OK(Memory_createFromBuffer(sycl_buffer, &memory));
  return memory;
}

ANeuralNetworksMemory* TOPTEngineOp::CreateTOPTMemoryFromTensorSafe(
    const Eigen::SyclDevice& eigen_device, const Tensor& t, uint32_t ops_idx,
    const std::vector<ANeuralNetworksOperandType>& ops, const std::string& name,
    uint32_t& offset, uint32_t& length) {
  const auto& op = ops[ops_idx];
  auto op_dtype = ConvertTOPTTypeToDType(op.type);
  if (op_dtype != t.dtype()) {
    LOG(FATAL) << "Failed to execute TensorOpt node, " << name << " #"
               << ops_idx << " expected type " << DataTypeString(op_dtype)
               << " but got " << DataTypeString(t.dtype());
  }
  TensorShape expected_shape;
  TF_CHECK_OK(
      TensorShapeUtils::MakeShape(reinterpret_cast<const int32*>(op.dimensions),
                                  op.dimensionCount, &expected_shape));
  // Only check for the number of elements here as the input can be an empty
  // shape which will then be expected as a shape with one dimension, one
  // element.
  if (expected_shape.num_elements() != t.shape().num_elements()) {
    LOG(FATAL) << "Failed to execute TensorOpt node, " << name << " #"
               << ops_idx << " expected shape " << expected_shape << " but got "
               << t.shape();
  }
  return CreateTOPTMemoryFromTensor(eigen_device, t, offset, length);
}

void TOPTEngineOp::Compute(OpKernelContext* context) {
  auto input_count = static_cast<uint32_t>(context->num_inputs());
  auto output_count = static_cast<uint32_t>(context->num_outputs());
  std::vector<TensorShape> input_shapes;
  input_shapes.reserve(input_count);
  for (unsigned i = 0; i < input_count; ++i) {
    input_shapes.push_back(context->input(i).shape());
  }
  auto exec_it = executions_.find(input_shapes);
  ExecutionCache* exec_ptr = nullptr;
  auto& eigen_device = context->eigen_sycl_device();

  if (exec_it == executions_.end()) {
    auto& exec = executions_[input_shapes];
    exec_ptr = &exec;

    ANeuralNetworksDevice* topt_device = nullptr;
    TOPT_CHECK_OK(
        Device_create(&eigen_device.sycl_queue(), false, &topt_device));
    topt_device_ptr_.reset(topt_device);

    // Convert the model
    auto node_name = context->op_kernel().name();
    LOG(INFO) << "Converting model " << node_name << " containing "
              << graph_order_.size() << " nodes";
    if (VLOG_IS_ON(2)) {
      std::stringstream input_shapes_ss;
      if (input_count > 0) {
        input_shapes_ss << input_shapes[0];
        for (unsigned i = 1; i < input_count; ++i) {
          input_shapes_ss << ", " << input_shapes[i];
        }
      }
      VLOG(2) << "Input shapes: " << input_shapes_ss.str();
    }
    std::vector<std::vector<uint8_t>> weight_store;
    ANeuralNetworksModel* topt_model = nullptr;
    TF_CHECK_OK(convert::ConvertSubGraphToTensorOptModel(
        graph_order_, input_names_, input_shapes, input_types_, output_names_,
        output_types_, weight_store, &topt_model));

    // Compile the model
    VLOG(2) << "Compiling model " << node_name;
    ANeuralNetworksCompilation* topt_compilation = nullptr;
    TOPT_CHECK_OK(Compilation_createForDevices(topt_model, &topt_device, 1,
                                               &topt_compilation));
    TOPT_CHECK_OK(Compilation_finish(topt_compilation));
    exec.topt_compilation_ptr.reset(topt_compilation);
    ANeuralNetworksModel_free(topt_model);

    ANeuralNetworksExecution* topt_execution = nullptr;
    TOPT_CHECK_OK(Execution_create(topt_compilation, &topt_execution));
    exec.topt_execution_ptr.reset(topt_execution);

    // Create inputs
    uint32_t topt_model_input_count =
        ANeuralNetworksExecution_getIdentifiedInputCount(topt_execution);
    if (topt_model_input_count != input_count) {
      LOG(FATAL) << "Failed to execute TensorOpt node, expected "
                 << topt_model_input_count << " inputs but got " << input_count;
    }
    // Get the identified operands for debug check
    std::vector<ANeuralNetworksOperandType> input_ops(topt_model_input_count);
    TOPT_CHECK_OK(
        Execution_getIdentifiedInputs(topt_execution, input_ops.data()));
    exec.topt_memory.resize(input_count + output_count);
    for (uint32_t i = 0; i < input_count; ++i) {
      auto input = context->input(i);
      uint32_t offset;
      uint32_t length;
      auto memory = CreateTOPTMemoryFromTensorSafe(
          eigen_device, input, i, input_ops, "input", offset, length);
      exec.topt_memory[i].reset(memory);
      TOPT_CHECK_OK(Execution_setInputFromMemory(topt_execution, i, nullptr,
                                                 memory, offset, length));
    }

    // Create and bind outputs
    uint32_t topt_model_output_count =
        ANeuralNetworksExecution_getIdentifiedOutputCount(topt_execution);
    if (topt_model_output_count != output_count) {
      LOG(FATAL) << "Failed to execute TensorOpt node, expected "
                 << topt_model_output_count << " outputs but got "
                 << output_count;
    }
    std::vector<ANeuralNetworksOperandType> output_ops(topt_model_output_count);
    TOPT_CHECK_OK(
        Execution_getIdentifiedOutputs(topt_execution, output_ops.data()));
    exec.output_tensors.reserve(output_count);
    std::stringstream output_shapes_ss;
    for (uint32_t i = 0; i < output_count; ++i) {
      exec.output_tensors.emplace_back();
      auto& output = exec.output_tensors.back();
      const auto& out_op = output_ops[i];
      TensorShape output_shape;
      for (uint32_t j = 0; j < output_ops[i].dimensionCount; ++j)
        output_shape.AddDim(output_ops[i].dimensions[j]);
      if (VLOG_IS_ON(2)) {
        if (i > 0) {
          output_shapes_ss << ", ";
        }
        output_shapes_ss << output_shape;
      }
      Tensor* out_tensor = nullptr;
      AllocatorAttributes attr;
      attr.set_on_host(false);
      attr.set_gpu_compatible(true);
      OP_REQUIRES_OK(
          context, context->allocate_persistent(output_types_[i], output_shape,
                                                &output, &out_tensor, attr));
      uint32_t offset;
      uint32_t length;
      auto memory = CreateTOPTMemoryFromTensorSafe(
          eigen_device, *out_tensor, i, output_ops, "output", offset, length);
      exec.topt_memory[input_count + i].reset(memory);
      TOPT_CHECK_OK(Execution_setOutputFromMemory(topt_execution, i, nullptr,
                                                  memory, offset, length));
    }
    VLOG(2) << "Output shapes: " << output_shapes_ss.str();
  } else {  // model is already compiled
    exec_ptr = &exec_it->second;

    // Update inputs
    auto& exec = *exec_ptr;
    for (uint32_t i = 0; i < input_count; ++i) {
      auto memory = exec.topt_memory[i].get();
      auto input = context->input(i);
      auto buffer_ptr = DMAHelper::base(&input);
      auto buffer = eigen_device.get_sycl_buffer(buffer_ptr);
      TOPT_CHECK_OK(Memory_resetBuffer(memory, buffer));
    }
  }

  // Set TF outputs
  for (unsigned i = 0; i < output_count; ++i) {
    context->set_output(i, *exec_ptr->output_tensors[i].AccessTensor(context));
  }

  // No need to wait on the outputed event as no outputs are on the host
  // The SYCL runtime will take care of the dependencies
  TOPT_CHECK_OK(Execution_startCompute(exec_ptr->topt_execution_ptr.get(), nullptr));
}

REGISTER_KERNEL_BUILDER(Name("TOPTEngineOp").Device(DEVICE_SYCL), TOPTEngineOp);

}  // namespace tensoropt
}  // namespace tensorflow

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
