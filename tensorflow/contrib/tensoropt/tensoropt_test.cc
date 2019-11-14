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

#include <cassert>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
#include "tensorflow/contrib/tensoropt/api/runtime_api.h"
#include "tensorflow/contrib/tensoropt/api/topt_lib_api.h"

namespace tensorflow {
namespace {

// Add a single operation as the graph optimizer would do i.e. without prior
// knowledge of what was added before and what will be added next.
// This operation simply adds a scalar on the host to a scalar on the device
// and outputs a scalar on the device.
void AppendBiasAddOperation(ANeuralNetworksModel* model, float bias) {
  assert(ANeuralNetworksModel_canAddOperation(
      model, nullptr, 1, ANeuralNetworksOperationType::ADD));
  uint32_t operation_count = ANeuralNetworksModel_getOperationCount(model);
  uint32_t operand_in_idx;
  if (operation_count == 0) {
    // If this is the first operation added, create as many inputs as needed
    ANeuralNetworksOperandType operand_in;
    operand_in.type =
        ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_FLOAT32;
    operand_in.dimensionCount = 0;
    operand_in.dimensions = nullptr;
    TOPT_CHECK_OK(Model_addOperand(model, &operand_in, &operand_in_idx));
  } else {
    // An operation has already been added so use the previous outputs as
    // inputs
    uint32_t prev_operation_idx = operation_count - 1;
    uint32_t prev_operation_output_count;
    TOPT_CHECK_OK(Model_getOperationOutputCount(model, prev_operation_idx,
                                                &prev_operation_output_count));
    assert(prev_operation_output_count == 1);
    const uint32_t* prev_operation_outputs;
    TOPT_CHECK_OK(Model_getOperationOutputs(model, prev_operation_idx,
                                            &prev_operation_outputs));
    operand_in_idx = prev_operation_outputs[0];
  }

  // Create a constant bias operand and set its value
  ANeuralNetworksOperandType operand_bias;
  operand_bias.type = ANeuralNetworksOperandCode::ANEURALNETWORKS_FLOAT32;
  operand_bias.dimensionCount = 0;
  operand_bias.dimensions = nullptr;
  uint32_t operand_bias_idx;
  ANeuralNetworksModel_addOperand(model, &operand_bias, &operand_bias_idx);
  ANeuralNetworksModel_setOperandValue(model, operand_bias_idx, &bias,
                                       sizeof(float));

  // Create an output operand
  ANeuralNetworksOperandType operand_out;
  operand_out.type = ANeuralNetworksOperandCode::ANEURALNETWORKS_TENSOR_FLOAT32;
  operand_out.dimensionCount = 0;
  operand_out.dimensions = nullptr;
  uint32_t operand_out_idx;
  TOPT_CHECK_OK(Model_addOperand(model, &operand_out, &operand_out_idx));

  uint32_t operation_inputs[] = {operand_in_idx, operand_bias_idx};
  TOPT_CHECK_OK(Model_addOperation(
      model, ANeuralNetworksOperationType::ANEURALNETWORKS_ADD, 2,
      operation_inputs, 1, &operand_out_idx));
}

// Create a network to compute y=x+3+2.
ANeuralNetworksModel* CreateModel() {
  ANeuralNetworksModel* model = nullptr;
  TOPT_CHECK_OK(Model_create(&model));

  AppendBiasAddOperation(model, 3.0f);
  AppendBiasAddOperation(model, 2.0f);

  // Identify model's inputs and outputs
  uint32_t operation_count = ANeuralNetworksModel_getOperationCount(model);
  assert(operation_count > 0);
  // inputs
  uint32_t first_operation_input_count;
  TOPT_CHECK_OK(
      Model_getOperationInputCount(model, 0, &first_operation_input_count));
  assert(first_operation_input_count > 0);
  const uint32_t* first_operation_inputs = nullptr;
  TOPT_CHECK_OK(Model_getOperationInputs(model, 0, &first_operation_inputs));
  // outputs
  uint32_t prev_operation_idx = operation_count - 1;
  uint32_t prev_operation_output_count;
  TOPT_CHECK_OK(Model_getOperationOutputCount(model, prev_operation_idx,
                                              &prev_operation_output_count));
  assert(prev_operation_output_count > 0);
  const uint32_t* prev_operation_outputs = nullptr;
  TOPT_CHECK_OK(Model_getOperationOutputs(model, prev_operation_idx,
                                          &prev_operation_outputs));
  // identify
  TOPT_CHECK_OK(Model_identifyInputsAndOutputs(
      model, first_operation_input_count, first_operation_inputs,
      prev_operation_output_count, prev_operation_outputs));

  TOPT_CHECK_OK(Model_finish(model));

  return model;
}

ANeuralNetworksCompilation* CompileModel(ANeuralNetworksModel* model,
                                         ANeuralNetworksDevice* device) {
  ANeuralNetworksCompilation* compilation = nullptr;
  TOPT_CHECK_OK(Compilation_createForDevices(model, &device, 1, &compilation));
  TOPT_CHECK_OK(Compilation_finish(compilation));
  return compilation;
}

void Execute(ANeuralNetworksCompilation* compilation, const float* input,
             float* output) {
  ANeuralNetworksExecution* execution = nullptr;
  TOPT_CHECK_OK(Execution_create(compilation, &execution));

  // Create device Memory
  tensoropt_buffer_t in_buf(cl::sycl::range<1>(sizeof(float)));
  tensoropt_buffer_t out_buf(cl::sycl::range<1>(sizeof(float)));
  ANeuralNetworksMemory* in_mem = nullptr;
  ANeuralNetworksMemory* out_mem = nullptr;
  TOPT_CHECK_OK(Memory_createFromBuffer(in_buf, &in_mem));
  TOPT_CHECK_OK(Memory_createFromBuffer(out_buf, &out_mem));

  // Set inputs and outputs
  assert(ANeuralNetworksExecution_getIdentifiedInputCount(execution) == 1);
  assert(ANeuralNetworksExecution_getIdentifiedOutputCount(execution) == 1);
  auto inputs_idx = ANeuralNetworksExecution_getIdentifiedInputs(execution);
  auto outputs_idx = ANeuralNetworksExecution_getIdentifiedOutputs(execution);
  TOPT_CHECK_OK(Execution_setInputFromMemory(execution, inputs_idx[0], nullptr,
                                             in_mem, 0, 1));
  TOPT_CHECK_OK(Execution_setOutputFromMemory(execution, outputs_idx[0],
                                              nullptr, out_mem, 0, 1));

  // Copy the input to the device, execute the network, and copy the output
  // back
  using mode = cl::sycl::access::mode;
  {
    auto host_acc = in_buf.get_access<mode::discard_write>();
    host_acc[0] = *input;
  }
  ANeuralNetworksEvent* event = nullptr;
  TOPT_CHECK_OK(Execution_startCompute(execution, &event));
  TOPT_CHECK_OK(Event_wait(event));
  {
    auto host_acc = out_buf.get_access<mode::read>();
    *output = host_acc[0];
  }

  ANeuralNetworksEvent_free(event);
  ANeuralNetworksMemory_free(in_mem);
  ANeuralNetworksMemory_free(out_mem);
  TOPT_CHECK_OK(Execution_free(execution));
}

TEST(TensorOptTest, BasicFunctions) {
  auto model = CreateModel();
  cl::sycl::queue queue;
  ANeuralNetworksDevice* device = nullptr;
  TOPT_CHECK_OK(Device_create(&queue, &device));
  auto compilation = CompileModel(model, device);
  TOPT_CHECK_OK(Model_free(model));
  float input = 1234;
  float output;
  Execute(compilation, &input, &output);
  EXPECT_NEAR(output, input + 3 + 2, 1e-5);

  TOPT_CHECK_OK(Compilation_free(compilation));
}

}  // namespace
}  // namespace tensorflow

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
