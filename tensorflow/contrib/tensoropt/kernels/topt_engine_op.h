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

#ifndef TENSORFLOW_CONTRIB_TENSOROPT_KERNELS_TOPT_ENGINE_OP_H_
#define TENSORFLOW_CONTRIB_TENSOROPT_KERNELS_TOPT_ENGINE_OP_H_

#include <list>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/graph.h"

#if defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT
#include "tensorflow/contrib/tensoropt/api/runtime_api.h"
#include "tensorflow/contrib/tensoropt/api/topt_lib_api.h"
#include "tensorflow/contrib/tensoropt/kernels/hash_shapes.h"

namespace tensorflow {
namespace tensoropt {

class TOPTEngineOp : public OpKernel {
 public:
  explicit TOPTEngineOp(OpKernelConstruction* context);
  ~TOPTEngineOp() = default;

  void Compute(OpKernelContext* context) override;

 private:
  struct DestroyDevice {
    void operator()(ANeuralNetworksDevice* d) { ANeuralNetworksDevice_free(d); }
  };
  struct DestroyCompilation {
    void operator()(ANeuralNetworksCompilation* c) {
      ANeuralNetworksCompilation_free(c);
    }
  };
  struct DestroyExecution {
    void operator()(ANeuralNetworksExecution* e) {
      ANeuralNetworksExecution_free(e);
    }
  };
  struct DestroyMemory {
    void operator()(ANeuralNetworksMemory* m) { ANeuralNetworksMemory_free(m); }
  };

  struct ExecutionCache {
    std::unique_ptr<ANeuralNetworksCompilation, DestroyCompilation>
        topt_compilation_ptr;
    std::unique_ptr<ANeuralNetworksExecution, DestroyExecution>
        topt_execution_ptr;
    std::vector<std::unique_ptr<ANeuralNetworksMemory, DestroyMemory>>
      topt_memory;
    std::vector<PersistentTensor> output_tensors;
  };

  // Cache the executions for specific input sizes
  std::unordered_map<std::vector<TensorShape>, ExecutionCache> executions_;
  std::unique_ptr<ANeuralNetworksDevice, DestroyDevice> topt_device_ptr_;
  std::vector<string> input_names_;
  std::vector<tensorflow::DataType> input_types_;
  std::vector<string> output_names_;
  std::vector<tensorflow::DataType> output_types_;
  std::unique_ptr<tensorflow::Graph> graph_ptr_;
  std::list<tensorflow::Node*> graph_order_;

  ANeuralNetworksMemory* CreateTOPTMemoryFromTensor(
      const Eigen::SyclDevice& eigen_device, const Tensor& t, uint32_t& offset,
      uint32_t& length);

  ANeuralNetworksMemory* CreateTOPTMemoryFromTensorSafe(
      const Eigen::SyclDevice& eigen_device, const Tensor& t, uint32_t ops_idx,
      const std::vector<ANeuralNetworksOperandType>& ops,
      const std::string& name, uint32_t& offset, uint32_t& length);
};

}  // namespace tensoropt
}  // namespace tensorflow

#endif  // defined(TENSORFLOW_USE_SYCL) && TF_SYCL_USE_TENSOROPT

#endif  // TENSORFLOW_CONTRIB_TENSOROPT_KERNELS_TOPT_ENGINE_OP_H_
